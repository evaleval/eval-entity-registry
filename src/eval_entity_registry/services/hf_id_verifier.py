"""Runtime HF repo-id verifier, the resolver chain's injected `hf_id_checker`.

Lookup order: in-memory cache (positive 24 h; negative 1 h so a new repo
becomes resolvable without a restart) -> cron-built `hub_stats_index` ->
live Hub API GET. Every live-path guard (single-flight, concurrency cap,
per-minute budget, circuit breaker, `hf_live_id_check_enabled`) is
non-blocking and answers None uncached rather than pinning a thread. Only
an authoritative 404 is negative-cached, so any other non-answer falls
through to alias/fuzzy and a later call may retry.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Optional

from eval_entity_resolver.models import HfIdHit, looks_like_hf_id

POSITIVE_TTL_SECONDS = 24 * 3600.0
NEGATIVE_TTL_SECONDS = 3600.0
MAX_CONCURRENT_LIVE = 4
LIVE_BUDGET_PER_MINUTE = 60
BREAKER_FAILURE_THRESHOLD = 5
BREAKER_COOLDOWN_SECONDS = 300.0

# Sentinel stored in the cache for an authoritative "not on the Hub".
_MISS = ""

# Soft cap; the oldest tenth is evicted past it so unique never-repeated ids
# cannot grow the dict unboundedly.
_MAX_CACHE_ENTRIES = 100_000


class HfIdVerifier:
    """Checks whether a raw string is a valid HF model repo id.

    `index_provider` is re-read on every check so the service can rebuild the
    index without invalidation coupling. `clock` must be monotonic."""

    def __init__(
        self,
        index_provider: Callable[[], dict],
        *,
        positive_ttl: float = POSITIVE_TTL_SECONDS,
        negative_ttl: float = NEGATIVE_TTL_SECONDS,
        max_concurrent: int = MAX_CONCURRENT_LIVE,
        budget_per_minute: int = LIVE_BUDGET_PER_MINUTE,
        breaker_threshold: int = BREAKER_FAILURE_THRESHOLD,
        breaker_cooldown: float = BREAKER_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._index_provider = index_provider
        self._positive_ttl = positive_ttl
        self._negative_ttl = negative_ttl
        self._budget_per_minute = budget_per_minute
        self._breaker_threshold = breaker_threshold
        self._breaker_cooldown = breaker_cooldown
        self._clock = clock
        # cache: lowercased raw id -> (expires_at, hf_true_id | _MISS)
        self._cache: dict[str, tuple[float, str]] = {}
        self._cache_lock = threading.Lock()
        self._live_semaphore = threading.BoundedSemaphore(max_concurrent)
        self._in_flight: set[str] = set()
        self._in_flight_lock = threading.Lock()
        self._budget_window: deque[float] = deque()
        self._budget_lock = threading.Lock()
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0
        self._breaker_lock = threading.Lock()
        self._hf_api = None

    # ------------------------------------------------------------------
    # Public entry point (the resolver's hf_id_checker callable)
    # ------------------------------------------------------------------

    def check(self, raw_value: str) -> Optional[HfIdHit]:
        if not looks_like_hf_id(raw_value):
            return None
        key = raw_value.lower()

        cached = self._cache_get(key)
        if cached is not None:
            if cached == _MISS:
                return None
            return self._hit(raw_value, cached, "hf_live")

        index_hit = self._check_index(raw_value)
        if index_hit is not None:
            return index_hit

        return self._check_live(raw_value, key)

    # ------------------------------------------------------------------
    # Tiers
    # ------------------------------------------------------------------

    def _check_index(self, raw_value: str) -> Optional[HfIdHit]:
        try:
            index = self._index_provider() or {}
        except Exception:
            return None
        if not index:
            return None
        from eval_entity_registry.services.hub_stats import normalize as _hsnorm

        entry = index.get(_hsnorm(raw_value))
        if not entry:
            return None
        if isinstance(entry, dict):
            # Two repos share this id_norm: only a case-insensitive verbatim
            # match may answer; the live tier disambiguates the rest.
            hf_id = entry.get(raw_value.lower())
            if not hf_id:
                return None
            return self._hit(raw_value, hf_id, "hub_stats_index")
        return self._hit(raw_value, entry, "hub_stats_index")

    def _check_live(self, raw_value: str, key: str) -> Optional[HfIdHit]:
        from eval_entity_registry.config import settings

        if not settings.hf_live_id_check_enabled:
            return None
        with self._breaker_lock:
            if self._clock() < self._breaker_open_until:
                return None
        # Single-flight: a concurrent duplicate answers unknown immediately.
        with self._in_flight_lock:
            if key in self._in_flight:
                return None
            self._in_flight.add(key)
        acquired = self._live_semaphore.acquire(blocking=False)
        try:
            if not acquired:
                return None
            # Budget is taken last so a single-flight duplicate or semaphore
            # miss never burns a token a genuine lookup could have used.
            if not self._budget_take(self._clock()):
                return None
            return self._live_lookup(raw_value, key)
        finally:
            if acquired:
                self._live_semaphore.release()
            with self._in_flight_lock:
                self._in_flight.discard(key)

    def _live_lookup(self, raw_value: str, key: str) -> Optional[HfIdHit]:
        from eval_entity_registry.config import settings

        try:
            from huggingface_hub import HfApi
            from huggingface_hub.errors import (
                GatedRepoError,
                HFValidationError,
                RepositoryNotFoundError,
            )
        except Exception:
            return None
        try:
            if self._hf_api is None:
                with self._cache_lock:
                    if self._hf_api is None:
                        self._hf_api = HfApi(token=settings.hf_token or None)
            info = self._hf_api.model_info(
                raw_value, timeout=settings.hf_live_id_check_timeout
            )
        except GatedRepoError:
            # A gated repo exists (metadata withheld, so the raw spelling is
            # the casing). GatedRepoError subclasses RepositoryNotFoundError
            # and must be caught first or every gated repo is negative-cached.
            self._record_success()
            self._cache_put(key, raw_value, self._positive_ttl)
            return self._hit(raw_value, raw_value, "hf_live")
        except RepositoryNotFoundError:
            # Authoritative miss; a private repo we cannot see counts as one.
            self._record_success()
            self._cache_put(key, _MISS, self._negative_ttl)
            return None
        except HFValidationError:
            # Client-side rejection of a malformed value (e.g. an inner space):
            # not a transport failure, so it must not trip the breaker.
            self._cache_put(key, _MISS, self._negative_ttl)
            return None
        except Exception:
            self._record_failure()
            return None
        hf_id = getattr(info, "id", None) or raw_value
        self._record_success()
        self._cache_put(key, hf_id, self._positive_ttl)
        return self._hit(raw_value, hf_id, "hf_live")

    # ------------------------------------------------------------------
    # Guards and bookkeeping
    # ------------------------------------------------------------------

    def _hit(self, raw_value: str, hf_id: str, source: str) -> HfIdHit:
        # HF repo ids are case-insensitively unique, so a case-insensitive
        # match is exact; separator collapse or a rename redirect is not.
        return HfIdHit(
            hf_id=hf_id,
            verbatim=raw_value.lower() == hf_id.lower(),
            source=source,
        )

    def _cache_get(self, key: str) -> Optional[str]:
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if self._clock() >= expires_at:
                del self._cache[key]
                return None
            return value

    def _cache_put(self, key: str, value: str, ttl: float) -> None:
        with self._cache_lock:
            if len(self._cache) >= _MAX_CACHE_ENTRIES:
                # Insertion order makes the first entries the oldest.
                for stale in list(self._cache)[: _MAX_CACHE_ENTRIES // 10]:
                    del self._cache[stale]
            self._cache[key] = (self._clock() + ttl, value)

    def _budget_take(self, now: float) -> bool:
        with self._budget_lock:
            while self._budget_window and now - self._budget_window[0] >= 60.0:
                self._budget_window.popleft()
            if len(self._budget_window) >= self._budget_per_minute:
                return False
            self._budget_window.append(now)
            return True

    def _record_failure(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._breaker_threshold:
                self._breaker_open_until = self._clock() + self._breaker_cooldown
                self._consecutive_failures = 0

    def _record_success(self) -> None:
        with self._breaker_lock:
            self._consecutive_failures = 0
