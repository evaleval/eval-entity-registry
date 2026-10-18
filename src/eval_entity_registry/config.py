import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    local_mode: bool = False
    fixtures_path: str = "./fixtures"
    hf_dataset_repo: str = ""
    hf_token: str = ""
    resolver_auto_merge_threshold: float = 0.85
    read_only: bool = False
    hf_log_bucket: str = ""
    log_flush_interval_seconds: int = 300
    # Live hub-stats enrichment of auto-created model drafts; every lookup
    # hits HF, so disable for tests and offline dev.
    hub_stats_lookup_enabled: bool = True
    # Live Hub API existence check (HfIdVerifier `hf_live` tier) for HF-shaped
    # model ids missing from the local hub_stats_index; off = index + cache only.
    hf_live_id_check_enabled: bool = True
    # Timeout (seconds) for a single live Hub API id check.
    hf_live_id_check_timeout: float = 5.0


settings = Settings()

# Libraries such as `datasets.load_dataset` read HF_TOKEN from the environment,
# not from settings.
if settings.hf_token and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = settings.hf_token
