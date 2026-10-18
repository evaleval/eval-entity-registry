from typing import Optional

from eval_entity_resolver.normalization import normalize


def normalized_match(
    raw_value: str,
    entity_type: str,
    alias_store,
    source_config: Optional[str] = None,
) -> Optional[str]:
    """Look the normalized input up in the normalized alias index (scoped
    aliases included only when ``source_config`` is given).

    No org-agreement guard, unlike fuzzy: the WHOLE normalized string must
    match, so an `org/name` raw only hits the same developer modulo
    case/separator (`ai21labs/x` -> `ai21-labs/x`); a different developer's org
    token would change the string. A bare leaf shared by two developers is
    deduped to one owner at seed time.
    """
    norm = normalize(raw_value)
    lookup = alias_store.get_normalized_lookup(entity_type, source_config)
    return lookup.get(norm)
