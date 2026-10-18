import re


def normalize(value: str) -> str:
    """Lowercase, strip, collapse all separators (space/_/-/slash) to one space.

    ``/`` counts as a separator so ``tau-bench-2/airline`` and
    ``tau-bench-2_airline`` normalize identically. Dots between digits become
    spaces so ``4.5`` and ``4-5`` agree. ``+`` is preserved: ``MBPP+`` and
    ``HumanEval+`` are distinct from their non-``+`` counterparts, and dropping
    it would collide them in the normalized index.
    """
    value = value.lower()
    value = value.strip()
    value = re.sub(r"(?<=\d)\.(?=\d)", " ", value)
    value = re.sub(r"[^\w\s\-/+]", "", value)
    value = re.sub(r"[\s_\-/]+", " ", value).strip()
    return value
