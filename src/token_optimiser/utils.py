import logging
from typing import Iterable

logger = logging.getLogger("token_optimiser")
logging.basicConfig(level=logging.INFO)

def estimate_tokens(text: str) -> int:
    # lightweight estimator: whitespace tokens + punctuation heuristic
    if not text:
        return 0
    return max(1, len(text.split()))

def total_tokens(entries: Iterable[str]) -> int:
    return sum(estimate_tokens(t) for t in entries)
