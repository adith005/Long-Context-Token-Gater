from typing import List
from .models import MemoryEntry
# attention-based gating will call an LLM/attention scorer; here is a pluggable stub.

def attention_score_stub(query: str, entries: List[MemoryEntry]) -> List[tuple]:
    """
    Returns list of (entry, score). Replace this with an LLM call that returns relevance.
    Current stub uses simple length-normalized substring match.
    """
    out = []
    ql = set(query.lower().split())
    for e in entries:
        common = ql.intersection(set(e.text.lower().split()))
        score = len(common) / (1 + len(e.text.split()))
        out.append((e, score))
    out.sort(key=lambda x: x[1], reverse=True)
    return out
