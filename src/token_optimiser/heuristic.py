from typing import List
from .models import MemoryEntry
from .retriever import Retriever
from datetime import datetime, timedelta

def top_k_by_similarity(retriever: Retriever, query: str, top_k: int = 10) -> List[MemoryEntry]:
    return retriever.query(query, top_k=top_k)

def recency_filter(entries: List[MemoryEntry], days: int = 365) -> List[MemoryEntry]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    return [e for e in entries if (e.created_at and e.created_at >= cutoff) or not e.created_at]

def keyword_filter(entries: List[MemoryEntry], keywords: List[str]) -> List[MemoryEntry]:
    ks = [k.lower() for k in keywords]
    out = []
    for e in entries:
        txt = e.text.lower()
        if any(k in txt for k in ks):
            out.append(e)
    return out
