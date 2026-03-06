from memory.history_db import retrieve_similar
from rag.retriever import retrieve

def gate(query, memory_weight=0.6, doc_weight=0.4, top_k=5):
    mem_results = retrieve_similar(query, k=top_k)
    doc_results = retrieve(query, k=top_k)

    # Merge and score
    merged = []
    for m in mem_results:
        merged.append({"source": "memory", "text": m["response"], "score": memory_weight})
    for d in doc_results:
        merged.append({"source": "document", "text": d, "score": doc_weight})

    # Simple heuristic scoring and sort
    merged.sort(key=lambda x: x["score"], reverse=True)
    return [m["text"] for m in merged[:top_k]]
