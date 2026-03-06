from token_optimiser.retriever import Retriever
from token_optimiser.heuristic import top_k_by_similarity
from token_optimiser.models import MemoryEntry

def test_retriever_simple(tmp_path):
    entries = [
        MemoryEntry(id="1", text="Alice likes apples"),
        MemoryEntry(id="2", text="Bob likes bananas"),
        MemoryEntry(id="3", text="Charlie likes cherries"),
    ]
    r = Retriever()
    r.add_documents(entries)
    res = top_k_by_similarity(r, "apples", top_k=2)
    assert any("apples" in e.text for e in res)
