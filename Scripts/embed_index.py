# simple script to load data/sample_memory.jsonl into retriever
from token_optimiser.retriever import Retriever
import sys

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_memory.jsonl"
    r = Retriever()
    r.load_from_jsonl(path)
    print(f"Loaded {len(r._docs)} documents.")
