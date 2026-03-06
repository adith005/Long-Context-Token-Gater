from typing import List, Optional
from .models import MemoryEntry
import json
import os

# Preferred: chroma/faiss/sentence-transformers.
# Fallback: sklearn TF-IDF vector store for simple testing.

class Retriever:
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path
        self._docs = []  # list[MemoryEntry]
        self._texts = []
        self._ids = []

        # lazy vectorizer for fallback
        self._use_tfidf = True
        self._vectorizer = None
        self._vectors = None

    def load_from_jsonl(self, path: str):
        docs = []
        with open(path, "r", encoding="utf8") as fh:
            for line in fh:
                obj = json.loads(line)
                docs.append(MemoryEntry(**obj))
        self.add_documents(docs)

    def add_documents(self, docs: List[MemoryEntry]):
        for d in docs:
            self._docs.append(d)
            self._texts.append(d.text)
            self._ids.append(d.id)
        self._fit_fallback()

    def _fit_fallback(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(max_features=20000)
            self._vectors = self._vectorizer.fit_transform(self._texts)
            self._use_tfidf = True
        except Exception:
            self._use_tfidf = False

    def query(self, q: str, top_k: int = 10):
        if self._use_tfidf and self._vectors is not None:
            from sklearn.metrics.pairwise import linear_kernel
            qv = self._vectorizer.transform([q])
            sims = linear_kernel(qv, self._vectors).flatten()
            idxs = sims.argsort()[::-1][:top_k]
            return [self._docs[i] for i in idxs if sims[i] > 0]
        # fallback naive substring
        result = [d for d in self._docs if q.lower() in d.text.lower()]
        return result[:top_k]
