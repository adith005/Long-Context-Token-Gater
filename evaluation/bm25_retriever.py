"""
evaluation/bm25_retriever.py
=============================

BM25-Okapi retriever — pure Python, zero external dependencies.

Used as the standard IR baseline in the NeedleBench evaluation.
Produces the exact same candidate dict shape as the embedding-based
retriever so it can be dropped straight into the gating pipeline.

BM25-Okapi formula
------------------
    score(D, Q) = Σ_{t∈Q}  IDF(t) · f(t,D) · (k1 + 1)
                            ─────────────────────────────
                            f(t,D) + k1·(1 − b + b·|D|/avgdl)

    IDF(t) = log( (N − df(t) + 0.5) / (df(t) + 0.5) + 1 )

    k1  = 1.5   (term-frequency saturation)
    b   = 0.75  (document length normalisation)

References
----------
  Robertson & Zaragoza (2009). "The Probabilistic Relevance Framework:
  BM25 and Beyond." Foundations and Trends in Information Retrieval.
"""

import math
import re
from typing import List, Dict


# ── Hyperparameters (Robertson & Zaragoza, 2009 defaults) ────────────────────
K1 = 1.5
B  = 0.75


def _tokenise(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25:
    """
    BM25-Okapi index over a fixed corpus of sentences.

    Parameters
    ----------
    corpus : list of strings — the sentences to index
    k1, b  : BM25 hyperparameters

    Usage
    -----
    bm25 = BM25(sentences)
    scores = bm25.score_all(query_string)   # → list of floats, same len as corpus
    """

    def __init__(self, corpus: List[str], k1: float = K1, b: float = B):
        self.k1     = k1
        self.b      = b
        self.corpus = corpus

        # Tokenise corpus
        self._tok: List[List[str]] = [_tokenise(doc) for doc in corpus]

        # Per-document term frequencies
        self._tf: List[Dict[str, int]] = []
        for tokens in self._tok:
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)

        # Document lengths
        dl        = [len(t) for t in self._tok]
        self._dl  = dl
        self._avgdl = sum(dl) / len(dl) if dl else 1.0

        # Inverted index: term → set of doc indices
        N = len(corpus)
        inv: Dict[str, set] = {}
        for i, tf in enumerate(self._tf):
            for term in tf:
                inv.setdefault(term, set()).add(i)

        # IDF per term  (Robertson & Zaragoza, 2009 smooth variant)
        self._idf: Dict[str, float] = {}
        for term, docs in inv.items():
            df = len(docs)
            self._idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def score_all(self, query: str) -> List[float]:
        """
        Return BM25 score for every document in the corpus.

        Returns a list of floats in the same order as self.corpus.
        """
        q_tokens = _tokenise(query)
        scores   = [0.0] * len(self.corpus)

        for term in q_tokens:
            idf = self._idf.get(term, 0.0)
            if idf == 0.0:
                continue
            for i, tf in enumerate(self._tf):
                freq  = tf.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * self._dl[i] / self._avgdl)
                scores[i] += idf * (freq * (self.k1 + 1)) / denom

        return scores


def bm25_retrieve(
    sentences:  List[str],
    query:      str,
    source:     str = "bm25",
    doc_name:   str = "bm25_corpus",
) -> List[Dict]:
    """
    Build a BM25 index over `sentences`, score against `query`, and return
    a candidate list in the exact same format as the embedding retriever.

    Confidence is normalised to [0, 100] using min-max scaling so it is
    directly comparable to cosine-similarity-based confidence scores.

    Parameters
    ----------
    sentences : haystack sentences to rank
    query     : user question
    source    : source tag written into each candidate dict
    doc_name  : doc_name tag written into each candidate dict

    Returns
    -------
    List of candidate dicts sorted by confidence descending:
        { sentence, content, confidence (0–100), source, doc_name, bm25_raw }
    """
    if not sentences:
        return []

    bm25   = BM25(sentences)
    raw    = bm25.score_all(query)

    # Normalise to [0, 100]
    lo, hi = min(raw), max(raw)
    span   = (hi - lo) or 1.0
    norm   = [(s - lo) / span * 100.0 for s in raw]

    candidates = [
        {
            "sentence":   sent,
            "content":    sent,
            "confidence": round(conf, 4),
            "source":     source,
            "doc_name":   doc_name,
            "bm25_raw":   round(raw_score, 6),
        }
        for sent, conf, raw_score in zip(sentences, norm, raw)
    ]

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates
