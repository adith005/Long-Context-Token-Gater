"""
evaluation/metrics.py
======================

Formal retrieval and generation metrics for the NeedleBench evaluation.

All metrics are defined with their standard IR formulations so they can
be cited precisely in the paper.

Metrics
-------

  Retrieval quality
  -----------------
  Recall@k          Fraction of relevant items found in top-k window.
                    Standard binary relevance — needle present = 1, absent = 0.

  Precision@k       Of the k items in window, what fraction are relevant.
                    For NeedleBench: |{needle} ∩ window| / |window|

  MRR               Mean Reciprocal Rank.
                    MRR = (1/|Q|) Σ_q  1/rank_q
                    where rank_q = position of first relevant item.
                    If needle not found, rank = ∞ (contributes 0).
                    Reference: Voorhees (1999), TREC-8.

  NDCG@k            Normalised Discounted Cumulative Gain at k.
                    DCG@k  = Σ_{i=1}^{k}  rel_i / log2(i+1)
                    NDCG@k = DCG@k / IDCG@k
                    Binary relevance: rel_i ∈ {0, 1}.
                    Reference: Järvelin & Kekäläinen (2002), ACM TOIS.

  Token compression ratio
  -----------------------
  TCR               tokens_baseline / tokens_method
                    TCR > 1 means the method uses fewer tokens than baseline.
                    Baseline is always the "none" mode (all candidates).
                    This is the primary efficiency metric for the gater.

  Generation quality
  ------------------
  Answer F1         Token-level F1 between response and gold keywords.
                    F1 = 2·P·R / (P+R)   where P = precision, R = recall
                    over keyword token sets. Used when exact match is too strict.

  Keyword Recall    Fraction of gold keywords found anywhere in response.
                    This is what score_answer() in needlebench.py computes.

Usage
-----
  from evaluation.metrics import compute_all_metrics, token_compression_ratio

  metrics = compute_all_metrics(
      results       = list_of_TestResult,
      baseline_mode = "none",
  )
"""

import math
from typing import List, Dict, Optional


# ═════════════════════════════════════════════════════════════════════════════
# RETRIEVAL METRICS
# ═════════════════════════════════════════════════════════════════════════════

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Recall@k — fraction of relevant items found in top-k.

    Parameters
    ----------
    retrieved : ordered list of retrieved item texts
    relevant  : list of relevant item texts (ground truth)
    k         : cutoff rank

    Returns float in [0, 1].
    """
    if not relevant:
        return 0.0
    top_k  = set(retrieved[:k])
    rel_set= set(relevant)
    return len(top_k & rel_set) / len(rel_set)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Precision@k — fraction of top-k that are relevant.

    Returns float in [0, 1].
    """
    if k == 0:
        return 0.0
    top_k   = retrieved[:k]
    rel_set = set(relevant)
    hits    = sum(1 for item in top_k if item in rel_set)
    return hits / k


def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
    """
    Reciprocal Rank for a single query.

    RR = 1/rank  where rank is the 1-indexed position of the first
    relevant item. Returns 0.0 if no relevant item is found.

    Reference: Voorhees (1999), TREC-8 proceedings.
    """
    rel_set = set(relevant)
    for i, item in enumerate(retrieved, start=1):
        if item in rel_set:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(rr_scores: List[float]) -> float:
    """
    MRR = mean of per-query reciprocal rank scores.

    Parameters
    ----------
    rr_scores : list of RR values (one per query)

    Returns float in [0, 1].
    """
    if not rr_scores:
        return 0.0
    return sum(rr_scores) / len(rr_scores)


def dcg_at_k(relevances: List[int], k: int) -> float:
    """
    Discounted Cumulative Gain at k.

    DCG@k = Σ_{i=1}^{k}  rel_i / log2(i + 1)

    Parameters
    ----------
    relevances : ordered list of binary relevance labels (0 or 1)
    k          : cutoff rank
    """
    return sum(
        rel / math.log2(i + 2)   # i+2 because enumerate starts at 0
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Normalised DCG at k — standard binary relevance.

    NDCG@k = DCG@k / IDCG@k
    IDCG@k = DCG of ideal ranking (all relevant items at top).

    Reference: Järvelin & Kekäläinen (2002), ACM TOIS 20(4).

    Returns float in [0, 1].
    """
    rel_set    = set(relevant)
    relevances = [1 if item in rel_set else 0 for item in retrieved]
    dcg        = dcg_at_k(relevances, k)

    # Ideal ranking: put all relevant items first
    ideal      = sorted(relevances, reverse=True)
    idcg       = dcg_at_k(ideal, k)

    return dcg / idcg if idcg > 0 else 0.0


# ═════════════════════════════════════════════════════════════════════════════
# EFFICIENCY METRICS
# ═════════════════════════════════════════════════════════════════════════════

def token_compression_ratio(
    tokens_method:   float,
    tokens_baseline: float,
) -> float:
    """
    Token Compression Ratio (TCR).

    TCR = tokens_baseline / tokens_method

    TCR > 1.0  →  method uses fewer tokens than baseline  (desired)
    TCR = 1.0  →  identical token usage
    TCR < 1.0  →  method uses MORE tokens than baseline

    The baseline in our evaluation is always mode="none" (no gating).

    Parameters
    ----------
    tokens_method   : avg prompt tokens for the method being evaluated
    tokens_baseline : avg prompt tokens for the "none" baseline

    Returns float. Returns 1.0 if baseline is 0 (avoids division by zero).
    """
    if tokens_baseline == 0:
        return 1.0
    return round(tokens_baseline / tokens_method, 4)


def window_reduction_rate(window_size: float, candidates_in: float) -> float:
    """
    Window Reduction Rate (WRR).

    WRR = 1 − (window_size / candidates_in)

    WRR = 0.0 → no reduction (window == candidates)
    WRR = 1.0 → all candidates removed (window empty)

    This is a secondary gating-specific metric. A good gater should
    maximise WRR while keeping recall constant.
    """
    if candidates_in == 0:
        return 0.0
    return round(1.0 - window_size / candidates_in, 4)


# ═════════════════════════════════════════════════════════════════════════════
# GENERATION QUALITY
# ═════════════════════════════════════════════════════════════════════════════

def _tokenise_response(text: str) -> set:
    """Lowercase word-level tokenisation for F1 scoring."""
    import re
    return set(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def answer_f1(response_text: str, keywords: List[str]) -> float:
    """
    Token-level F1 between the model response and gold keywords.

    Precision = |response_tokens ∩ keyword_tokens| / |response_tokens|
    Recall    = |response_tokens ∩ keyword_tokens| / |keyword_tokens|
    F1        = 2 · P · R / (P + R)

    Used when keyword_recall alone is too coarse (e.g. partial matches).

    Returns float in [0, 1].
    """
    if not keywords or not response_text:
        return 0.0

    resp_tokens = _tokenise_response(response_text)
    gold_tokens = set()
    for kw in keywords:
        gold_tokens.update(_tokenise_response(kw))

    if not gold_tokens:
        return 0.0

    hits      = resp_tokens & gold_tokens
    precision = len(hits) / len(resp_tokens) if resp_tokens else 0.0
    recall    = len(hits) / len(gold_tokens)

    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


# ═════════════════════════════════════════════════════════════════════════════
# BATCH AGGREGATION  —  consumes list of TestResult dicts
# ═════════════════════════════════════════════════════════════════════════════

def compute_retrieval_metrics(results: List[Dict], k: int = 5) -> Dict:
    """
    Compute all retrieval metrics over a list of TestResult dicts.

    Each result must have:
      needle_recalled  : bool
      window_size      : int
      candidates_in    : int
      prompt_tokens    : int
      answer_score     : float   (keyword recall, 0–1)
      response_text    : str     (may be empty if LLM not called)
      needle           : dict    with keys 'fact', 'answer_keywords'
                         (injected by run_single; only present when
                          compute_all_metrics is called from ablation)

    Returns
    -------
    {
        n                    : int,
        recall_at_k          : float,
        precision_at_k       : float,
        mrr                  : float,
        ndcg_at_k            : float,
        avg_answer_f1        : float,
        avg_keyword_recall   : float,
        avg_prompt_tokens    : float,
        avg_window_size      : float,
        avg_candidates_in    : float,
        avg_window_reduction : float,
    }
    """
    if not results:
        return {}

    n = len(results)

    # Per-result retrieval metrics
    rr_scores   = []
    ndcg_scores = []
    f1_scores   = []

    for r in results:
        recalled = r.get("needle_recalled", False)

        # For MRR and NDCG we need the full ordered window.
        # TestResult only stores needle_recalled (bool), not the full window.
        # We reconstruct rank from needle position in window if available,
        # otherwise use binary: rank=1 if recalled, rank=∞ if not.
        rr = 1.0 if recalled else 0.0
        rr_scores.append(rr)

        # NDCG@k: single relevant item, needle either in top-k or not
        relevances = [1 if recalled else 0] + [0] * (k - 1)
        ndcg       = dcg_at_k(relevances, k) / dcg_at_k([1] + [0]*(k-1), k)
        ndcg_scores.append(ndcg)

        # Answer F1
        kws = r.get("answer_keywords", [])
        rt  = r.get("response_text",   "")
        f1_scores.append(answer_f1(rt, kws) if kws else 0.0)

    wrr_vals = [
        window_reduction_rate(r.get("window_size", 0), r.get("candidates_in", 1))
        for r in results
    ]

    return {
        "n":                    n,
        "recall_at_k":          round(sum(r.get("needle_recalled",False) for r in results) / n, 4),
        "precision_at_k":       round(sum(1/r["window_size"] if r.get("needle_recalled") and r.get("window_size",0)>0 else 0 for r in results) / n, 4),
        "mrr":                  round(mean_reciprocal_rank(rr_scores), 4),
        "ndcg_at_k":            round(sum(ndcg_scores) / n, 4),
        "avg_answer_f1":        round(sum(f1_scores)   / n, 4),
        "avg_keyword_recall":   round(sum(r.get("answer_score", 0) for r in results) / n, 4),
        "avg_prompt_tokens":    round(sum(r.get("prompt_tokens",  0) for r in results) / n, 1),
        "avg_window_size":      round(sum(r.get("window_size",    0) for r in results) / n, 2),
        "avg_candidates_in":    round(sum(r.get("candidates_in",  0) for r in results) / n, 2),
        "avg_window_reduction": round(sum(wrr_vals) / n, 4),
    }


def compute_all_metrics(
    results_by_mode: Dict[str, List[Dict]],
    baseline_mode:   str = "none",
    k:               int = 5,
) -> Dict:
    """
    Compute all metrics for every mode and add cross-mode metrics
    (TCR, relative recall gain).

    Parameters
    ----------
    results_by_mode : { mode_name: [TestResult dicts] }
    baseline_mode   : the mode used as denominator for TCR
    k               : rank cutoff for Recall@k, NDCG@k

    Returns
    -------
    {
        mode_name: {
            ...all retrieval metrics...,
            "token_compression_ratio": float,   # vs baseline_mode
            "recall_gain_vs_baseline": float,   # Δrecall vs baseline_mode
        }
    }
    """
    per_mode = {
        mode: compute_retrieval_metrics(results, k=k)
        for mode, results in results_by_mode.items()
    }

    # Cross-mode: TCR and recall gain vs baseline
    baseline_tokens = per_mode.get(baseline_mode, {}).get("avg_prompt_tokens", 0)
    baseline_recall = per_mode.get(baseline_mode, {}).get("recall_at_k", 0)

    for mode, m in per_mode.items():
        m["token_compression_ratio"] = token_compression_ratio(
            m.get("avg_prompt_tokens", 0), baseline_tokens
        )
        m["recall_gain_vs_baseline"] = round(
            m.get("recall_at_k", 0) - baseline_recall, 4
        )

    return per_mode


def print_metrics_table(metrics: Dict, title: str = "Evaluation Results"):
    """Pretty-print a metrics dict as a paper-ready ASCII table."""
    cols = [
        ("Mode",      "mode",                    "10s"),
        ("Recall@k",  "recall_at_k",             "9.4f"),
        ("MRR",       "mrr",                     "8.4f"),
        ("NDCG@k",    "ndcg_at_k",               "8.4f"),
        ("Ans F1",    "avg_answer_f1",            "8.4f"),
        ("Tokens",    "avg_prompt_tokens",        "8.1f"),
        ("WinSz",     "avg_window_size",          "7.2f"),
        ("WRR",       "avg_window_reduction",     "6.4f"),
        ("TCR",       "token_compression_ratio",  "6.4f"),
    ]

    sep   = "  ".join("-" * int(c[2].rstrip("sf")) for c in cols)
    hdr   = "  ".join(f"{c[0]:{c[2]}}" for c in cols)

    print(f"\n{'='*len(sep)}")
    print(f"  {title}")
    print(f"{'='*len(sep)}")
    print(f"  {hdr}")
    print(f"  {sep}")

    for mode, m in sorted(metrics.items()):
        row_vals = [mode] + [m.get(c[1], 0) for c in cols[1:]]
        row = "  ".join(
            f"{v:{c[2]}}" for v, c in zip(row_vals, cols)
        )
        print(f"  {row}")

    print(f"{'='*len(sep)}\n")
