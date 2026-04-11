"""
gating/hybrid_gater.py
=======================

Hybrid gating — four prompt-level context selection strategies that
operate on the assembled prompt text rather than on the candidate list.

This is distinct from the candidate-list gaters (entropy, joint, BM25,
quantum) which select *which* memories to include. Hybrid gating assumes
the prompt is already assembled and applies a secondary word-level filter
to further reduce token count.

Strategies
----------
  stepwise      Simple positional truncation to a keep_ratio of total words.
  hybrid_score  Scores each word by position, keyword relevance, and length.
  diversity     Removes repeated words while maintaining a keep_ratio.
  budget        Enforces a strict word budget using position or hybrid scoring.

Pipeline integration
--------------------
  When gating_mode == "hybrid", the pipeline runs standard multi-factor
  retrieval, assembles the full prompt, then passes it through
  apply_hybrid_gating(), which applies all four strategies and returns
  the one that produces the best compression while staying above the
  minimum word threshold. The selected sub-strategy and stats are
  recorded in gating_stats for the tracer and frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Optional


DEFAULT_KEYWORDS = {
    "benefits", "architecture", "quantum", "python", "function",
    "memory", "context", "model", "retrieval", "entropy", "gating",
    "performance", "latency", "token", "embedding", "inference",
    "system", "algorithm", "data", "result", "analysis", "method",
}


# =============================================================================
# CORE STRATEGY FUNCTIONS  (original signatures preserved exactly)
# =============================================================================

def apply_stepwise_gating(prompt_text: str, keep_ratio: float = 0.7) -> str:
    """Simple truncation based on a ratio of total words."""
    words = prompt_text.split()
    keep_count = int(len(words) * keep_ratio)
    return " ".join(words[:keep_count])


def apply_hybrid_scoring_gating(
    prompt_text: str,
    keyword_set: Optional[set] = None,
) -> str:
    """
    Scores words based on position, keyword relevance, and length.
    Returns the text reconstructed from highest-scoring words.
    NOTE: word order is NOT preserved — words are sorted by score.
    """
    if keyword_set is None:
        keyword_set = DEFAULT_KEYWORDS

    words = prompt_text.split()
    if not words:
        return ""

    scored_words: List[Tuple[str, float]] = []
    for i, w in enumerate(words):
        pos_score     = 1.0 - (i / len(words))
        clean_w       = w.lower().strip(",.?!:;")
        keyword_score = 1.0 if clean_w in keyword_set else 0.0
        length_score  = min(len(w) / 10, 1.0)
        total_score   = (0.5 * pos_score) + (0.3 * keyword_score) + (0.2 * length_score)
        scored_words.append((w, total_score))

    selected = sorted(scored_words, key=lambda x: x[1], reverse=True)
    return " ".join(w for w, _ in selected)


def apply_diversity_gating(prompt_text: str, keep_ratio: float = 0.7) -> str:
    """Filters redundant words while maintaining a specified keep ratio."""
    words = prompt_text.split()
    keep_count  = int(len(words) * keep_ratio)
    unique_words: List[str] = []
    seen: set = set()

    for w in words:
        clean_w = w.lower().strip(",.?!:;")
        if clean_w not in seen:
            unique_words.append(w)
            seen.add(clean_w)
        if len(unique_words) >= keep_count:
            break

    return " ".join(unique_words)


def apply_token_budget_gating(
    prompt_text: str,
    budget: int = 64,
    scoring: str = "hybrid",
    keyword_set: Optional[set] = None,
) -> str:
    """
    Enforces a strict word budget using either positional clipping or
    importance scoring.
    """
    if keyword_set is None:
        keyword_set = DEFAULT_KEYWORDS

    words = prompt_text.split()
    if len(words) <= budget:
        return prompt_text

    if scoring == "position":
        return " ".join(words[:budget])

    elif scoring == "hybrid":
        scored_words: List[Tuple[str, float]] = []
        for i, w in enumerate(words):
            pos_score     = 1.0 - (i / len(words))
            clean_w       = w.lower().strip(",.?!:;")
            keyword_score = 1.0 if clean_w in keyword_set else 0.0
            length_score  = min(len(w) / 10, 1.0)
            total_score   = (0.5 * pos_score) + (0.3 * keyword_score) + (0.2 * length_score)
            scored_words.append((w, total_score))

        selected = sorted(scored_words, key=lambda x: x[1], reverse=True)[:budget]
        return " ".join(w for w, _ in selected)

    else:
        raise ValueError(f"Unknown scoring method: {scoring!r}. Use 'position' or 'hybrid'.")


# =============================================================================
# AUTO-SELECT WRAPPER  —  used by pipeline.py
# =============================================================================

@dataclass
class HybridGatingResult:
    """Result of apply_hybrid_gating(), returned to pipeline.py."""
    compressed_prompt:     str
    selected_strategy:     str
    original_word_count:   int
    compressed_word_count: int
    compression_ratio:     float
    stats:                 dict = field(default_factory=dict)


def apply_hybrid_gating(
    prompt_text:  str,
    mode:         str   = "auto",
    keep_ratio:   float = 0.7,
    budget:       int   = 512,
    keyword_set:  Optional[set] = None,
) -> HybridGatingResult:
    """
    Apply hybrid gating to an assembled prompt and return the compressed
    result with full statistics for the pipeline tracer.

    Parameters
    ----------
    prompt_text  : assembled RAG prompt string
    mode         : "auto"         — try all four, pick best compression
                   "stepwise"     — positional truncation only
                   "hybrid_score" — importance scoring only
                   "diversity"    — deduplication only
                   "budget"       — word budget enforcement only
    keep_ratio   : target fraction of words to retain (stepwise + diversity)
    budget       : hard word cap for budget mode (default 512)
    keyword_set  : domain keywords for hybrid_score + budget modes
    """
    if keyword_set is None:
        keyword_set = DEFAULT_KEYWORDS

    original_words = len(prompt_text.split())

    strategies = {
        "stepwise":     lambda: apply_stepwise_gating(prompt_text, keep_ratio),
        "hybrid_score": lambda: apply_hybrid_scoring_gating(prompt_text, keyword_set),
        "diversity":    lambda: apply_diversity_gating(prompt_text, keep_ratio),
        "budget":       lambda: apply_token_budget_gating(prompt_text, budget, "hybrid", keyword_set),
    }

    if mode != "auto" and mode in strategies:
        result_text = strategies[mode]()
        selected    = mode
    else:
        min_words  = max(10, int(original_words * keep_ratio * 0.5))
        best_text  = prompt_text
        best_name  = "none"
        best_ratio = 1.0

        for name, fn in strategies.items():
            try:
                candidate  = fn()
                cand_words = len(candidate.split())
                cand_ratio = cand_words / original_words if original_words > 0 else 1.0
                if cand_words >= min_words and cand_ratio < best_ratio:
                    best_text  = candidate
                    best_name  = name
                    best_ratio = cand_ratio
            except Exception:
                continue

        result_text = best_text
        selected    = best_name

    compressed_words = len(result_text.split())
    ratio = compressed_words / original_words if original_words > 0 else 1.0

    return HybridGatingResult(
        compressed_prompt     = result_text,
        selected_strategy     = selected,
        original_word_count   = original_words,
        compressed_word_count = compressed_words,
        compression_ratio     = round(ratio, 4),
        stats = {
            "strategy":               "hybrid",
            "sub_strategy":           selected,
            "original_words":         original_words,
            "compressed_words":       compressed_words,
            "word_compression_ratio": round(1.0 - ratio, 4),
            "keep_ratio":             keep_ratio,
            "budget":                 budget,
        },
    )
