"""
context_window.py  —  Entropy-Guided Minimal Context Window
============================================================

Takes pre-loaded, pre-scored candidates (from memory + document store)
and returns the smallest stable window that retains all relevant info.

Shannon Entropy
---------------
    H = -Σ p_i · log2(p_i)     p = softmax(confidence scores in window)

    High H  →  diverse, non-redundant information
    Low  H  →  window is dominated by near-duplicates (wasteful)

Two-phase algorithm
-------------------
    Phase A  Forward pass — add items while ΔH > delta_thresh
             Stop the moment new items stop raising entropy (plateau).

    Phase B  Backward prune — remove the weakest item repeatedly
             as long as H stays above the floor.
             This sheds items that were added before the plateau but
             turn out to be individually redundant.

Result: smallest set whose joint entropy is stable and ≥ floor.
"""

import numpy as np

# ── Tunables ──────────────────────────────────────────────────────────────────
ENTROPY_FLOOR        = 1.0    # bits — minimum acceptable window entropy
ENTROPY_DELTA_THRESH = 0.05   # bits — ΔH below this means "plateau, stop adding"
MAX_WINDOW           = 15     # hard cap before entropy logic runs


# ═════════════════════════════════════════════════════════════════════════════
# ENTROPY MATH
# ═════════════════════════════════════════════════════════════════════════════

def _entropy(scores: list[float]) -> float:
    """
    Shannon entropy (bits) over softmax-normalised confidence scores.

    Returns 0.0 for fewer than 2 items (entropy undefined for single item).
    """
    if len(scores) < 2:
        return 0.0
    a  = np.array(scores, dtype=np.float64)
    a -= a.max()                        # numerical stability
    p  = np.exp(a) / np.exp(a).sum()
    p  = np.clip(p, 1e-12, None)
    return float(-np.sum(p * np.log2(p)))


# ═════════════════════════════════════════════════════════════════════════════
# CONTEXT WINDOW BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_context_window(
    candidates: list[dict],
    entropy_floor: float = ENTROPY_FLOOR,
    entropy_delta: float = ENTROPY_DELTA_THRESH,
    max_window: int      = MAX_WINDOW,
) -> dict:
    """
    Build the minimal stable context window from a ranked candidate list.

    Parameters
    ----------
    candidates    : list of dicts, each must have a "confidence" key (0–100 %).
                    Must be sorted by confidence descending before calling.
                    Each dict can carry any fields (memory turn, doc chunk, etc.)
    entropy_floor : H must not fall below this value after pruning  [bits]
    entropy_delta : ΔH threshold for plateau detection              [bits]
    max_window    : hard cap on items considered before entropy logic

    Returns
    -------
    {
        window         : list[dict],   ← minimal context items
        window_entropy : float,        ← final H in bits
        is_stable      : bool,         ← H >= floor (or window size <= 2)
        stats          : {
            candidates_in  : int,
            window_size    : int,
            pruned         : int,
            plateau_at     : int,      ← index where forward pass stopped
        }
    }
    """

    # ── Phase A: Forward pass — stop at entropy plateau ───────────────────────
    window  = []
    scores  = []
    prev_H  = 0.0
    plateau = len(candidates)          # index where we stopped (default = end)

    for idx, item in enumerate(candidates[:max_window]):
        trial_H = _entropy(scores + [item["confidence"]])
        delta_H = trial_H - prev_H

        if len(window) >= 2 and delta_H < entropy_delta:
            plateau = idx
            break                      # new item adds negligible information

        window.append(item)
        scores.append(item["confidence"])
        prev_H = trial_H

    # ── Phase B: Backward prune — shed redundant tail items ──────────────────
    pruned = 0
    while len(window) > 1:
        if _entropy(scores[:-1]) >= entropy_floor:
            window.pop()
            scores.pop()
            pruned += 1
        else:
            break                      # further pruning would destabilise H

    # ── Final stats ───────────────────────────────────────────────────────────
    final_H   = _entropy(scores)
    is_stable = final_H >= entropy_floor or len(window) <= 2

    return {
        "window"        : window,
        "window_entropy": round(final_H, 4),
        "is_stable"     : is_stable,
        "stats": {
            "candidates_in": len(candidates),
            "window_size"  : len(window),
            "pruned"       : pruned,
            "plateau_at"   : plateau,
        },
    }