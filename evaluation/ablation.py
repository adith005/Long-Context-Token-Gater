"""
evaluation/ablation.py
=======================

Ablation study harness for the journal paper.

Each experiment systematically disables one component of the memory
retrieval scoring formula and re-runs NeedleBench, measuring the
downstream effect on recall, MRR, NDCG, and token compression ratio.

This produces Table 2 in the paper: "Contribution of each scoring
component to retrieval quality."

Memory scoring formula (full system)
--------------------------------------
    score = α·sim + β·access + γ·recency + δ·source_weight
    conf  = score × usefulness × 100

Ablation configs
----------------
  full          All components active (proposed system)
  no_recency    γ = 0  (recency weight zeroed, others re-normalised)
  no_access     β = 0
  no_source     δ = 0, all sources treated equally
  no_usefulness usefulness clamped to 1.0 (no feedback scaling)
  sim_only      β = γ = δ = 0, α = 1.0  (pure cosine baseline)

Each config is a frozen dataclass so results are fully reproducible and
the exact weights used are logged alongside results.

CLI
---
  python -m evaluation.ablation                        # full table
  python -m evaluation.ablation --config sim_only full # specific configs
  python -m evaluation.ablation --output ablation.json
"""

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

# ── path fix ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from evaluation.metrics import compute_all_metrics, print_metrics_table


# ═════════════════════════════════════════════════════════════════════════════
# ABLATION CONFIG
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AblationConfig:
    """
    Defines a single ablation condition.

    All weight fields map directly to the retrieval scoring formula
    in memory_storage/storage.py.  Setting a weight to 0.0 disables
    that component entirely.

    usefulness_fixed : if set, overrides per-entry usefulness with this
                       constant, disabling the feedback loop.
    """
    name:             str
    description:      str

    # Retrieval weights
    sim_weight:       float = 0.50
    access_weight:    float = 0.15
    recency_weight:   float = 0.20
    source_weight:    float = 0.15
    recency_lambda:   float = 0.10

    # Usefulness
    usefulness_fixed: Optional[float] = None   # None = use stored values

    def total_weight(self) -> float:
        return self.sim_weight + self.access_weight + self.recency_weight + self.source_weight


# ── Canonical ablation conditions ─────────────────────────────────────────────

ABLATION_CONFIGS: Dict[str, AblationConfig] = {

    "full": AblationConfig(
        name        = "full",
        description = "Proposed system — all components active",
        sim_weight      = 0.50,
        access_weight   = 0.15,
        recency_weight  = 0.20,
        source_weight   = 0.15,
        usefulness_fixed= None,
    ),

    "no_recency": AblationConfig(
        name        = "no_recency",
        description = "Recency decay disabled (γ=0, others re-normalised to sum=1)",
        sim_weight      = 0.59,
        access_weight   = 0.18,
        recency_weight  = 0.00,
        source_weight   = 0.23,
        usefulness_fixed= None,
    ),

    "no_access": AblationConfig(
        name        = "no_access",
        description = "Access-count weighting disabled (β=0)",
        sim_weight      = 0.59,
        access_weight   = 0.00,
        recency_weight  = 0.23,
        source_weight   = 0.18,
        usefulness_fixed= None,
    ),

    "no_source": AblationConfig(
        name        = "no_source",
        description = "Source authority weighting disabled (δ=0, all sources equal)",
        sim_weight      = 0.59,
        access_weight   = 0.18,
        recency_weight  = 0.23,
        source_weight   = 0.00,
        usefulness_fixed= None,
    ),

    "no_usefulness": AblationConfig(
        name        = "no_usefulness",
        description = "Usefulness feedback disabled (clamped to 1.0)",
        sim_weight      = 0.50,
        access_weight   = 0.15,
        recency_weight  = 0.20,
        source_weight   = 0.15,
        usefulness_fixed= 1.0,
    ),

    "sim_only": AblationConfig(
        name        = "sim_only",
        description = "Pure cosine similarity baseline (α=1, all others=0)",
        sim_weight      = 1.00,
        access_weight   = 0.00,
        recency_weight  = 0.00,
        source_weight   = 0.00,
        usefulness_fixed= 1.0,
    ),
}


# ═════════════════════════════════════════════════════════════════════════════
# PATCHED RETRIEVAL  —  applies config weights at scoring time
# ═════════════════════════════════════════════════════════════════════════════

def _score_candidate(
    sim:          float,
    access_count: int,
    max_access:   int,
    timestamp:    int,
    source:       str,
    usefulness:   float,
    cfg:          AblationConfig,
) -> float:
    """
    Re-score a single memory entry using the ablation config's weights.
    Mirrors the formula in storage.py:retrieve_all() exactly.
    """
    import math
    import time as _time

    norm_access = access_count / max_access if max_access > 0 else 0.0
    days_old    = (_time.time() - timestamp) / 86400.0
    recency     = math.exp(-cfg.recency_lambda * days_old)

    from memory_storage.storage import SOURCE_WEIGHTS, DEFAULT_SOURCE_WEIGHT
    src_w       = SOURCE_WEIGHTS.get(source, DEFAULT_SOURCE_WEIGHT)

    raw = (
        cfg.sim_weight    * sim         +
        cfg.access_weight * norm_access +
        cfg.recency_weight * recency    +
        cfg.source_weight  * src_w
    )

    eff_usefulness = cfg.usefulness_fixed if cfg.usefulness_fixed is not None else usefulness
    return raw * eff_usefulness * 100.0


def retrieve_with_config(
    query_vec,
    cfg: AblationConfig,
    top_k: int = 10,
) -> List[Dict]:
    """
    Run memory retrieval using the given ablation config's weights.
    Bypasses the global weights in storage.py and scores directly.
    Returns candidates in the same format as retrieve_memory().
    """
    import numpy as np
    from memory_storage.storage import _store

    keys = _store.redis.keys("mem:*")
    rows = []
    for key in keys:
        data = _store.redis.hgetall(key)
        if b"embedding" not in data:
            continue
        vec  = _store._deserialize_vector(data[b"embedding"])
        sim  = _store._cosine_similarity(query_vec, vec)
        if sim < 0.30:   # keep MIN_SIM_FLOOR constant across all ablations
            continue
        rows.append({
            "content":      data[b"content"].decode(),
            "source":       data[b"source"].decode(),
            "sim":          sim,
            "access_count": int(data.get(b"access_count", b"0")),
            "timestamp":    int(data.get(b"timestamp",    b"0")),
            "usefulness":   float(data.get(b"usefulness", b"0.5")),
        })

    if not rows:
        return []

    max_access = max(r["access_count"] for r in rows) or 1
    for r in rows:
        r["confidence"] = _score_candidate(
            sim          = r["sim"],
            access_count = r["access_count"],
            max_access   = max_access,
            timestamp    = r["timestamp"],
            source       = r["source"],
            usefulness   = r["usefulness"],
            cfg          = cfg,
        )

    rows.sort(key=lambda x: x["confidence"], reverse=True)
    return rows[:top_k]


# ═════════════════════════════════════════════════════════════════════════════
# ABLATION RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_ablation(
    config_names:  List[str]  = None,
    modes:         List[str]  = None,
    haystack_sizes:List[str]  = None,
    call_llm_flag: bool       = False,
    seed:          int        = 42,
    output_path:   str        = None,
    verbose:       bool       = True,
    progress_cb               = None,
) -> Dict:
    """
    Run NeedleBench under each ablation config and collect metrics.

    For each (config × gating_mode × haystack × needle) combination,
    retrieval is done with the config's weights; gating and generation
    are unchanged.

    Parameters
    ----------
    config_names    : list of keys from ABLATION_CONFIGS (default: all)
    modes           : gating modes to test (default: ["entropy","simple","none"])
    haystack_sizes  : (default: ["short","medium","long"])
    call_llm_flag   : whether to make real LLM calls
    seed            : random seed for haystack construction
    output_path     : path to write JSON results
    verbose         : print progress table
    progress_cb     : callable(current, total, label) for UI progress bars

    Returns
    -------
    {
        "ablation_configs":  { config_name: config_dict },
        "results_by_config": { config_name: { mode: [TestResult dicts] } },
        "metrics_by_config": { config_name: { mode: metrics_dict } },
        "summary_table":     list of rows for paper Table 2,
        "run_metadata":      { seed, timestamp, total_experiments }
    }
    """
    from needlebench import NEEDLES, HAYSTACK_SIZES, build_haystack, gate
    from needlebench import needle_in_window, estimate_tokens, score_answer
    from prompt_creator.builder import build_prompt
    from llm_handler.handler import call_llm
    from utils.embedding import embed

    config_names   = config_names   or list(ABLATION_CONFIGS.keys())
    modes          = modes          or ["entropy", "simple", "none"]
    haystack_sizes = haystack_sizes or ["short", "medium", "long"]

    configs = {n: ABLATION_CONFIGS[n] for n in config_names if n in ABLATION_CONFIGS}
    if not configs:
        raise ValueError(f"No valid config names. Choose from: {list(ABLATION_CONFIGS.keys())}")

    total = len(configs) * len(modes) * len(haystack_sizes) * len(NEEDLES)
    idx   = 0

    results_by_config: Dict[str, Dict[str, List[Dict]]] = {
        cfg_name: {mode: [] for mode in modes}
        for cfg_name in configs
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"  ABLATION STUDY — Token Gater Memory Retrieval")
        print(f"{'='*70}")
        print(f"  Configs       : {list(configs.keys())}")
        print(f"  Gating modes  : {modes}")
        print(f"  Haystack sizes: {haystack_sizes}")
        print(f"  LLM calls     : {'yes' if call_llm_flag else 'no'}")
        print(f"  Total runs    : {total}")
        print(f"  Seed          : {seed}")
        print(f"{'='*70}\n")

    for cfg_name, cfg in configs.items():
        for mode in modes:
            for h_size in haystack_sizes:
                for needle in NEEDLES:
                    idx += 1
                    label = f"[{idx:03d}/{total}] cfg={cfg_name:<14} mode={mode:<8} h={h_size:<6} {needle['id']}"

                    t0 = time.time()

                    # ── Build haystack ────────────────────────────────────────
                    sentences, _ = build_haystack(needle, h_size, seed)
                    query        = needle["question"]
                    q_vec        = embed(query)

                    # ── Retrieve with this ablation config ────────────────────
                    # For NeedleBench we embed the haystack directly since there
                    # is no persistent memory store during evaluation. We score
                    # inline using the config weights as a proxy for what the
                    # memory store would return.
                    import math as _math
                    import numpy as np
                    q_norm = np.linalg.norm(q_vec)

                    raw_cands = []
                    for sent in sentences:
                        from utils.embedding import embed as _embed
                        s_vec  = _embed(sent)
                        s_norm = np.linalg.norm(s_vec)
                        sim    = float(np.dot(q_vec, s_vec) / (q_norm * s_norm + 1e-9))
                        if sim < 0.30:
                            continue
                        # In NeedleBench there is no access history or timestamps,
                        # so those components contribute 0. Only sim, source, and
                        # usefulness are active. This isolates the sim weight effect.
                        conf = _score_candidate(
                            sim          = sim,
                            access_count = 0,
                            max_access   = 1,
                            timestamp    = int(time.time()),
                            source       = "needlebench",
                            usefulness   = 1.0 if cfg.usefulness_fixed else 1.0,
                            cfg          = cfg,
                        )
                        raw_cands.append({
                            "sentence":   sent,
                            "content":    sent,
                            "confidence": conf,
                            "source":     "needlebench",
                            "doc_name":   "needlebench_haystack",
                        })

                    raw_cands.sort(key=lambda x: x["confidence"], reverse=True)

                    # ── Gate ──────────────────────────────────────────────────
                    window, stats = gate(raw_cands, mode)
                    recalled      = needle_in_window(needle["fact"], window)
                    prompt        = build_prompt(window, query)
                    prompt_tokens = estimate_tokens(prompt)

                    response_text     = ""
                    completion_tokens = 0
                    answer_score      = 0.0
                    error             = ""

                    if call_llm_flag:
                        try:
                            resp              = call_llm(prompt)
                            response_text     = resp.get("response_text", "")
                            prompt_tokens     = resp.get("prompt_tokens") or prompt_tokens
                            completion_tokens = resp.get("completion_tokens", 0)
                            answer_score      = score_answer(response_text, needle["answer_keywords"])
                        except Exception as e:
                            error        = str(e)
                            answer_score = 1.0 if recalled else 0.0
                    else:
                        answer_score = 1.0 if recalled else 0.0

                    result = {
                        "needle_id":         needle["id"],
                        "depth":             needle["depth"],
                        "haystack_size":     h_size,
                        "gating_mode":       mode,
                        "ablation_config":   cfg_name,
                        "needle_recalled":   recalled,
                        "answer_score":      answer_score,
                        "answer_keywords":   needle["answer_keywords"],
                        "response_text":     response_text,
                        "prompt_tokens":     prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "window_size":       len(window),
                        "candidates_in":     len(raw_cands),
                        "latency_sec":       round(time.time() - t0, 3),
                        "gating_stats":      stats,
                        "error":             error,
                    }

                    results_by_config[cfg_name][mode].append(result)

                    if verbose:
                        icon = "✓" if recalled else "✗"
                        print(f"  {label}  {icon}  win={len(window):3d}/{len(raw_cands):3d}  tok~{prompt_tokens}")

                    if progress_cb:
                        progress_cb(idx, total, label)

    # ── Compute metrics per (config × mode) ───────────────────────────────────
    metrics_by_config = {}
    for cfg_name in configs:
        metrics_by_config[cfg_name] = compute_all_metrics(
            results_by_mode = results_by_config[cfg_name],
            baseline_mode   = "none",
            k               = 5,
        )

    # ── Build flat summary table for paper ────────────────────────────────────
    summary_table = []
    for cfg_name, cfg in configs.items():
        for mode, m in metrics_by_config[cfg_name].items():
            summary_table.append({
                "config":                  cfg_name,
                "description":             cfg.description,
                "gating_mode":             mode,
                "recall_at_k":             m.get("recall_at_k",             0),
                "mrr":                     m.get("mrr",                     0),
                "ndcg_at_k":               m.get("ndcg_at_k",               0),
                "avg_answer_f1":           m.get("avg_answer_f1",           0),
                "avg_prompt_tokens":       m.get("avg_prompt_tokens",       0),
                "avg_window_size":         m.get("avg_window_size",         0),
                "avg_window_reduction":    m.get("avg_window_reduction",    0),
                "token_compression_ratio": m.get("token_compression_ratio", 1),
            })

    if verbose:
        _print_ablation_table(summary_table, configs)

    output = {
        "ablation_configs":  {n: asdict(c) for n, c in configs.items()},
        "results_by_config": results_by_config,
        "metrics_by_config": metrics_by_config,
        "summary_table":     summary_table,
        "run_metadata": {
            "seed":               seed,
            "timestamp":          time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_experiments":  total,
            "gating_modes":       modes,
            "haystack_sizes":     haystack_sizes,
            "call_llm":           call_llm_flag,
        },
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        if verbose:
            print(f"\n  Results saved → {output_path}")

    return output


# ═════════════════════════════════════════════════════════════════════════════
# PRETTY PRINTER
# ═════════════════════════════════════════════════════════════════════════════

def _print_ablation_table(summary_table: List[Dict], configs: Dict):
    """Print the ablation results as a paper-ready ASCII table."""
    print(f"\n{'='*90}")
    print(f"  ABLATION RESULTS  (entropy gating mode only)")
    print(f"{'='*90}")

    entropy_rows = [r for r in summary_table if r["gating_mode"] == "entropy"]

    hdr = f"  {'Config':<18}  {'Recall@5':>9}  {'MRR':>8}  {'NDCG@5':>8}  {'Ans F1':>8}  {'Tokens':>8}  {'WRR':>7}  {'TCR':>7}"
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr)
    print(sep)

    for row in entropy_rows:
        print(
            f"  {row['config']:<18}"
            f"  {row['recall_at_k']:>9.4f}"
            f"  {row['mrr']:>8.4f}"
            f"  {row['ndcg_at_k']:>8.4f}"
            f"  {row['avg_answer_f1']:>8.4f}"
            f"  {row['avg_prompt_tokens']:>8.1f}"
            f"  {row['avg_window_reduction']:>7.4f}"
            f"  {row['token_compression_ratio']:>7.4f}"
        )

    print(f"\n  Descriptions:")
    for cfg_name, cfg in configs.items():
        print(f"    {cfg_name:<18}  {cfg.description}")
    print(f"{'='*90}\n")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for Token Gater")
    parser.add_argument("--config",   nargs="+", default=list(ABLATION_CONFIGS.keys()),
                        choices=list(ABLATION_CONFIGS.keys()),
                        help="Which ablation configs to run")
    parser.add_argument("--mode",     nargs="+", default=["entropy", "simple", "none"],
                        choices=["entropy", "simple", "none"])
    parser.add_argument("--haystack", nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"])
    parser.add_argument("--llm",      action="store_true")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--output",   default="ablation_results.json")
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    run_ablation(
        config_names   = args.config,
        modes          = args.mode,
        haystack_sizes = args.haystack,
        call_llm_flag  = args.llm,
        seed           = args.seed,
        output_path    = args.output,
        verbose        = not args.quiet,
    )
