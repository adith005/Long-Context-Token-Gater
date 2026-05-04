"""
scripts/mass_query_runner.py
==============================

Runs a battery of queries across all gating modes and saves full results
to JSON for the paper's experimental section.

What it tests
-------------
For each (query × gating_mode) combination:
  - Runs the full pipeline via run_pipeline()
  - Records all output metrics: tokens, latency, cost, window size,
    entropy stats, gating stats, TCR vs none baseline
  - Optionally tests multiple LLM model strings by patching settings

Gating modes tested
-------------------
  entropy   — proposed method (Shannon entropy two-phase gating)
  simple    — top-15 cosine similarity baseline
  none      — no gating, all candidates (token baseline)
  bm25      — BM25-Okapi retrieval baseline (via needlebench retriever)

Output
------
  paper_results.json — one record per (query × mode), plus aggregate summary

Usage
-----
  # Full run, all modes, all queries
  python scripts/mass_query_runner.py --output results/paper_results.json

  # Quick test: 5 queries, entropy + none only
  python scripts/mass_query_runner.py --modes entropy none --queries 5

  # Specify LM Studio model string
  python scripts/mass_query_runner.py --model lfm2.5-1.2b-thinking

Requirements
------------
  - Redis running with memories injected
  - LM Studio running on localhost:1234 (or set LLM_BASE_URL env var)
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from orchestrator.pipeline import run_pipeline
from utils.embedding import embed
from evaluation.bm25_retriever import bm25_retrieve
from gating.token_gater import build_context_window
from memory_storage.storage import retrieve_memory
from prompt_creator.builder import build_prompt
from llm_handler.handler import call_llm


# =============================================================================
# QUERY BANK  —  diverse test queries for paper
# =============================================================================

QUERY_BANK = [
    # AI/ML
    "What is the difference between supervised and unsupervised learning?",
    "How does backpropagation work in a neural network?",
    "What is the vanishing gradient problem and how is it solved?",
    "Explain the transformer architecture and self-attention.",
    "What is the difference between overfitting and underfitting?",
    "What is transfer learning and when is it used?",
    "How does the attention mechanism work in neural networks?",
    "What is reinforcement learning from human feedback?",
    "What is the difference between precision and recall?",
    "How does cosine similarity work for embedding comparison?",

    # Computer Science
    "What is the difference between a stack and a queue?",
    "Explain dynamic programming and memoisation.",
    "What is Big O notation and why does it matter?",
    "What is the difference between TCP and UDP?",
    "How does a hash table work and what are collision strategies?",
    "What is database normalisation and why is it important?",
    "Explain the ACID properties of a database transaction.",
    "What is the difference between a process and a thread?",
    "How does garbage collection work in programming languages?",
    "What is the difference between SQL and NoSQL databases?",

    # Physics
    "What is Newton's second law of motion?",
    "How does nuclear fission differ from nuclear fusion?",
    "What is the Heisenberg uncertainty principle?",
    "Explain the Doppler effect with an example.",
    "What is the difference between potential and kinetic energy?",
    "How does electromagnetic induction work?",
    "What is special relativity and what are its consequences?",
    "What is entropy in the context of thermodynamics?",
    "How do black holes form and what is the event horizon?",
    "What is quantum entanglement?",

    # Biology
    "What is the difference between mitosis and meiosis?",
    "How does photosynthesis work?",
    "What is the central dogma of molecular biology?",
    "How does the immune system identify and destroy pathogens?",
    "What is CRISPR and how is it used in gene editing?",
    "What is the difference between a virus and a bacterium?",
    "How does natural selection drive evolution?",
    "What is the role of ATP in cellular energy metabolism?",
    "What is epigenetics and how does it affect gene expression?",
    "How do vaccines create immunological memory?",

    # Economics
    "What is the difference between monetary and fiscal policy?",
    "How does compound interest work over time?",
    "What is opportunity cost and why does it matter?",
    "Explain supply and demand equilibrium.",
    "What is the difference between a stock and a bond?",
    "What is quantitative easing and when is it used?",
    "What is GDP and how is it measured?",
    "What is inflation and how does it affect purchasing power?",
    "What is comparative advantage in international trade?",
    "How does diversification reduce investment risk?",

    # Drones / Robotics
    "How does PID control maintain drone stability?",
    "What is the difference between ArduPilot and PX4?",
    "How does a quadcopter achieve yaw rotation?",
    "What is MAVLink and how is it used in drone communication?",
    "How does GPS provide position data to a flight controller?",
    "What is the purpose of a companion computer on a drone?",
    "How does obstacle avoidance work in autonomous drones?",
    "What is geofencing and how is it implemented in drones?",
    "What is SLAM and how is it used in drone navigation?",
    "How does LoRa enable long-range drone telemetry?",
]


# =============================================================================
# COST HELPERS
# =============================================================================

INPUT_COST_PER_TOKEN  = 0.20 / 1_000_000
OUTPUT_COST_PER_TOKEN = 1.00 / 1_000_000


def calc_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        prompt_tokens     * INPUT_COST_PER_TOKEN +
        completion_tokens * OUTPUT_COST_PER_TOKEN,
        8,
    )


def calc_tps(completion_tokens: int, latency_sec: float) -> float:
    return round(completion_tokens / latency_sec, 2) if latency_sec > 0 else 0.0


def calc_tcr(tokens_method: float, tokens_baseline: float) -> float:
    """Token Compression Ratio vs baseline (none mode)."""
    return round(tokens_baseline / tokens_method, 4) if tokens_method > 0 else 1.0


# =============================================================================
# BM25 PIPELINE  — mirrors entropy/simple/none but uses BM25 retrieval
# =============================================================================

def run_bm25_pipeline(query: str) -> dict:
    """
    Run query through BM25 retrieval + top-15 selection + LLM.
    Returns same shape as run_pipeline() output["gated"].
    """
    t0 = time.time()

    # Get all memory content as corpus
    from memory_storage.storage import _store
    keys = _store.redis.keys("mem:*")

    sentences = []
    for key in keys:
        data = _store.redis.hgetall(key)
        content = data.get(b"content", b"").decode()
        if content:
            sentences.append(content)

    if not sentences:
        return {
            "response_text":    "",
            "prompt_tokens":    0,
            "completion_tokens":0,
            "total_time_sec":   0,
            "error":            "no memories in store",
            "gating_stats":     {"strategy": "bm25", "window_size": 0},
        }

    # BM25 retrieval
    candidates = bm25_retrieve(sentences, query, source="memory", doc_name="redis_memory")
    window     = candidates[:15]

    prompt        = build_prompt(window, query)
    prompt_tokens = max(1, len(prompt) // 4)

    try:
        resp              = call_llm(prompt)
        response_text     = resp.get("response_text", "")
        prompt_tokens     = resp.get("prompt_tokens") or prompt_tokens
        completion_tokens = resp.get("completion_tokens", 0)
        error             = resp.get("error", "")
    except Exception as e:
        response_text     = ""
        completion_tokens = 0
        error             = str(e)

    return {
        "response_text":    response_text,
        "prompt_tokens":    prompt_tokens,
        "completion_tokens":completion_tokens,
        "total_time_sec":   round(time.time() - t0, 3),
        "error":            error,
        "gating_stats": {
            "strategy":      "bm25",
            "window_size":   len(window),
            "candidates_in": len(candidates),
        },
    }


# =============================================================================
# SINGLE QUERY RUNNER
# =============================================================================

def run_query(query: str, mode: str, model: str = None) -> dict:
    """
    Run one query in one gating mode. Returns a flat result dict.
    """
    t0 = time.time()

    if mode == "bm25":
        raw = run_bm25_pipeline(query)
        g   = raw
        gs  = raw.get("gating_stats", {})
    else:
        result = run_pipeline(query, gating_mode=mode)
        g      = result.get("gated", {})
        gs     = result.get("gating_stats", {})

    pt  = g.get("prompt_tokens",     0)
    ct  = g.get("completion_tokens", 0)
    ts  = g.get("total_time_sec",    0) or 0

    return {
        # Identity
        "query":             query,
        "gating_mode":       mode,
        "model":             model or "default",
        "timestamp":         datetime.utcnow().isoformat() + "Z",

        # Token metrics
        "prompt_tokens":     pt,
        "completion_tokens": ct,
        "total_tokens":      pt + ct,

        # Time metrics
        "latency_sec":       round(ts, 3),
        "tokens_per_second": calc_tps(ct, ts),

        # Cost metrics
        "hypothetical_cost": calc_cost(pt, ct),
        "cost_of_pass":      calc_cost(pt, ct),

        # Window metrics
        "window_size":       gs.get("window_size",   0),
        "candidates_in":     gs.get("candidates_in", 0),
        "pruned":            gs.get("pruned",        0),
        "plateau_at":        gs.get("plateau_at",    None),

        # Entropy metrics (entropy mode only)
        "window_entropy":    None,   # filled below if available
        "is_stable":         None,

        # Response
        "response_text":     g.get("response_text", "")[:500],
        "error":             g.get("error", ""),
    }


# =============================================================================
# AGGREGATE SUMMARY
# =============================================================================

def summarise(records: list, modes: list) -> dict:
    summary = {}
    for mode in modes:
        mrs = [r for r in records if r["gating_mode"] == mode]
        if not mrs:
            continue

        n              = len(mrs)
        avg_pt         = round(sum(r["prompt_tokens"]     for r in mrs) / n, 1)
        avg_ct         = round(sum(r["completion_tokens"] for r in mrs) / n, 1)
        avg_lat        = round(sum(r["latency_sec"]       for r in mrs) / n, 3)
        avg_tps        = round(sum(r["tokens_per_second"] for r in mrs) / n, 2)
        avg_cost       = round(sum(r["hypothetical_cost"] for r in mrs) / n, 8)
        avg_win        = round(sum(r["window_size"]       for r in mrs) / n, 2)
        avg_cands      = round(sum(r["candidates_in"]     for r in mrs) / n, 2)
        total_errors   = sum(1 for r in mrs if r["error"])

        summary[mode] = {
            "n":                    n,
            "avg_prompt_tokens":    avg_pt,
            "avg_completion_tokens":avg_ct,
            "avg_latency_sec":      avg_lat,
            "avg_tokens_per_second":avg_tps,
            "avg_hypothetical_cost":avg_cost,
            "avg_window_size":      avg_win,
            "avg_candidates_in":    avg_cands,
            "total_errors":         total_errors,
            "token_compression_ratio": 1.0,   # filled after all modes computed
        }

    # TCR relative to "none" baseline
    baseline_tokens = summary.get("none", {}).get("avg_prompt_tokens", 0)
    for mode, s in summary.items():
        mt = s.get("avg_prompt_tokens", 0)
        s["token_compression_ratio"] = calc_tcr(mt, baseline_tokens)

    return summary


def print_summary_table(summary: dict, modes: list):
    print(f"\n{'='*90}")
    print(f"  MASS QUERY RESULTS SUMMARY")
    print(f"{'='*90}")
    hdr = (f"  {'Mode':<10}  {'Queries':>7}  {'Tokens':>7}  {'Lat(s)':>7}  "
           f"{'TPS':>6}  {'Cost':>12}  {'WinSz':>6}  {'TCR':>6}  {'Errors':>6}")
    print(hdr)
    print(f"  {'-'*88}")
    for mode in modes:
        s = summary.get(mode, {})
        if not s:
            continue
        print(
            f"  {mode:<10}"
            f"  {s['n']:>7}"
            f"  {s['avg_prompt_tokens']:>7.0f}"
            f"  {s['avg_latency_sec']:>7.3f}"
            f"  {s['avg_tokens_per_second']:>6.1f}"
            f"  ${s['avg_hypothetical_cost']:>11.8f}"
            f"  {s['avg_window_size']:>6.1f}"
            f"  {s['token_compression_ratio']:>6.4f}"
            f"  {s['total_errors']:>6}"
        )
    print(f"{'='*90}\n")


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_mass_queries(
    modes:       list = None,
    query_limit: int  = None,
    model:       str  = None,
    output_path: str  = "results/paper_results.json",
    verbose:     bool = True,
) -> dict:
    """
    Run all queries across all gating modes and save results.

    Parameters
    ----------
    modes       : list of gating modes to test. Default: all four.
    query_limit : cap number of queries. Default: all 60.
    model       : LM Studio model string for metadata logging.
    output_path : where to save the JSON results.
    verbose     : print progress.
    """
    modes   = modes   or ["entropy", "simple", "none", "bm25", "joint", "quantum", "hybrid"]
    queries = QUERY_BANK[:query_limit] if query_limit else QUERY_BANK

    total   = len(queries) * len(modes)
    records = []
    idx     = 0

    if verbose:
        print(f"\n{'='*70}")
        print(f"  MASS QUERY RUNNER")
        print(f"{'='*70}")
        print(f"  Gating modes  : {modes}")
        print(f"  Queries       : {len(queries)}")
        print(f"  Total runs    : {total}")
        print(f"  Model         : {model or 'LM Studio default'}")
        print(f"  Output        : {output_path}")
        print(f"{'='*70}\n")

    for mode in modes:
        if verbose:
            print(f"\n── Mode: {mode} ─────────────────────────────────────────")

        for query in queries:
            idx += 1
            t_start = time.time()

            try:
                record = run_query(query, mode, model)
            except Exception as e:
                record = {
                    "query":             query,
                    "gating_mode":       mode,
                    "model":             model or "default",
                    "timestamp":         datetime.utcnow().isoformat() + "Z",
                    "prompt_tokens":     0,
                    "completion_tokens": 0,
                    "total_tokens":      0,
                    "latency_sec":       round(time.time() - t_start, 3),
                    "tokens_per_second": 0.0,
                    "hypothetical_cost": 0.0,
                    "cost_of_pass":      0.0,
                    "window_size":       0,
                    "candidates_in":     0,
                    "pruned":            0,
                    "plateau_at":        None,
                    "window_entropy":    None,
                    "is_stable":         None,
                    "response_text":     "",
                    "error":             str(e),
                }

            records.append(record)

            if verbose:
                status = "✓" if not record["error"] else "✗"
                print(
                    f"  [{idx:03d}/{total}] {status} "
                    f"mode={mode:<8} "
                    f"tok={record['prompt_tokens']:>5}  "
                    f"win={record['window_size']:>3}  "
                    f"lat={record['latency_sec']:.2f}s  "
                    f"Q: {query[:55]}"
                )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    summary = summarise(records, modes)

    if verbose:
        print_summary_table(summary, modes)

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "run_metadata": {
            "timestamp":    datetime.utcnow().isoformat() + "Z",
            "model":        model or "LM Studio default",
            "modes":        modes,
            "n_queries":    len(queries),
            "total_runs":   total,
            "seed":         42,
        },
        "summary":     summary,
        "all_results": records,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    if verbose:
        print(f"  Saved → {output_path}\n")

    return output


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mass query runner for paper results")
    parser.add_argument("--modes", nargs="+",
                    default=["entropy", "simple", "none", "bm25", "joint", "quantum", "hybrid"],
                    choices=["entropy", "simple", "none", "bm25", "joint", "quantum", "hybrid"],
                    help="Gating modes to test")
    parser.add_argument("--queries", type=int, default=None,
                        help="Max queries to run (default: all 60)")
    parser.add_argument("--model",   type=str, default=None,
                        help="LM Studio model string for metadata")
    parser.add_argument("--output",  type=str,
                        default="results/paper_results.json",
                        help="Output JSON path")
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    run_mass_queries(
        modes       = args.modes,
        query_limit = args.queries,
        model       = args.model,
        output_path = args.output,
        verbose     = not args.quiet,
    )
