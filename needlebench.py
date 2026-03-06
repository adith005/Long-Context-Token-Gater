"""
needlebench.py  —  NeedleBench-style Evaluation for Token Gater
================================================================

NeedleBench tests the pipeline's ability to retrieve and answer
questions about facts ("needles") buried in long documents ("haystacks").

Three axes measured:
  1. Recall@k     — was the needle sentence retrieved into the context window?
  2. Answer Score — did the LLM correctly answer using the needle?
  3. Token Cost   — how many prompt tokens were used (lower = better efficiency)

Gating modes compared:  entropy  |  simple  |  none

Integration
-----------
  run_benchmark(...)          →  returns structured dict, consumed by frontend
  run_single(...)             →  one needle × mode × haystack, returns TestResult
  NEEDLES / HAYSTACK_SIZES    →  shared constants used by frontend for config UI

CLI
---
  python needlebench.py                         # run all tests, retrieval-only
  python needlebench.py --llm                   # include real LLM calls
  python needlebench.py --mode entropy simple
  python needlebench.py --haystack long
  python needlebench.py --output results.json
"""

import os
import sys
import json
import time
import argparse
import random
from dataclasses import dataclass, field, asdict

import numpy as np

# ── path fix ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.embedding import embed
from gating.token_gater import build_context_window
from prompt_creator.builder import build_prompt
from llm_handler.handler import call_llm
from app_io.output_handler import process_output


# ═════════════════════════════════════════════════════════════════════════════
# NEEDLE DATASET
# ═════════════════════════════════════════════════════════════════════════════

NEEDLES = [
    {
        "id":              "nb_001",
        "fact":            "The secret launch code for Project Helios is ZETA-7742-OMEGA.",
        "question":        "What is the secret launch code for Project Helios?",
        "answer_keywords": ["ZETA-7742-OMEGA", "ZETA", "7742"],
        "depth":           "shallow",
    },
    {
        "id":              "nb_002",
        "fact":            "Dr. Amara Chen discovered the protein folding shortcut in March 1987.",
        "question":        "Who discovered the protein folding shortcut and when?",
        "answer_keywords": ["Amara Chen", "Chen", "1987", "March"],
        "depth":           "middle",
    },
    {
        "id":              "nb_003",
        "fact":            "The maximum safe operating temperature for Reactor 4-B is 847 degrees Celsius.",
        "question":        "What is the maximum safe operating temperature for Reactor 4-B?",
        "answer_keywords": ["847", "degrees", "Celsius"],
        "depth":           "deep",
    },
    {
        "id":              "nb_004",
        "fact":            "Agent Valeria Moreno uses the alias 'Nightingale' during field operations.",
        "question":        "What alias does Agent Valeria Moreno use in the field?",
        "answer_keywords": ["Nightingale", "nightingale"],
        "depth":           "middle",
    },
    {
        "id":              "nb_005",
        "fact":            "The encryption passphrase for Vault 9 is: broken-mirror-cascade-41.",
        "question":        "What is the encryption passphrase for Vault 9?",
        "answer_keywords": ["broken-mirror-cascade-41", "broken-mirror", "cascade-41"],
        "depth":           "deep",
    },
]

HAYSTACK_SIZES = {
    "short":  15,
    "medium": 40,
    "long":   80,
}

FILLER_POOL = [
    "The committee reviewed all submitted proposals before the final vote.",
    "Annual rainfall in the northern region averaged 340mm over the last decade.",
    "Section 4.2 of the regulation requires written consent from all parties.",
    "The bridge construction was completed six months ahead of schedule.",
    "Laboratory samples must be stored at minus twenty degrees Celsius.",
    "The quarterly report showed a seven percent increase in operating costs.",
    "All vehicles must undergo inspection before crossing the border checkpoint.",
    "The archaeological dig revealed pottery fragments dating to the 3rd century.",
    "Network latency must remain below fifty milliseconds for real-time use.",
    "Staff members are required to complete annual safety training by December.",
    "The satellite achieved stable orbit at an altitude of 420 kilometres.",
    "Water quality tests indicated elevated phosphate levels in the eastern basin.",
    "The merger agreement was signed by both boards on the fifteenth of June.",
    "Wind turbine efficiency drops significantly when temperatures fall below zero.",
    "Historical records show the town was founded by settlers in 1802.",
    "The clinical trial enrolled 1,200 participants across five medical centres.",
    "Emergency evacuation routes must be posted in all public-facing corridors.",
    "The new firmware update resolves a critical authentication vulnerability.",
    "Peak electricity demand typically occurs between 6 and 9 pm on weekdays.",
    "Customs declarations are mandatory for all shipments exceeding 1,000 euros.",
    "The compiler optimisation reduced average build times by thirty percent.",
    "All patient records are encrypted using AES-256 before storage.",
    "The telescope's primary mirror measures 6.5 metres in diameter.",
    "Training datasets were balanced to ensure equal class representation.",
    "The treaty was ratified by twelve member states within the first year.",
    "Soil samples from grid sector C showed unusually high nitrogen content.",
    "The pilot programme was extended for another six months pending review.",
    "Revenue from subscriptions now accounts for sixty percent of total income.",
    "The algorithm's time complexity is O(n log n) in the average case.",
    "All outbound communications are logged and retained for 90 days.",
]


# ═════════════════════════════════════════════════════════════════════════════
# HAYSTACK BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_haystack(needle: dict, size: str = "medium", seed: int = 42) -> tuple:
    """Returns (sentences, needle_position_index)."""
    rng = random.Random(seed)
    n_fillers = HAYSTACK_SIZES[size]

    fillers = [FILLER_POOL[i % len(FILLER_POOL)] for i in range(n_fillers)]
    rng.shuffle(fillers)

    if needle["depth"] == "shallow":
        pos = rng.randint(0, max(1, n_fillers // 5))
    elif needle["depth"] == "middle":
        pos = rng.randint(n_fillers * 2 // 5, n_fillers * 3 // 5)
    else:
        pos = rng.randint(n_fillers * 4 // 5, n_fillers)

    sentences = fillers[:pos] + [needle["fact"]] + fillers[pos:]
    return sentences, pos


# ═════════════════════════════════════════════════════════════════════════════
# CANDIDATE BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_candidates(sentences: list, query: str) -> list:
    """
    Embed haystack sentences and score against query.
    Produces the same candidate dict shape as pipeline.py step 3.
    """
    q_vec  = embed(query)
    q_norm = np.linalg.norm(q_vec)

    candidates = []
    for sent in sentences:
        s_vec  = embed(sent)
        s_norm = np.linalg.norm(s_vec)
        sim    = float(np.dot(q_vec, s_vec) / (q_norm * s_norm + 1e-9))
        candidates.append({
            "sentence":   sent,
            "content":    sent,
            "confidence": sim * 100,
            "source":     "needlebench",
            "doc_name":   "needlebench_haystack",
        })

    candidates.sort(key=lambda x: x["confidence"], reverse=True)
    return candidates


# ═════════════════════════════════════════════════════════════════════════════
# GATING  — mirrors pipeline.py step 4 exactly
# ═════════════════════════════════════════════════════════════════════════════

def gate(candidates: list, mode: str) -> tuple:
    if mode == "entropy":
        result = build_context_window(candidates)
        return result["window"], result["stats"]
    elif mode == "simple":
        selected = candidates[:15]
        return selected, {"strategy": "simple", "window_size": len(selected)}
    else:
        return candidates, {"strategy": "none", "window_size": len(candidates)}


# ═════════════════════════════════════════════════════════════════════════════
# SCORING
# ═════════════════════════════════════════════════════════════════════════════

def needle_in_window(needle_fact: str, window: list) -> bool:
    for item in window:
        text = item.get("sentence") or item.get("content") or ""
        if needle_fact.strip().lower() in text.strip().lower():
            return True
    return False


def score_answer(response_text: str, keywords: list) -> float:
    text = response_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return round(hits / len(keywords), 4) if keywords else 0.0


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ═════════════════════════════════════════════════════════════════════════════
# TEST RESULT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    needle_id:         str
    depth:             str
    haystack_size:     str
    gating_mode:       str
    needle_recalled:   bool
    answer_score:      float
    prompt_tokens:     int
    completion_tokens: int
    latency_sec:       float
    window_size:       int
    candidates_in:     int
    gating_stats:      dict = field(default_factory=dict)
    response_text:     str  = ""
    error:             str  = ""

    def to_pipeline_output(self) -> dict:
        """
        Wraps result in process_output() shape so the frontend can reuse
        its existing display logic for gated/non_gated/gating_stats.
        """
        resp = {
            "response_text":    self.response_text,
            "prompt_tokens":    self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_time_sec":   self.latency_sec,
        }
        return process_output(
            gated_response     = resp,
            non_gated_response = resp,
            gating_stats       = {
                **self.gating_stats,
                "needle_recalled": self.needle_recalled,
                "answer_score":    self.answer_score,
                "window_size":     self.window_size,
                "candidates_in":   self.candidates_in,
            },
        )


# ═════════════════════════════════════════════════════════════════════════════
# SINGLE TEST
# ═════════════════════════════════════════════════════════════════════════════

def run_single(
    needle:        dict,
    haystack_size: str,
    gating_mode:   str,
    call_llm_flag: bool = False,
    seed:          int  = 42,
) -> TestResult:
    """
    One needle × haystack × gating_mode.
    Uses gating/token_gater, prompt_creator/builder, llm_handler/handler
    — the same modules as pipeline.py.
    """
    sentences, _ = build_haystack(needle, haystack_size, seed)
    query = needle["question"]

    t0 = time.time()

    candidates    = build_candidates(sentences, query)
    window, stats = gate(candidates, gating_mode)
    recalled      = needle_in_window(needle["fact"], window)
    prompt        = build_prompt(window, query)
    prompt_tokens = estimate_tokens(prompt)

    response_text     = ""
    completion_tokens = 0
    answer_score      = 0.0
    error             = ""

    if call_llm_flag:
        try:
            llm_resp          = call_llm(prompt)
            response_text     = llm_resp.get("response_text", "")
            prompt_tokens     = llm_resp.get("prompt_tokens") or prompt_tokens
            completion_tokens = llm_resp.get("completion_tokens", 0)
            answer_score      = score_answer(response_text, needle["answer_keywords"])
        except Exception as e:
            error        = str(e)
            answer_score = 0.0
    else:
        answer_score = 1.0 if recalled else 0.0

    return TestResult(
        needle_id         = needle["id"],
        depth             = needle["depth"],
        haystack_size     = haystack_size,
        gating_mode       = gating_mode,
        needle_recalled   = recalled,
        answer_score      = answer_score,
        prompt_tokens     = prompt_tokens,
        completion_tokens = completion_tokens,
        latency_sec       = round(time.time() - t0, 3),
        window_size       = len(window),
        candidates_in     = len(candidates),
        gating_stats      = stats,
        response_text     = response_text[:300] if response_text else "",
        error             = error,
    )


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER  — called by frontend AND CLI
# ═════════════════════════════════════════════════════════════════════════════

def run_benchmark(
    modes:          list = None,
    haystack_sizes: list = None,
    call_llm_flag:  bool = False,
    output_path:    str  = None,
    verbose:        bool = True,
    progress_cb          = None,   # callable(current, total, result) for Streamlit
) -> dict:
    """
    Run all needle × mode × haystack combinations.

    Returns
    -------
    {
        "summary":     { mode: { overall, by_haystack, by_depth } },
        "all_results": [ TestResult as dict, ... ]
    }

    progress_cb(current, total, result) is called after each test — use this
    to drive a Streamlit st.progress() bar in real time.
    """
    modes          = modes          or ["entropy", "simple", "none"]
    haystack_sizes = haystack_sizes or ["short", "medium", "long"]

    all_results = []
    total = len(NEEDLES) * len(modes) * len(haystack_sizes)
    idx   = 0

    if verbose:
        _print_header(modes, haystack_sizes, total, call_llm_flag)

    for mode in modes:
        for h_size in haystack_sizes:
            for needle in NEEDLES:
                idx += 1
                result = run_single(needle, h_size, mode, call_llm_flag)
                all_results.append(result)

                if verbose:
                    _print_row(idx, total, result)

                if progress_cb:
                    progress_cb(idx, total, result)

    summary = _aggregate(all_results, modes, haystack_sizes)

    if verbose:
        _print_summary(summary, modes, haystack_sizes)

    output = {
        "summary":     summary,
        "all_results": [asdict(r) for r in all_results],
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"\n  Results saved -> {output_path}")

    return output


# ═════════════════════════════════════════════════════════════════════════════
# AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

def _aggregate(results, modes, haystack_sizes) -> dict:
    summary = {}
    for mode in modes:
        mr = [r for r in results if r.gating_mode == mode]
        summary[mode] = {
            "overall":     _agg(mr),
            "by_haystack": {h: _agg([r for r in mr if r.haystack_size == h])
                            for h in haystack_sizes},
            "by_depth":    {d: _agg([r for r in mr if r.depth == d])
                            for d in ["shallow", "middle", "deep"]},
        }
    return summary


def _agg(results) -> dict:
    if not results:
        return {}
    n = len(results)
    return {
        "n":                 n,
        "recall_rate":       round(sum(r.needle_recalled for r in results) / n, 4),
        "avg_answer_score":  round(sum(r.answer_score    for r in results) / n, 4),
        "avg_prompt_tokens": round(sum(r.prompt_tokens   for r in results) / n, 1),
        "avg_window_size":   round(sum(r.window_size     for r in results) / n, 2),
        "avg_latency_sec":   round(sum(r.latency_sec     for r in results) / n, 3),
    }


# ═════════════════════════════════════════════════════════════════════════════
# CLI PRINTERS
# ═════════════════════════════════════════════════════════════════════════════

def _print_header(modes, haystack_sizes, total, call_llm_flag):
    print(f"\n{'='*64}")
    print(f"  NeedleBench -- Token Gater Evaluation")
    print(f"{'='*64}")
    print(f"  Needles       : {len(NEEDLES)}")
    print(f"  Gating modes  : {modes}")
    print(f"  Haystack sizes: {haystack_sizes}")
    print(f"  LLM calls     : {'yes' if call_llm_flag else 'no  (retrieval-only)'}")
    print(f"  Total tests   : {total}")
    print(f"{'='*64}\n")


def _print_row(idx, total, r: TestResult):
    icon = "YES" if r.needle_recalled else "NO "
    print(
        f"  [{idx:02d}/{total}] "
        f"{r.gating_mode:8s} | {r.haystack_size:6s} | {r.needle_id} | "
        f"depth={r.depth:7s} | recall={icon}  score={r.answer_score:.2f}  "
        f"win={r.window_size:3d}/{r.candidates_in}  tok~{r.prompt_tokens}"
    )


def _print_summary(summary: dict, modes, haystack_sizes):
    print(f"\n{'='*64}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*64}")
    print(f"\n  -- Overall --")
    print(f"  {'Mode':<10}  {'Recall':>8}  {'Score':>8}  {'Tokens':>8}  {'WinSz':>6}")
    print(f"  {'-'*52}")
    for mode in modes:
        s = summary[mode]["overall"]
        print(f"  {mode:<10}  {s['recall_rate']:>8.1%}  {s['avg_answer_score']:>8.3f}"
              f"  {s['avg_prompt_tokens']:>8.0f}  {s['avg_window_size']:>6.1f}")

    print(f"\n  -- Recall by Haystack Size --")
    print(f"  {'Mode':<10}" + "".join(f"  {h:>8}" for h in haystack_sizes))
    for mode in modes:
        row = f"  {mode:<10}"
        for h in haystack_sizes:
            r = summary[mode]["by_haystack"].get(h, {}).get("recall_rate", 0)
            row += f"  {r:>8.1%}"
        print(row)

    print(f"\n  -- Token Cost by Haystack Size --")
    print(f"  {'Mode':<10}" + "".join(f"  {h:>8}" for h in haystack_sizes))
    for mode in modes:
        row = f"  {mode:<10}"
        for h in haystack_sizes:
            t = summary[mode]["by_haystack"].get(h, {}).get("avg_prompt_tokens", 0)
            row += f"  {t:>8.0f}"
        print(row)

    print(f"\n{'='*64}\n")


# ═════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeedleBench for Token Gater")
    parser.add_argument("--mode",     nargs="+", default=["entropy", "simple", "none"],
                        choices=["entropy", "simple", "none"])
    parser.add_argument("--haystack", nargs="+", default=["short", "medium", "long"],
                        choices=["short", "medium", "long"])
    parser.add_argument("--llm",      action="store_true",
                        help="Real LLM calls (requires LM Studio on localhost:1234)")
    parser.add_argument("--output",   default="needlebench_results.json")
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()

    run_benchmark(
        modes          = args.mode,
        haystack_sizes = args.haystack,
        call_llm_flag  = args.llm,
        output_path    = args.output,
        verbose        = not args.quiet,
    )
