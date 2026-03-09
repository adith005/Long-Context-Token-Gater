"""
locomo_adapter.py  —  LoCoMo → NeedleBench adapter
====================================================

Drop-in replacement for the hand-crafted NEEDLES and FILLER_POOL
constants in needlebench.py.

Usage
-----
    from locomo_adapter import load_locomo_needles, load_locomo_fillers

    NEEDLES      = load_locomo_needles("path/to/locomo10.json")
    FILLER_POOL  = load_locomo_fillers("path/to/locomo10.json")

    # Pass into run_benchmark / run_single exactly as before — nothing
    # else in needlebench.py needs to change.
"""

import json
import re
import random
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dia_id(dia_id: str) -> tuple:
    """'D3:7' → (session=3, line=7)"""
    m = re.match(r"D(\d+):(\d+)", dia_id)
    if not m:
        raise ValueError(f"Unrecognised dia_id format: {dia_id!r}")
    return int(m.group(1)), int(m.group(2))


def _build_utterance_map(conv: dict) -> dict:
    """Returns {dia_id: text} for every text-bearing utterance."""
    umap = {}
    for key, val in conv.items():
        if not (key.startswith("session_") and isinstance(val, list)):
            continue
        for u in val:
            if "dia_id" in u and "text" in u:
                umap[u["dia_id"]] = u["text"].strip()
    return umap


def _get_session_keys(conv: dict) -> list:
    """Sorted list of session keys that contain dialogue lists."""
    keys = [
        k for k, v in conv.items()
        if k.startswith("session_") and not k.endswith("_date_time")
        and isinstance(v, list)
    ]
    keys.sort(key=lambda k: int(k.split("_")[1]))
    return keys


def _depth_from_session(session_num: int, total_sessions: int) -> str:
    """Map a session number to shallow / middle / deep by thirds."""
    third = max(1, total_sessions // 3)
    if session_num <= third:
        return "shallow"
    elif session_num <= third * 2:
        return "middle"
    else:
        return "deep"


def _answer_to_keywords(answer) -> list:
    """
    Convert answer (str / int / float) to a keyword list.
    First keyword is the full answer string; remaining are individual tokens.
    """
    if isinstance(answer, (int, float)):
        return [str(answer)]
    text = str(answer)
    tokens = [t.strip() for t in re.split(r"[,\s]+", text) if len(t.strip()) > 2]
    seen, out = set(), []
    for kw in [text] + tokens:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            out.append(kw)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def load_locomo_needles(
    json_path: str,
    categories: tuple = (1, 2),
    max_needles: int = 30,
    balanced_depth: bool = True,
    single_evidence_only: bool = True,
    seed: int = 42,
) -> list:
    """
    Build a NEEDLES-compatible list from LoCoMo QA pairs.

    Parameters
    ----------
    json_path            : path to locomo10.json
    categories           : QA categories to include.
                             1 = single-hop fact  (best for retrieval tests)
                             2 = temporal          (best for retrieval tests)
                             3 = multi-hop         (harder, avoid for basic eval)
                             4 = open-ended        (avoid — no clean answer)
                             5 = adversarial       (avoid — unanswerable)
    max_needles          : total needles to return
    balanced_depth       : if True, sample equal counts from shallow/middle/deep
    single_evidence_only : skip QAs that need more than one evidence utterance
    seed                 : random seed for reproducibility when sampling

    Returns
    -------
    List of needle dicts with keys:
        id, fact, question, answer_keywords, depth, sample_idx, evidence_ids
    """
    data = json.loads(Path(json_path).read_text())

    # ── collect ALL valid needles across all samples ──────────────────────────
    by_depth = defaultdict(list)

    for sample_idx, sample in enumerate(data):
        conv   = sample["conversation"]
        umap   = _build_utterance_map(conv)
        n_sess = len(_get_session_keys(conv))

        for qi, qa in enumerate(sample["qa"]):
            if qa.get("category") not in categories:
                continue
            evidence = qa.get("evidence", [])
            if not evidence:
                continue
            if single_evidence_only and len(evidence) > 1:
                continue

            primary_ev = evidence[0]
            fact_text  = umap.get(primary_ev)
            if not fact_text:
                continue

            try:
                sess_num, _ = _parse_dia_id(primary_ev)
            except ValueError:
                continue

            depth = _depth_from_session(sess_num, n_sess)
            by_depth[depth].append({
                "id":              f"lc{sample_idx:02d}_q{qi:03d}",
                "fact":            fact_text,
                "question":        qa["question"],
                "answer_keywords": _answer_to_keywords(qa["answer"]),
                "depth":           depth,
                "sample_idx":      sample_idx,
                "evidence_ids":    evidence,
            })

    # ── sample ────────────────────────────────────────────────────────────────
    rng = random.Random(seed)
    needles = []

    if balanced_depth:
        # Equal share from each depth bucket, up to max_needles
        per_depth = max_needles // 3
        for depth in ("shallow", "middle", "deep"):
            pool = by_depth[depth]
            rng.shuffle(pool)
            needles.extend(pool[:per_depth])
    else:
        # Flat pool, random sample
        flat = [n for bucket in by_depth.values() for n in bucket]
        rng.shuffle(flat)
        needles = flat[:max_needles]

    return needles


def load_locomo_fillers(
    json_path: str,
    exclude_sample_idx: int = None,
) -> list:
    """
    Build a FILLER_POOL-compatible list from LoCoMo utterances.

    Collects all unique utterance texts across all samples (~5,800+),
    so any haystack size can be built without repetition.

    Parameters
    ----------
    exclude_sample_idx : optionally exclude one sample's utterances to
                         prevent leakage when its QAs are used as needles.
                         Pass None to include everything (utterances from
                         other samples still vastly outnumber needles).
    """
    data    = json.loads(Path(json_path).read_text())
    seen    = set()
    fillers = []

    for sample_idx, sample in enumerate(data):
        if sample_idx == exclude_sample_idx:
            continue
        conv = sample["conversation"]
        for key, val in conv.items():
            if not (key.startswith("session_") and isinstance(val, list)):
                continue
            for u in val:
                text = u.get("text", "").strip()
                if text and len(text) > 10 and text not in seen:
                    seen.add(text)
                    fillers.append(text)

    return fillers


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SANITY CHECK  (python locomo_adapter.py path/to/locomo10.json)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from collections import Counter

    path = sys.argv[1] if len(sys.argv) > 1 else "locomo10.json"

    needles = load_locomo_needles(path)
    fillers = load_locomo_fillers(path)

    print(f"Needles loaded : {len(needles)}")
    print(f"Filler pool    : {len(fillers):,} unique utterances")
    print(f"Depth dist     : {dict(Counter(n['depth'] for n in needles))}")
    print()
    for n in needles[:3]:
        print(f"  [{n['id']}] depth={n['depth']}")
        print(f"    fact     : {n['fact'][:80]}")
        print(f"    question : {n['question']}")
        print(f"    keywords : {n['answer_keywords']}")
        print()
