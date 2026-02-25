"""
gater.py  —  Entropy-Optimised Context Gater
=============================================

Goal: smallest possible context window that retains all relevant information.

Pipeline
--------
  1. Embed query once, reuse everywhere.
  2. Gate memory  (Redis LIST)   →  confidence ≥ 50 %
  3. Gate documents (Redis clusters)  →  confidence ≥ 50 %
  4. Merge candidates, rank by confidence.
  5. Entropy-guided window builder:
       a. Add items greedily while H(window) is rising or stable.
       b. Stop as soon as a plateau is detected — extra items add
          redundancy without new information.
       c. After building, prune: remove the lowest-confidence item
          if removing it keeps H above the floor AND window shrinks.
       d. Repeat prune until no further reduction is possible.
  6. Return the minimal stable window + diagnostics.

Entropy mechanics
-----------------
  H = -Σ p_i log2(p_i)   p = softmax(confidence scores in window)

  Maximum diversity   →  log2(N) bits
  Zero diversity      →  0 bits
  Stable plateau      →  ΔH < ENTROPY_DELTA_THRESH between consecutive adds

  We stop adding when new items no longer raise entropy meaningfully,
  then prune backward to find the smallest set that keeps H ≥ ENTROPY_FLOOR.

Redis layout (from document_store.py)
--------------------------------------
  <doc_name>:meta           → JSON metadata
  <doc_name>:chunk:<i>      → JSON { sentences, vectors }
  memory:turns              → Redis LIST of JSON turn objects
"""

import json
import datetime
from typing import Optional

import numpy as np
import redis
from sentence_transformers import SentenceTransformer

# ── Tunables ───────────────────────────────────────────────────────────────────
REDIS_HOST           = "localhost"
REDIS_PORT           = 6379
MODEL_NAME           = "all-MiniLM-L6-v2"
MEMORY_KEY           = "memory:turns"

CONFIDENCE_THRESHOLD = 50.0    # % — hard gate; anything below is discarded
ENTROPY_FLOOR        = 1.0     # bits — minimum acceptable window entropy
ENTROPY_DELTA_THRESH = 0.05    # bits — ΔH below this means "no new information"
MAX_WINDOW_SIZE      = 15      # absolute safety cap

# ── Singletons ─────────────────────────────────────────────────────────────────
r     = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
model = SentenceTransformer(MODEL_NAME)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE MATH
# ═══════════════════════════════════════════════════════════════════════════════

def _softmax(scores: list[float]) -> np.ndarray:
    arr = np.array(scores, dtype=np.float64)
    arr -= arr.max()                  # numerical stability
    e    = np.exp(arr)
    return e / e.sum()


def _entropy(scores: list[float]) -> float:
    """Shannon entropy in bits over softmax-normalised confidence scores."""
    if len(scores) < 2:
        return 0.0
    p = _softmax(scores)
    p = np.clip(p, 1e-12, None)
    return float(-np.sum(p * np.log2(p)))


def _embed(text: str) -> np.ndarray:
    """Single embed → L2-normalised float32 (384,)."""
    return model.encode(
        [text], convert_to_numpy=True, normalize_embeddings=True
    )[0].astype(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))        # unit vectors: dot == cosine


def _confidence(cosine_score: float) -> float:
    """cosine [-1,1]  →  confidence [0, 100] %"""
    return ((cosine_score + 1.0) / 2.0) * 100.0


def _now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY — write
# ═══════════════════════════════════════════════════════════════════════════════

def remember(role: str, content: str) -> None:
    """Persist a conversation turn with its embedding."""
    vec  = _embed(content)
    turn = {
        "role"     : role,
        "content"  : content,
        "vector"   : vec.tolist(),
        "timestamp": _now(),
    }
    r.rpush(MEMORY_KEY, json.dumps(turn))


def clear_memory() -> None:
    r.delete(MEMORY_KEY)
    print("[memory] Cleared.")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — MEMORY GATE
# ═══════════════════════════════════════════════════════════════════════════════

def gate_memory(q_vec: np.ndarray) -> list[dict]:
    """
    Score all memory turns against the pre-embedded query vector.
    Return turns with confidence >= CONFIDENCE_THRESHOLD, sorted best-first.
    """
    raw_turns = r.lrange(MEMORY_KEY, 0, -1)
    passed    = []

    for raw in raw_turns:
        turn   = json.loads(raw)
        vec    = np.array(turn["vector"], dtype=np.float32)
        cosine = _cosine(q_vec, vec)
        conf   = _confidence(cosine)

        if conf >= CONFIDENCE_THRESHOLD:
            passed.append({
                "source"    : "memory",
                "role"      : turn["role"],
                "content"   : turn["content"],
                "timestamp" : turn["timestamp"],
                "cosine"    : round(cosine, 4),
                "confidence": round(conf, 2),
            })

    passed.sort(key=lambda x: x["confidence"], reverse=True)

    print(f"[memory]  {len(raw_turns):>4} turns   →  "
          f"{len(passed)} passed  (conf ≥ {CONFIDENCE_THRESHOLD}%)")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — DOCUMENT GATE
# ═══════════════════════════════════════════════════════════════════════════════

def _doc_names() -> list[str]:
    return [
        k.replace(":meta", "")
        for k in r.keys("*:meta")
        if not k.startswith("memory")
    ]


def gate_document(
    q_vec: np.ndarray,
    doc_name: Optional[str] = None,
) -> list[dict]:
    """
    Score every sentence in every document cluster against the query.
    Return sentences with confidence >= CONFIDENCE_THRESHOLD, sorted best-first.
    """
    names   = [doc_name] if doc_name else _doc_names()
    passed  = []
    scanned = 0

    for name in names:
        meta_raw = r.get(f"{name}:meta")
        if not meta_raw:
            continue
        meta       = json.loads(meta_raw)
        num_chunks = meta.get("chunk_count", 0)

        # Bulk-fetch all chunks in one round-trip
        pipe = r.pipeline()
        for i in range(num_chunks):
            pipe.get(f"{name}:chunk:{i}")
        raw_chunks = pipe.execute()

        sent_global = 0
        for chunk_idx, raw in enumerate(raw_chunks):
            if raw is None:
                continue
            payload = json.loads(raw)

            for sentence, vec_list in zip(payload["sentences"], payload["vectors"]):
                vec    = np.array(vec_list, dtype=np.float32)
                cosine = _cosine(q_vec, vec)
                conf   = _confidence(cosine)
                scanned += 1

                if conf >= CONFIDENCE_THRESHOLD:
                    passed.append({
                        "source"        : "document",
                        "doc_name"      : name,
                        "chunk_index"   : chunk_idx,
                        "sentence_index": sent_global,
                        "cosine"        : round(cosine, 4),
                        "confidence"    : round(conf, 2),
                        "sentence"      : sentence,
                    })
                sent_global += 1

        # Update access metadata
        meta["access_count"] += 1
        meta["last_accessed"] = _now()
        r.set(f"{name}:meta", json.dumps(meta))

    passed.sort(key=lambda x: x["confidence"], reverse=True)

    print(f"[docs]    {scanned:>4} sentences →  "
          f"{len(passed)} passed  (conf ≥ {CONFIDENCE_THRESHOLD}%)")
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — ENTROPY-GUIDED MINIMAL WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

def build_context_window(
    candidates: list[dict],           # merged, sorted by confidence desc
    entropy_floor: float = ENTROPY_FLOOR,
    entropy_delta: float = ENTROPY_DELTA_THRESH,
    max_size: int = MAX_WINDOW_SIZE,
) -> dict:
    """
    Build the smallest context window that:
      • Keeps Shannon entropy >= entropy_floor  (information diversity)
      • Stops adding items once ΔH < entropy_delta  (plateau = redundancy)
      • Prunes backward: removes the weakest item if H stays above floor

    Returns
    -------
    {
        window         : list[dict],    ← final minimal context items
        window_entropy : float,         ← bits
        entropy_floor  : float,
        is_stable      : bool,
        stats          : { … }
    }
    """
    # ── Phase A: Greedy forward pass ─────────────────────────────────────────
    window        = []
    scores        = []
    prev_H        = 0.0
    stopped_early = False

    for item in candidates[:max_size]:
        trial_scores = scores + [item["confidence"]]
        trial_H      = _entropy(trial_scores)
        delta_H      = trial_H - prev_H

        # Stop if we've hit the entropy plateau (new item adds nothing)
        if len(window) >= 2 and delta_H < entropy_delta:
            stopped_early = True
            break

        window.append(item)
        scores.append(item["confidence"])
        prev_H = trial_H

    current_H = _entropy(scores)

    # ── Phase B: Backward prune ───────────────────────────────────────────────
    # Remove weakest items (end of list = lowest confidence) if H stays stable.
    pruned = 0
    while len(window) > 1:
        trial_scores = scores[:-1]
        trial_H      = _entropy(trial_scores)

        if trial_H >= entropy_floor:
            window.pop()
            scores.pop()
            current_H = trial_H
            pruned   += 1
        else:
            break                    # pruning any further would destabilise H

    is_stable  = current_H >= entropy_floor or len(window) <= 2
    mem_count  = sum(1 for i in window if i["source"] == "memory")
    doc_count  = sum(1 for i in window if i["source"] == "document")

    print(
        f"[window]  size={len(window)}  H={current_H:.3f} bits  "
        f"floor={entropy_floor} bits  stable={'✓' if is_stable else '✗'}  "
        f"pruned={pruned}  early_stop={'✓' if stopped_early else '✗'}"
    )

    return {
        "window"         : window,
        "window_entropy" : round(current_H, 4),
        "entropy_floor"  : entropy_floor,
        "is_stable"      : is_stable,
        "stats": {
            "candidates_in"  : len(candidates),
            "window_size"    : len(window),
            "pruned"         : pruned,
            "early_stop"     : stopped_early,
            "memory_count"   : mem_count,
            "doc_count"      : doc_count,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def gate(
    query: str,
    doc_name: Optional[str] = None,
    entropy_floor: float = ENTROPY_FLOOR,
    entropy_delta: float = ENTROPY_DELTA_THRESH,
    max_size: int = MAX_WINDOW_SIZE,
) -> dict:
    """
    Full gater: embed once → gate memory → gate docs → build minimal window.

    Parameters
    ----------
    query         : user's current query
    doc_name      : restrict doc search to one cluster  (None = all)
    entropy_floor : minimum H(bits) the window must sustain after pruning
    entropy_delta : ΔH below this = plateau, stop adding
    max_size      : absolute cap before entropy logic runs

    Returns
    -------
    {
        query, window, window_entropy, entropy_floor,
        is_stable, stats
    }
    """
    print(f'\n{"═"*62}')
    print(f'  query : "{query}"')
    print(f'{"═"*62}')

    # Single embed — shared by both gates
    q_vec = _embed(query)

    memory_items = gate_memory(q_vec)
    doc_items    = gate_document(q_vec, doc_name=doc_name)

    # Merge and re-sort by confidence
    candidates = sorted(
        memory_items + doc_items,
        key=lambda x: x["confidence"],
        reverse=True,
    )

    context          = build_context_window(
                            candidates,
                            entropy_floor=entropy_floor,
                            entropy_delta=entropy_delta,
                            max_size=max_size,
                       )
    context["query"] = query
    return context


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt(context: dict) -> str:
    """Format the minimal window into a plain-text LLM context block."""
    h       = context["window_entropy"]
    stable  = "stable" if context["is_stable"] else "unstable"
    s       = context["stats"]

    lines = [
        f"=== Context  [H={h:.3f} bits | {stable} | "
        f"items={s['window_size']} | mem={s['memory_count']} doc={s['doc_count']}] ===",
        "",
    ]

    mem_items = [i for i in context["window"] if i["source"] == "memory"]
    doc_items = [i for i in context["window"] if i["source"] == "document"]

    if mem_items:
        lines.append("── Memory ───────────────────────────────────────────────")
        for item in mem_items:
            lines.append(
                f"[{item['role'].upper()}  {item['confidence']}%]  {item['content']}"
            )
        lines.append("")

    if doc_items:
        lines.append("── Documents ────────────────────────────────────────────")
        for item in doc_items:
            lines.append(
                f"[{item['doc_name']} | chunk {item['chunk_index']} | {item['confidence']}%]"
            )
            lines.append(item["sentence"])
        lines.append("")

    lines.append("── Query ────────────────────────────────────────────────")
    lines.append(context["query"])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Seed memory with a mix of relevant and irrelevant turns
    remember("user",      "What are the payment terms in the contract?")
    remember("assistant", "The contract states net-30 payment terms.")
    remember("user",      "Does it mention penalties for late payment?")
    remember("assistant", "Yes, 1.5% monthly interest applies after the due date.")
    remember("user",      "What is the weather like today?")       # off-topic
    remember("assistant", "I am not sure about the current weather.") # off-topic

    query   = "What happens if payment is late?"
    context = gate(query)

    print("\n── Window Items ─────────────────────────────────────────")
    for item in context["window"]:
        text = item.get("content") or item.get("sentence", "")
        print(f"  [{item['source']:<8} {item['confidence']:5.1f}%]  {text[:90]}")

    print(f"\n── Stats ────────────────────────────────────────────────")
    print(json.dumps(context["stats"], indent=2))

    print(f"\n── Prompt ───────────────────────────────────────────────")
    print(build_prompt(context))