"""
pipeline.py

Central orchestration layer.
Accepts an optional PipelineTracer to record every step in real time.
"""

import time

from app_io.input_handler import process_chat_input
from app_io.output_handler import process_output

from utils.embedding import embed

from memory_storage.storage import retrieve_memory, remember_exchange
from doc_storage.document_store import retrieve_all as retrieve_docs_from_store

from gating.token_gater import build_context_window
from prompt_creator.builder import build_prompt
from llm_handler.handler import call_llm


def run_pipeline(user_query: str, gating_mode: str = "entropy", tracer=None) -> dict:
    """
    Execute full RAG + gating pipeline.

    Parameters
    ----------
    user_query   : raw user input string
    gating_mode  : "entropy" | "simple" | "none"
    tracer       : optional PipelineTracer instance — pass one in to get
                   a live event log of every step (input, output, timing).
                   If None, pipeline runs normally with zero overhead.
    """

    def trace(step, name, data, status="ok"):
        if tracer:
            tracer.log(step, name, data, status)

    start_time = time.time()

    # ── 1. Normalize Input ────────────────────────────────────────────────────
    input_obj = process_chat_input(user_query)
    query     = input_obj["user_query"]

    trace(1, "Input Normalization", {
        "raw_input":    user_query,
        "normalized":   query,
        "request_id":   input_obj["request_id"],
        "timestamp":    input_obj["timestamp"],
    })

    # ── 2. Embed Query ────────────────────────────────────────────────────────
    q_vec = embed(query)

    trace(2, "Query Embedding", {
        "model":      "all-MiniLM-L6-v2",
        "vector_dim": len(q_vec),
        "vector_preview": [round(float(x), 4) for x in q_vec[:6]],
    })

    # ── 3. Retrieve Candidates ────────────────────────────────────────────────
    memory_items = retrieve_memory(q_vec)
    doc_results  = retrieve_docs_from_store(query, top_k=10)

    for item in doc_results:
        item["confidence"] = item.get("score", 0) * 100

    candidates = memory_items + doc_results

    trace(3, "Candidate Retrieval", {
        "memory_hits":     len(memory_items),
        "doc_hits":        len(doc_results),
        "total_candidates": len(candidates),
        "top_5": [
            {
                "source":     c.get("source") or c.get("doc_name", "?"),
                "confidence": round(c.get("confidence", 0), 2),
                "preview":    (c.get("content") or c.get("sentence", ""))[:80],
            }
            for c in sorted(candidates, key=lambda x: x.get("confidence", 0), reverse=True)[:5]
        ],
    })

    # ── 4. Gating ─────────────────────────────────────────────────────────────
    if gating_mode == "entropy":
        context        = build_context_window(candidates)
        selected_items = context["window"]

        trace(4, "Token Gating (entropy)", {
            "mode":           "entropy",
            "candidates_in":  len(candidates),
            "window_size":    len(selected_items),
            "window_entropy": context.get("window_entropy"),
            "is_stable":      context.get("is_stable"),
            "pruned":         context["stats"].get("pruned", 0),
            "plateau_at":     context["stats"].get("plateau_at"),
            "selected": [
                {
                    "confidence": round(i.get("confidence", 0), 2),
                    "source":     i.get("source") or i.get("doc_name", "?"),
                    "preview":    (i.get("content") or i.get("sentence", ""))[:80],
                }
                for i in selected_items
            ],
        })

    elif gating_mode == "simple":
        selected_items = sorted(
            candidates,
            key=lambda x: x.get("confidence", 0),
            reverse=True
        )[:15]
        context = {"stats": {"strategy": "simple"}}

        trace(4, "Token Gating (simple)", {
            "mode":          "simple",
            "candidates_in": len(candidates),
            "window_size":   len(selected_items),
            "selected": [
                {
                    "confidence": round(i.get("confidence", 0), 2),
                    "source":     i.get("source") or i.get("doc_name", "?"),
                    "preview":    (i.get("content") or i.get("sentence", ""))[:80],
                }
                for i in selected_items
            ],
        })

    else:  # none
        selected_items = candidates
        context = {"stats": {"strategy": "none"}}

        trace(4, "Token Gating (none)", {
            "mode":          "none",
            "candidates_in": len(candidates),
            "window_size":   len(selected_items),
        })

    # ── 5. Prompt Creation ────────────────────────────────────────────────────
    prompt = build_prompt(selected_items, query)

    trace(5, "Prompt Assembly", {
        "context_items":   len(selected_items),
        "prompt_chars":    len(prompt),
        "estimated_tokens": max(1, len(prompt) // 4),
        "prompt_preview":  prompt[:300] + ("..." if len(prompt) > 300 else ""),
    })

    # ── 6. LLM Call ───────────────────────────────────────────────────────────
    llm_response = call_llm(prompt)
    total_time   = time.time() - start_time

    llm_response.setdefault("prompt_tokens", 0)
    llm_response.setdefault("completion_tokens", 0)
    llm_response.setdefault("response_text", "")
    llm_response.setdefault("total_time_sec", total_time)

    trace(6, "LLM Response", {
        "prompt_tokens":     llm_response["prompt_tokens"],
        "completion_tokens": llm_response["completion_tokens"],
        "latency_sec":       round(llm_response["total_time_sec"], 3),
        "response_preview":  llm_response["response_text"][:300],
        "error":             llm_response.get("error", None),
    }, status="error" if llm_response.get("error") else "ok")

    # ── 7. Store Memory ───────────────────────────────────────────────────────
    remember_exchange(query, llm_response["response_text"])

    trace(7, "Memory Storage", {
        "stored_as":         "single exchange unit",
        "user_preview":      query[:80],
        "assistant_preview": llm_response["response_text"][:80],
    })

    # ── 8. Output Formatting ──────────────────────────────────────────────────
    output = process_output(
        gated_response     = llm_response,
        non_gated_response = llm_response,
        gating_stats       = context.get("stats", {}),
    )

    trace(8, "Output Ready", {
        "total_pipeline_sec": round(time.time() - start_time, 3),
        "gating_mode":        gating_mode,
        "response_chars":     len(llm_response["response_text"]),
    })

    return output
