"""
pipeline.py

Central orchestration layer.
"""

import time

from app_io.input_handler import process_chat_input
from app_io.output_handler import process_output

from utils.embedding import embed

from memory_storage.storage import retrieve_memory, remember
from doc_storage.document_store import retrieve_all as retrieve_docs_from_store

from gating.token_gater import build_context_window
from prompt_creator.builder import build_prompt
from llm_handler.handler import call_llm


def run_pipeline(user_query: str, gating_mode: str = "entropy") -> dict:
    """
    Execute full RAG + gating pipeline.
    """

    start_time = time.time()

    # 1️⃣ Normalize Input
    input_obj = process_chat_input(user_query)
    query = input_obj["user_query"]

    # 2️⃣ Embed Once
    q_vec = embed(query)

    # 3️⃣ Retrieve Candidates
    memory_items = retrieve_memory(q_vec)
    
    # doc_storage.retrieve_all expects a string query for now
    doc_results = retrieve_docs_from_store(query, top_k=10)
    
    # Normalize doc items to have 'confidence' key for the gater (using their score)
    for item in doc_results:
        item["confidence"] = item.get("score", 0) * 100 

    candidates = memory_items + doc_results

    # 4️⃣ Gating
    if gating_mode == "entropy":
        context = build_context_window(candidates)
        selected_items = context["window"]

    elif gating_mode == "simple":
        selected_items = sorted(
            candidates,
            key=lambda x: x.get("confidence", 0),
            reverse=True
        )[:15]
        context = {"stats": {"strategy": "simple"}}

    else:  # none
        selected_items = candidates
        context = {"stats": {"strategy": "none"}}

    # 5️⃣ Prompt Creation
    prompt = build_prompt(selected_items, query)

    # 6️⃣ LLM Call
    llm_response = call_llm(prompt)

    total_time = time.time() - start_time

    # Ensure required keys exist
    llm_response.setdefault("prompt_tokens", 0)
    llm_response.setdefault("completion_tokens", 0)
    llm_response.setdefault("response_text", "")
    llm_response.setdefault("total_time_sec", total_time)

    # 7️⃣ Store Chat Memory
    remember("user", query)
    remember("assistant", llm_response["response_text"])

    # 8️⃣ Output Formatting
    return process_output(
        gated_response=llm_response,
        non_gated_response=llm_response,
        gating_stats=context.get("stats", {})
    )
