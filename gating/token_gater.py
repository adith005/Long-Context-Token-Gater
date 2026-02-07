import re
from collections import Counter
from memory.history_db import retrieve_similar
from rag.retriever import retrieve

# ==============================================================================
# 1. Core Gating Logic (Retrieval-based)
# ==============================================================================

def gate(query, memory_weight=0.6, doc_weight=0.4, top_k=5):
    """
    Retrieves and merges context from memory and documents based on weighted scores.
    """
    mem_results = retrieve_similar(query, k=top_k)
    doc_results = retrieve(query, k=top_k)

    # Merge and score
    merged = []
    for m in mem_results:
        merged.append({"source": "memory", "text": m["response"], "score": memory_weight})
    for d in doc_results:
        merged.append({"source": "document", "text": d, "score": doc_weight})

    # Simple heuristic scoring and sort
    merged.sort(key=lambda x: x["score"], reverse=True)
    return [m["text"] for m in merged[:top_k]]

# ==============================================================================
# 2. Token Optimization Gating Strategies
# ==============================================================================

def apply_stepwise_gating(prompt_text, keep_ratio=0.7):
    """Simple truncation based on a ratio of total words."""
    words = prompt_text.split()
    keep_count = int(len(words) * keep_ratio)
    return " ".join(words[:keep_count])

def apply_hybrid_scoring_gating(prompt_text):
    """
    Scores words based on position, keyword relevance, and length.
    Returns the text reconstructed from highest scoring words.
    """
    keyword_set = {"benefits", "architecture", "quantum", "python", "function"}
    words = prompt_text.split()
    if not words:
        return ""

    scored_words = []
    for i, w in enumerate(words):
        pos_score = 1.0 - (i / len(words))
        # Strip punctuation for keyword matching
        clean_w = w.lower().strip(",.?!:;")
        keyword_score = 1.0 if clean_w in keyword_set else 0.0
        length_score = min(len(w) / 10, 1.0)
        
        # Weighted total: 50% position, 30% keywords, 20% word complexity
        total_score = (0.5 * pos_score) + (0.3 * keyword_score) + (0.2 * length_score)
        scored_words.append((w, total_score))
    
    # Reconstruct based on importance (keeping original order can be done via index sorting)
    selected = sorted(scored_words, key=lambda x: x[1], reverse=True)
    selected_words = [w for w, _ in selected]
    return " ".join(selected_words)

def apply_diversity_gating(prompt_text, keep_ratio=0.7):
    """Filters redundant words while maintaining a specified keep ratio."""
    words = prompt_text.split()
    keep_count = int(len(words) * keep_ratio)
    unique_words = []
    seen = set()
    
    for w in words:
        clean_w = w.lower().strip(",.?!:;")
        if clean_w not in seen:
            unique_words.append(w)
            seen.add(clean_w)
        if len(unique_words) >= keep_count:
            break
    return " ".join(unique_words)

def apply_token_budget_gating(prompt_text, budget=64, scoring="hybrid"):
    """
    Enforces a strict word budget using either positional clipping or importance scoring.
    """
    words = prompt_text.split()
    if len(words) <= budget:
        return prompt_text

    if scoring == "position":
        return " ".join(words[:budget])
    
    elif scoring == "hybrid":
        keyword_set = {"benefits", "architecture", "quantum", "python", "function"}
        scored_words = []
        for i, w in enumerate(words):
            pos_score = 1.0 - (i / len(words))
            clean_w = w.lower().strip(",.?!:;")
            keyword_score = 1.0 if clean_w in keyword_set else 0.0
            length_score = min(len(w) / 10, 1.0)
            total_score = (0.5 * pos_score) + (0.3 * keyword_score) + (0.2 * length_score)
            scored_words.append((w, total_score))
        
        # Take the top N words within budget
        selected = sorted(scored_words, key=lambda x: x[1], reverse=True)[:budget]
        return " ".join([w for w, _ in selected])
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")