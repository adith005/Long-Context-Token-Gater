from memory.history_db import retrieve_similar
from rag.retriever import retrieve
import math
import numpy as np
from typing import List, Callable, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# ============================================================
# Entropy Utilities
# ============================================================
def shannon_entropy(probs: List[float]) -> float:
    """Compute Shannon entropy from token probabilities."""
    return -sum(p * math.log(p + 1e-9) for p in probs)

def estimate_entropy(prompt: str, llm_predict: Callable[[str], List[float]]) -> float:
    """
    Estimate model uncertainty for a given prompt.
    llm_predict must return token probabilities.
    """
    probs = llm_predict(prompt)
    return shannon_entropy(probs)

# ============================================================
# Mock LLM Predictor (replace later with real logprobs)
# ============================================================
def mock_llm_predict(prompt: str) -> List[float]:
    """
    Temporary entropy proxy.
    Replace with OpenAI / HF logprob-based predictor later.
    """
    import random
    vocab_size = 20
    probs = [random.random() for _ in range(vocab_size)]
    total = sum(probs)
    return [p / total for p in probs]

# ============================================================
# Semantic Entropy Utilities
# ============================================================
def compute_semantic_entropy(
    outputs: List[str],
    embed_model,
    similarity_threshold: float = 0.8
) -> float:
    """
    Compute semantic entropy by clustering outputs and calculating Shannon entropy.
    
    Args:
        outputs: List of generated text outputs
        embed_model: Embedding model with encode() method
        similarity_threshold: Threshold for clustering (higher = fewer clusters)
    
    Returns:
        Semantic entropy value
    """
    if len(outputs) == 0:
        return 0.0
    
    if len(outputs) == 1:
        return 0.0
    
    # Encode all outputs
    embeddings = np.array([embed_model.encode(output) for output in outputs])
    
    # Cluster by semantic similarity
    # Using agglomerative clustering with cosine distance
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - similarity_threshold,
        metric='cosine',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(embeddings)
    
    # Compute cluster probabilities
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    p_clusters = cluster_counts / len(outputs)
    
    # Compute Shannon entropy
    se = -sum(p * math.log(p + 1e-9) for p in p_clusters)
    
    return se

def softmax(scores: List[float], alpha: float = 1.0) -> np.ndarray:
    """Apply softmax with temperature parameter alpha."""
    scores_array = np.array(scores)
    exp_scores = np.exp(scores_array / alpha)
    return exp_scores / np.sum(exp_scores)

# ============================================================
# Gater with Semantic Entropy and Similarity
# ============================================================
def gater_semantic_entropy_similarity(
    query: str,
    chat_history: List[str],
    llm_sample: Callable[[str], str],
    embed_model,
    K: int = 5,
    R: int = 10,
    alpha: float = 1.0,
    lambda_param: float = 0.5,
    T: int = 5,
    similarity_threshold: float = 0.8
) -> List[str]:
    """
    Gate function using semantic similarity and semantic entropy.
    
    Args:
        query: User query
        chat_history: List of historical messages
        llm_sample: Function to sample output from LLM given context
        embed_model: Embedding model with encode() method
        K: Final number of messages to select
        R: Number of messages in shortlist (based on similarity)
        alpha: Temperature parameter for softmax
        lambda_param: Weight for similarity vs entropy (0-1)
        T: Number of output samples for entropy estimation
        similarity_threshold: Threshold for semantic clustering
    
    Returns:
        List of selected messages
    """
    if len(chat_history) == 0:
        return []
    
    # Step 1: Semantic similarity shortlist
    q_embed = embed_model.encode(query).reshape(1, -1)
    
    sim_scores = []
    for message in chat_history:
        m_embed = embed_model.encode(message).reshape(1, -1)
        sim = cosine_similarity(q_embed, m_embed)[0][0]
        sim_scores.append(sim)
    
    # Apply softmax to get relevance scores
    rel_scores = softmax(sim_scores, alpha)
    
    # Select top-R by similarity
    R_actual = min(R, len(chat_history))
    shortlist_idx = np.argsort(rel_scores)[-R_actual:][::-1]
    
    # Step 2: Compute semantic entropy for shortlist
    SE_scores = []
    for idx in shortlist_idx:
        message = chat_history[idx]
        context = query + "\n" + message
        
        # Generate T candidate outputs
        outputs = []
        for _ in range(T):
            output = llm_sample(context)
            outputs.append(output)
        
        # Compute semantic entropy
        se = compute_semantic_entropy(outputs, embed_model, similarity_threshold)
        SE_scores.append(se)
    
    # Normalize semantic entropy
    if len(SE_scores) > 0 and max(SE_scores) > 0:
        max_SE = max(SE_scores)
        normalized_SE = [se / max_SE for se in SE_scores]
    else:
        normalized_SE = [0.0] * len(SE_scores)
    
    # Step 3: Combined score for shortlist
    combined = []
    for j, idx in enumerate(shortlist_idx):
        score = lambda_param * rel_scores[idx] + (1 - lambda_param) * normalized_SE[j]
        combined.append((idx, score))
    
    # Step 4: Select top-K overall
    K_actual = min(K, len(combined))
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
    final_idx = [idx for idx, _ in combined_sorted[:K_actual]]
    
    selected_messages = [chat_history[i] for i in final_idx]
    
    return selected_messages

# ============================================================
# Entropy-Based Memory Hit Ratio Gating (ORIGINAL)
# ============================================================
def apply_entropy_memory_hit_gating(
    query: str,
    candidates: List[str],
    llm_predict: Callable[[str], List[float]] = mock_llm_predict,
    entropy_threshold: float = 0.05,
    token_budget: int = 256
) -> Tuple[List[str], float]:
    """
    Select memories that reduce model entropy.
    Returns:
    - gated memory list
    - memory hit ratio
    """
    base_entropy = estimate_entropy(query, llm_predict)
    selected = []
    useful = 0
    total_tokens = 0
    
    for text in candidates:
        token_cost = len(text.split())
        if total_tokens + token_cost > token_budget:
            continue
        
        combined_prompt = text + "\n" + query
        new_entropy = estimate_entropy(combined_prompt, llm_predict)
        delta_entropy = base_entropy - new_entropy
        
        if delta_entropy >= entropy_threshold:
            selected.append(text)
            total_tokens += token_cost
            useful += 1
    
    memory_hit_ratio = useful / max(len(candidates), 1)
    return selected, memory_hit_ratio

# ============================================================
# Main Gate Function (ORIGINAL INTERFACE)
# ============================================================
def gate(
    query,
    memory_weight=0.6,
    doc_weight=0.4,
    top_k=5,
    llm_predict: Callable[[str], List[float]] = mock_llm_predict
):
    """
    Retrieval + entropy-based memory hit ratio gating.
    """
    mem_results = retrieve_similar(query, k=top_k)
    doc_results = retrieve(query, k=top_k)
    
    # Merge memory + document retrieval
    merged = []
    for m in mem_results:
        merged.append({
            "source": "memory",
            "text": m["response"],
            "score": memory_weight
        })
    for d in doc_results:
        merged.append({
            "source": "document",
            "text": d,
            "score": doc_weight
        })
    
    # Initial heuristic ordering
    merged.sort(key=lambda x: x["score"], reverse=True)
    texts = [m["text"] for m in merged[:top_k]]
    
    # Entropy-based gating
    gated_texts, memory_hit_ratio = apply_entropy_memory_hit_gating(
        query=query,
        candidates=texts,
        llm_predict=llm_predict
    )
    
    return gated_texts, memory_hit_ratio