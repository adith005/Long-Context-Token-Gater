"""
COMPLETE ENTROPY CATALOG FOR MEMORY SELECTION
==============================================

This document provides ALL entropy formulations you can use to describe
different variables in your memory selection problem.

Each entry includes:
- Mathematical definition
- What it measures
- When to use it
- Code implementation
- Example application
"""

import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import mutual_info_score
from collections import Counter
import math


# =============================================================================
# SECTION 1: BASIC INFORMATION-THEORETIC ENTROPIES
# =============================================================================

def shannon_entropy(values):
    """
    Shannon Entropy: H(X) = -Σ p(x) * log₂(p(x))
    
    Measures: Uncertainty, unpredictability, information content
    
    Variables it describes:
    - Token distribution diversity
    - Similarity score spread
    - Memory size variation
    - Any probability distribution
    
    Low entropy: Concentrated, predictable
    High entropy: Diverse, unpredictable
    
    Example:
    - values = [0.9, 0.05, 0.05] → H = 0.57 (concentrated)
    - values = [0.33, 0.33, 0.34] → H = 1.58 (uniform)
    """
    if not values or sum(values) == 0:
        return 0.0
    
    # Normalize to probabilities
    total = sum(values)
    probs = [v / total for v in values if v > 0]
    
    return -sum(p * math.log2(p) for p in probs)


def joint_entropy(x_values, y_values, n_bins=10):
    """
    Joint Entropy: H(X,Y) = -Σ Σ p(x,y) * log₂(p(x,y))
    
    Measures: Combined uncertainty of two variables
    
    Variables it describes:
    - (Similarity, Memory) joint distribution
    - (Relevance, Recency) combinations
    - Any two correlated features
    
    Use when: Need to consider two variables together
    
    Example: High similarity + Low memory = Low joint entropy (good!)
    """
    if len(x_values) != len(y_values):
        raise ValueError("x and y must have same length")
    
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        x_values, y_values, bins=n_bins
    )
    
    # Convert to probability distribution
    prob = hist / np.sum(hist)
    
    # Calculate joint entropy
    h = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if prob[i, j] > 0:
                h -= prob[i, j] * math.log2(prob[i, j])
    
    return h


def conditional_entropy(x_values, y_values, n_bins=10):
    """
    Conditional Entropy: H(X|Y) = H(X,Y) - H(Y)
    
    Measures: Uncertainty in X given we know Y
    
    Variables it describes:
    - Redundancy between memories
    - Information gain from adding memory
    - Predictability given context
    
    Use when: Measuring incremental information
    
    Example: H(new_memory | selected_memories)
    """
    h_xy = joint_entropy(x_values, y_values, n_bins)
    h_y = shannon_entropy(y_values)
    
    return h_xy - h_y


def mutual_information(x_values, y_values, n_bins=10):
    """
    Mutual Information: I(X;Y) = H(X) + H(Y) - H(X,Y)
    
    Measures: Shared information between variables
    
    Variables it describes:
    - How much similarity tells us about relevance
    - Dependency between memory size and quality
    - Feature correlation strength
    
    High MI: Variables are dependent
    Low MI: Variables are independent
    
    Use when: Understanding feature relationships
    """
    h_x = shannon_entropy(x_values)
    h_y = shannon_entropy(y_values)
    h_xy = joint_entropy(x_values, y_values, n_bins)
    
    return h_x + h_y - h_xy


def cross_entropy(p_values, q_values):
    """
    Cross Entropy: H(P,Q) = -Σ p(x) * log₂(q(x))
    
    Measures: Distance from distribution P to Q
    
    Variables it describes:
    - Actual vs expected similarity distribution
    - Real vs ideal memory allocation
    - Model vs target distribution
    
    Use when: Comparing to a reference distribution
    
    Example: Compare selected memories to ideal distribution
    """
    # Normalize to probabilities
    p_total = sum(p_values)
    q_total = sum(q_values)
    
    if p_total == 0 or q_total == 0:
        return 0.0
    
    p = [v / p_total for v in p_values]
    q = [v / q_total for v in q_values]
    
    h = 0.0
    for p_i, q_i in zip(p, q):
        if p_i > 0 and q_i > 0:
            h -= p_i * math.log2(q_i)
    
    return h


def kl_divergence(p_values, q_values):
    """
    KL Divergence: D_KL(P||Q) = Σ p(x) * log₂(p(x)/q(x))
    
    Measures: How different P is from Q (asymmetric)
    
    Variables it describes:
    - Distance from current to optimal selection
    - Similarity distribution vs uniform
    - Actual vs desired memory allocation
    
    Use when: Optimizing toward target distribution
    
    Example: Minimize KL(selected_distribution || ideal_distribution)
    """
    # Normalize
    p_total = sum(p_values)
    q_total = sum(q_values)
    
    if p_total == 0 or q_total == 0:
        return float('inf')
    
    p = [v / p_total for v in p_values]
    q = [v / q_total for v in q_values]
    
    kl = 0.0
    for p_i, q_i in zip(p, q):
        if p_i > 0:
            if q_i > 0:
                kl += p_i * math.log2(p_i / q_i)
            else:
                return float('inf')
    
    return kl


def renyi_entropy(values, alpha=2):
    """
    Rényi Entropy: H_α(X) = (1/(1-α)) * log₂(Σ p(x)^α)
    
    Generalization of Shannon entropy.
    α=1 gives Shannon entropy
    α=2 gives "collision entropy"
    α→∞ gives min-entropy
    
    Variables it describes:
    - Different aspects of distribution
    - Sensitivity to rare/common events
    - Tail behavior of distribution
    
    Use when: Need different sensitivity to probabilities
    """
    if alpha == 1:
        return shannon_entropy(values)
    
    total = sum(values)
    if total == 0:
        return 0.0
    
    probs = [v / total for v in values if v > 0]
    
    sum_p_alpha = sum(p ** alpha for p in probs)
    
    return (1 / (1 - alpha)) * math.log2(sum_p_alpha)


# =============================================================================
# SECTION 2: SIMILARITY-SPECIFIC ENTROPIES
# =============================================================================

def similarity_distribution_entropy(similarities, n_bins=10):
    """
    Entropy of similarity score distribution.
    
    Measures: Diversity in how similar memories are
    
    Low: All memories have similar similarity scores
    High: Wide range of similarity scores
    
    Use for: Understanding selection concentration
    """
    if not similarities:
        return 0.0
    
    # Create histogram
    hist, _ = np.histogram(similarities, bins=n_bins, range=(0, 1))
    
    # Convert to probabilities
    hist = hist / np.sum(hist)
    
    # Calculate entropy
    return -sum(p * math.log2(p) for p in hist if p > 0)


def similarity_concentration(similarities):
    """
    How concentrated similarities are (normalized entropy).
    
    Measures: Whether selections are tightly clustered
    
    Range: [0, 1]
    0 = All same similarity (concentrated)
    1 = Uniformly distributed (spread out)
    
    Use for: Selection confidence metric
    """
    if not similarities or len(similarities) <= 1:
        return 0.0
    
    # Normalize similarities to create distribution
    total = sum(similarities)
    if total == 0:
        return 0.0
    
    probs = [s / total for s in similarities]
    
    # Calculate entropy
    h = -sum(p * math.log2(p) for p in probs if p > 0)
    
    # Normalize by max possible entropy
    max_h = math.log2(len(similarities))
    
    return h / max_h if max_h > 0 else 0.0


def weighted_similarity_entropy(similarities, weights):
    """
    Entropy weighted by importance/confidence.
    
    Measures: Uncertainty accounting for weights
    
    Use when: Some similarities are more reliable than others
    
    Example: Recent memories weighted higher
    """
    if len(similarities) != len(weights):
        raise ValueError("Lengths must match")
    
    # Weight the similarities
    weighted = [s * w for s, w in zip(similarities, weights)]
    
    return shannon_entropy(weighted)


def similarity_range_entropy(similarities):
    """
    Entropy based on similarity ranges.
    
    Bins: [0.5-0.6), [0.6-0.7), ..., [0.9-1.0]
    
    Measures: Distribution across quality tiers
    
    Use for: Understanding quality spread
    """
    if not similarities:
        return 0.0
    
    # Define bins
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Count in each bin
    counts = [0] * (len(bins) - 1)
    for sim in similarities:
        for i in range(len(bins) - 1):
            if bins[i] <= sim < bins[i + 1]:
                counts[i] += 1
                break
        else:  # sim >= 1.0
            counts[-1] += 1
    
    return shannon_entropy(counts)


# =============================================================================
# SECTION 3: MEMORY-SPECIFIC ENTROPIES
# =============================================================================

def memory_allocation_entropy(memory_sizes):
    """
    How evenly memory is distributed across selections.
    
    H_mem = -Σ (m_i / M_total) * log₂(m_i / M_total)
    
    Measures: Balance of memory usage
    
    Low: One memory dominates
    High: Equal memory distribution
    
    Use for: Fair memory allocation
    """
    return shannon_entropy(memory_sizes)


def memory_efficiency_entropy(contents, memory_sizes):
    """
    Information entropy per unit memory.
    
    H_eff = H(content) / total_memory
    
    Measures: Information density
    
    Use for: Selecting compact, information-rich memories
    """
    # Calculate content entropy (e.g., token diversity)
    all_tokens = []
    for content in contents:
        all_tokens.extend(content.split())
    
    token_counts = Counter(all_tokens)
    h_content = shannon_entropy(list(token_counts.values()))
    
    total_memory = sum(memory_sizes)
    
    return h_content / total_memory if total_memory > 0 else 0.0


def memory_variance_entropy(memory_sizes):
    """
    Entropy from memory size variance.
    
    Low variance → Low entropy → Consistent sizes
    High variance → High entropy → Mixed sizes
    
    Use for: Preferring uniform-sized selections
    """
    if not memory_sizes:
        return 0.0
    
    # Normalize to distribution
    total = sum(memory_sizes)
    if total == 0:
        return 0.0
    
    probs = [m / total for m in memory_sizes]
    
    return -sum(p * math.log2(p) for p in probs if p > 0)


def marginal_memory_entropy(current_selection, new_memory):
    """
    Entropy change from adding new memory.
    
    ΔH = H(current ∪ new) - H(current)
    
    Measures: Incremental uncertainty added
    
    Use for: Greedy selection (add memory with lowest ΔH)
    """
    current_entropy = memory_allocation_entropy(current_selection)
    new_entropy = memory_allocation_entropy(current_selection + [new_memory])
    
    return new_entropy - current_entropy


# =============================================================================
# SECTION 4: CONTENT-BASED ENTROPIES
# =============================================================================

def token_distribution_entropy(texts):
    """
    Entropy of token distribution across all selected texts.
    
    Measures: Vocabulary diversity
    
    Low: Repetitive language
    High: Rich vocabulary
    
    Use for: Ensuring diverse content
    """
    all_tokens = []
    for text in texts:
        all_tokens.extend(text.split())
    
    token_counts = Counter(all_tokens)
    
    return shannon_entropy(list(token_counts.values()))


def ngram_entropy(texts, n=2):
    """
    Entropy of n-gram distribution.
    
    Measures: Phrasal diversity
    
    Use for: Detecting repetitive phrases
    
    Example: n=2 for bigrams, n=3 for trigrams
    """
    ngrams = []
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
    
    ngram_counts = Counter(ngrams)
    
    return shannon_entropy(list(ngram_counts.values()))


def semantic_cluster_entropy(embeddings, n_clusters=5):
    """
    Entropy of semantic clusters.
    
    Requires: Memory embeddings
    
    Measures: Topical diversity
    
    Use for: Ensuring coverage of different topics
    """
    from sklearn.cluster import KMeans
    
    if len(embeddings) < n_clusters:
        n_clusters = len(embeddings)
    
    # Cluster embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    
    # Count cluster membership
    cluster_counts = Counter(labels)
    
    return shannon_entropy(list(cluster_counts.values()))


def type_token_ratio_entropy(texts, window_size=10):
    """
    Entropy based on type-token ratio.
    
    TTR = unique_tokens / total_tokens
    
    Measures: Lexical diversity in windows
    
    Use for: Detecting redundant content
    """
    ttrs = []
    
    for text in texts:
        tokens = text.split()
        for i in range(0, len(tokens), window_size):
            window = tokens[i:i+window_size]
            if window:
                ttr = len(set(window)) / len(window)
                ttrs.append(ttr)
    
    return shannon_entropy(ttrs)


# =============================================================================
# SECTION 5: REDUNDANCY ENTROPIES
# =============================================================================

def overlap_entropy(texts):
    """
    Entropy from pairwise text overlaps.
    
    Measures: How much memories share content
    
    Low: High redundancy (much overlap)
    High: Low redundancy (unique content)
    
    Use for: Avoiding redundant selections
    """
    if len(texts) < 2:
        return 0.0
    
    overlaps = []
    
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            tokens_i = set(texts[i].split())
            tokens_j = set(texts[j].split())
            
            intersection = len(tokens_i & tokens_j)
            union = len(tokens_i | tokens_j)
            
            overlap = intersection / union if union > 0 else 0
            overlaps.append(overlap)
    
    # Bin overlaps
    n_bins = 10
    hist, _ = np.histogram(overlaps, bins=n_bins, range=(0, 1))
    
    return shannon_entropy(hist)


def incremental_information_entropy(selected_texts, candidate_text):
    """
    New information provided by candidate.
    
    H(candidate | selected) = H(selected ∪ candidate) - H(selected)
    
    Measures: Marginal information gain
    
    Use for: Greedy selection minimizing redundancy
    """
    h_selected = token_distribution_entropy(selected_texts)
    h_with_candidate = token_distribution_entropy(selected_texts + [candidate_text])
    
    return h_with_candidate - h_selected


def jaccard_diversity_entropy(texts):
    """
    Entropy of Jaccard similarities between text pairs.
    
    Measures: Diversity of pairwise similarities
    
    Use for: Understanding selection cohesion
    """
    if len(texts) < 2:
        return 0.0
    
    jaccard_sims = []
    
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            tokens_i = set(texts[i].split())
            tokens_j = set(texts[j].split())
            
            jaccard = len(tokens_i & tokens_j) / len(tokens_i | tokens_j) if tokens_i | tokens_j else 0
            jaccard_sims.append(jaccard)
    
    # Create distribution
    n_bins = 10
    hist, _ = np.histogram(jaccard_sims, bins=n_bins, range=(0, 1))
    
    return shannon_entropy(hist)


# =============================================================================
# SECTION 6: TEMPORAL ENTROPIES
# =============================================================================

def recency_weighted_entropy(values, ages, decay_rate=0.1):
    """
    Entropy with exponential decay by age.
    
    H_recency = -Σ w(age) * p(x) * log₂(p(x))
    where w(age) = exp(-λ * age)
    
    Measures: Uncertainty emphasizing recent items
    
    Use for: Time-sensitive selection
    """
    if len(values) != len(ages):
        raise ValueError("Lengths must match")
    
    # Calculate weights
    weights = [math.exp(-decay_rate * age) for age in ages]
    
    # Weighted probabilities
    total = sum(values)
    if total == 0:
        return 0.0
    
    weighted_probs = [(v / total) * w for v, w in zip(values, weights)]
    weighted_total = sum(weighted_probs)
    
    if weighted_total == 0:
        return 0.0
    
    # Normalize
    probs = [wp / weighted_total for wp in weighted_probs]
    
    return -sum(p * math.log2(p) for p in probs if p > 0)


def temporal_diversity_entropy(timestamps):
    """
    Entropy of time distribution.
    
    Measures: Temporal spread of selections
    
    Use for: Ensuring coverage across time periods
    """
    if not timestamps:
        return 0.0
    
    # Create time bins
    min_time = min(timestamps)
    max_time = max(timestamps)
    
    if min_time == max_time:
        return 0.0
    
    n_bins = 10
    hist, _ = np.histogram(timestamps, bins=n_bins, range=(min_time, max_time))
    
    return shannon_entropy(hist)


# =============================================================================
# SECTION 7: OPTIMIZATION ENTROPIES
# =============================================================================

def pareto_entropy(pareto_front):
    """
    Entropy of Pareto-optimal solutions.
    
    pareto_front: List of (objective1, objective2) tuples
    
    Measures: Diversity of non-dominated solutions
    
    Use for: Multi-objective uncertainty
    """
    if not pareto_front:
        return 0.0
    
    # Create 2D histogram of Pareto points
    obj1 = [p[0] for p in pareto_front]
    obj2 = [p[1] for p in pareto_front]
    
    return joint_entropy(obj1, obj2)


def regret_entropy(selected_values, all_values):
    """
    Entropy of regret (opportunity cost).
    
    regret_i = max(all_values) - selected_value_i
    
    Measures: Uncertainty in what we missed
    
    Use for: Understanding selection risk
    """
    max_value = max(all_values)
    regrets = [max_value - v for v in selected_values]
    
    return shannon_entropy(regrets)


def portfolio_entropy(combination_scores):
    """
    Entropy across different possible combinations.
    
    Measures: Uncertainty in which combination to choose
    
    Use for: Solution space exploration
    """
    return shannon_entropy(combination_scores)


# =============================================================================
# SECTION 8: COMPOSITE ENTROPIES
# =============================================================================

def multi_objective_joint_entropy(objectives_dict, n_bins=5):
    """
    Joint entropy of multiple objectives.
    
    objectives_dict = {
        'similarity': [0.9, 0.8, ...],
        'memory': [100, 150, ...],
        'recency': [0.9, 0.5, ...]
    }
    
    Measures: Combined uncertainty of all objectives
    
    Use for: Holistic optimization
    """
    # Normalize each objective to [0, 1]
    normalized = {}
    for name, values in objectives_dict.items():
        min_val = min(values)
        max_val = max(values)
        if max_val > min_val:
            normalized[name] = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized[name] = [0.5] * len(values)
    
    # Create multi-dimensional histogram
    # For simplicity, sum individual entropies (approximation)
    total_entropy = sum(
        shannon_entropy(vals) for vals in normalized.values()
    )
    
    return total_entropy


def weighted_multi_entropy(entropies_dict):
    """
    Weighted combination of different entropy measures.
    
    entropies_dict = {
        'similarity': (h_sim, weight_sim),
        'memory': (h_mem, weight_mem),
        ...
    }
    
    H_total = Σ w_i * H_i
    
    Use for: Custom trade-offs between objectives
    """
    total = 0.0
    weight_sum = 0.0
    
    for name, (entropy_val, weight) in entropies_dict.items():
        total += entropy_val * weight
        weight_sum += weight
    
    return total / weight_sum if weight_sum > 0 else 0.0


# =============================================================================
# SECTION 9: YOUR SPECIFIC FORMULATIONS
# =============================================================================

def similarity_memory_joint_entropy(similarities, memory_sizes, n_bins=5):
    """
    THE MAIN ENTROPY FOR YOUR PROJECT.
    
    H(similarity, memory) = -Σ Σ p(s_i, m_j) * log₂(p(s_i, m_j))
    
    Finds combinations with:
    - High similarity scores (concentrated high values)
    - Low memory usage (concentrated small sizes)
    = LOW JOINT ENTROPY
    
    This is what you minimize to find optimal selection.
    """
    return joint_entropy(similarities, memory_sizes, n_bins)


def efficiency_weighted_entropy(similarities, memory_sizes, alpha=0.7):
    """
    Alternative formulation emphasizing efficiency.
    
    H_eff = α * H(sim) + (1-α) * H(mem)
    
    where we want:
    - Low H(sim) = concentrated high similarities
    - High H(mem) = diverse memory sizes acceptable
    
    alpha controls trade-off
    """
    h_sim = shannon_entropy(similarities)
    h_mem = shannon_entropy(memory_sizes)
    
    return alpha * h_sim + (1 - alpha) * h_mem


def normalized_objective_entropy(similarities, memory_sizes):
    """
    Normalize both to [0,1] then calculate joint entropy.
    
    Ensures fair comparison between different scales.
    """
    # Normalize similarities
    sim_min, sim_max = min(similarities), max(similarities)
    if sim_max > sim_min:
        norm_sim = [(s - sim_min) / (sim_max - sim_min) for s in similarities]
    else:
        norm_sim = [0.5] * len(similarities)
    
    # Normalize memory (invert so lower memory = higher value)
    mem_min, mem_max = min(memory_sizes), max(memory_sizes)
    if mem_max > mem_min:
        norm_mem = [1 - (m - mem_min) / (mem_max - mem_min) for m in memory_sizes]
    else:
        norm_mem = [0.5] * len(memory_sizes)
    
    return joint_entropy(norm_sim, norm_mem)


# =============================================================================
# USAGE GUIDE
# =============================================================================

"""
QUICK REFERENCE: WHICH ENTROPY FOR WHICH VARIABLE?

SIMILARITY:
- similarity_distribution_entropy() - Diversity of similarities
- similarity_concentration() - How clustered similarities are
- similarity_range_entropy() - Distribution across quality tiers

MEMORY USAGE:
- memory_allocation_entropy() - Balance of memory distribution
- memory_variance_entropy() - Consistency of memory sizes
- memory_efficiency_entropy() - Information per memory unit

COMBINED (SIMILARITY + MEMORY):
- similarity_memory_joint_entropy() - PRIMARY FOR YOUR PROJECT
- efficiency_weighted_entropy() - Weighted combination
- normalized_objective_entropy() - Fair scaling

CONTENT QUALITY:
- token_distribution_entropy() - Vocabulary diversity
- ngram_entropy() - Phrasal patterns
- semantic_cluster_entropy() - Topic coverage

REDUNDANCY:
- overlap_entropy() - Content overlap
- incremental_information_entropy() - New information added
- jaccard_diversity_entropy() - Pairwise similarity spread

TEMPORAL:
- recency_weighted_entropy() - Time-sensitive uncertainty
- temporal_diversity_entropy() - Coverage across time

OPTIMIZATION:
- pareto_entropy() - Multi-objective solution diversity
- regret_entropy() - Opportunity cost
- portfolio_entropy() - Solution space uncertainty


RECOMMENDED FOR YOUR PROJECT:

Main objective:
    minimize similarity_memory_joint_entropy(similarities, memory_sizes)

With constraints:
    all(s >= 0.5 for s in similarities)  # Filter first
    
This finds the combination with lowest joint uncertainty,
which corresponds to:
- Concentrated high similarities
- Concentrated low memory usage
- Most "certain" selection given both objectives
"""
