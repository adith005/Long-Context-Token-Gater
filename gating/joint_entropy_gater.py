"""
Combinatorial Memory Selection via Joint Entropy Minimization

Algorithm:
1. Filter memories with similarity >= 50%
2. Calculate entropy of each combination considering similarity + memory space
3. Find combination with minimum total entropy (lowest uncertainty)

Goal: High similarity + Low memory usage = Low entropy
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations
from collections import Counter
import math


class JointEntropyMemorySelector:
    """
    Selects optimal memory subset by minimizing joint entropy of
    similarity and memory usage.
    """
    
    def __init__(self, memories: List[Dict], similarity_threshold: float = 0.5):
        """
        Initialize selector with memories.
        
        Args:
            memories: List of memory dicts with keys:
                - 'text': memory content (str)
                - 'similarity': similarity to query (float 0-1)
                - 'embedding': optional embedding vector
            similarity_threshold: Minimum similarity to consider (default 0.5)
        """
        # Step 1: Filter memories with similarity >= threshold
        self.filtered_memories = [
            m for m in memories 
            if m.get('similarity', 0) >= similarity_threshold
        ]
        self.similarity_threshold = similarity_threshold
        
        # Extract features for each memory
        self.features = []
        for mem in self.filtered_memories:
            self.features.append({
                'similarity': mem['similarity'],
                'memory_size': len(mem['text'].split()),  # token count
                'text': mem['text'],
                'index': mem.get('index', 0)
            })
    
    def calculate_shannon_entropy(self, values: List[float]) -> float:
        """Calculate Shannon entropy of a distribution."""
        if not values or len(values) == 0:
            return 0.0
        
        # Create probability distribution
        total = sum(values)
        if total == 0:
            return 0.0
        
        probs = [v / total for v in values]
        
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def calculate_joint_entropy_2d(
        self, 
        similarities: List[float], 
        memory_sizes: List[int],
        n_bins: int = 5
    ) -> float:
        """
        Calculate joint entropy H(similarity, memory).
        
        Uses 2D histogram to discretize continuous space.
        
        Args:
            similarities: List of similarity scores
            memory_sizes: List of memory sizes
            n_bins: Number of bins for discretization
            
        Returns:
            Joint entropy value
        """
        if not similarities or not memory_sizes:
            return 0.0
        
        # Create 2D histogram
        sim_bins = np.linspace(
            self.similarity_threshold, 1.0, n_bins + 1
        )
        mem_bins = np.linspace(
            min(memory_sizes), max(memory_sizes) + 1, n_bins + 1
        )
        
        # Count points in each bin
        hist_2d = np.zeros((n_bins, n_bins))
        
        for sim, mem in zip(similarities, memory_sizes):
            # Find bin indices
            sim_idx = np.searchsorted(sim_bins[:-1], sim, side='right') - 1
            mem_idx = np.searchsorted(mem_bins[:-1], mem, side='right') - 1
            
            sim_idx = min(max(0, sim_idx), n_bins - 1)
            mem_idx = min(max(0, mem_idx), n_bins - 1)
            
            hist_2d[sim_idx, mem_idx] += 1
        
        # Calculate joint probability distribution
        total_count = np.sum(hist_2d)
        if total_count == 0:
            return 0.0
        
        joint_prob = hist_2d / total_count
        
        # Calculate joint entropy
        entropy = 0.0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint_prob[i, j] > 0:
                    entropy -= joint_prob[i, j] * math.log2(joint_prob[i, j])
        
        return entropy
    
    def calculate_weighted_entropy(
        self,
        similarities: List[float],
        memory_sizes: List[int],
        sim_weight: float = 0.5,
        mem_weight: float = 0.5
    ) -> float:
        """
        Calculate weighted sum of individual entropies.
        
        H_total = w_sim * H(similarity) + w_mem * H(memory)
        """
        # Normalize similarities to create distribution
        if sum(similarities) > 0:
            sim_dist = [s / sum(similarities) for s in similarities]
            h_sim = -sum(p * math.log2(p) if p > 0 else 0 for p in sim_dist)
        else:
            h_sim = 0.0
        
        # Normalize memory sizes to create distribution
        if sum(memory_sizes) > 0:
            mem_dist = [m / sum(memory_sizes) for m in memory_sizes]
            h_mem = -sum(p * math.log2(p) if p > 0 else 0 for p in mem_dist)
        else:
            h_mem = 0.0
        
        return sim_weight * h_sim + mem_weight * h_mem
    
    def calculate_efficiency_score(
        self,
        similarities: List[float],
        memory_sizes: List[int]
    ) -> float:
        """
        Calculate efficiency: maximize similarity, minimize memory variance.
        
        Score combines:
        - High total similarity (good)
        - Low memory entropy (good - consistent sizes)
        - Low total memory (good - efficient)
        """
        if not similarities or not memory_sizes:
            return float('inf')
        
        total_sim = sum(similarities)
        avg_memory = np.mean(memory_sizes)
        
        # Memory distribution entropy (lower is better - more uniform)
        if sum(memory_sizes) > 0:
            mem_dist = [m / sum(memory_sizes) for m in memory_sizes]
            h_mem = -sum(p * math.log2(p) if p > 0 else 0 for p in mem_dist)
        else:
            h_mem = 0.0
        
        # Combined score (lower is better)
        # Penalize: low similarity, high memory, high memory variance
        score = (1 / (total_sim + 1e-6)) + (avg_memory / 100) + h_mem
        
        return score

    def select_optimal_greedy(
        self,
        target_size: int,
        method: str = 'joint_entropy'
    ) -> Dict:
        """
        Greedy approximation - much faster for large sets.
        
        Iteratively adds memory that minimizes marginal entropy.
        """
        if not self.features:
            return {
                'selected_indices': [],
                'selected_memories': [],
                'total_entropy': 0.0,
                'metrics': {}
            }
        
        selected_indices = []
        remaining_indices = set(range(len(self.features)))
        
        for _ in range(min(target_size, len(self.features))):
            best_next_idx = None
            best_next_score = float('inf')
            
            # Try adding each remaining memory
            for idx in remaining_indices:
                candidate_indices = selected_indices + [idx]
                
                # Calculate score
                combo_sims = [self.features[i]['similarity'] for i in candidate_indices]
                combo_mems = [self.features[i]['memory_size'] for i in candidate_indices]
                
                if method == 'joint_entropy':
                    score = self.calculate_joint_entropy_2d(combo_sims, combo_mems)
                elif method == 'weighted':
                    score = self.calculate_weighted_entropy(combo_sims, combo_mems)
                elif method == 'efficiency':
                    score = self.calculate_efficiency_score(combo_sims, combo_mems)
                
                if score < best_next_score:
                    best_next_score = score
                    best_next_idx = idx
            
            # Add best next memory
            if best_next_idx is not None:
                selected_indices.append(best_next_idx)
                remaining_indices.remove(best_next_idx)
        
        # Compile results
        combo_sims = [self.features[i]['similarity'] for i in selected_indices]
        combo_mems = [self.features[i]['memory_size'] for i in selected_indices]
        
        selected_memories = [self.filtered_memories[i] for i in selected_indices]
        
        return {
            'selected_indices': selected_indices,
            'selected_memories': selected_memories,
            'total_entropy': best_next_score,
            'metrics': {
                'similarities': combo_sims,
                'memory_sizes': combo_mems,
                'total_similarity': sum(combo_sims),
                'total_memory': sum(combo_mems),
                'avg_similarity': np.mean(combo_sims),
                'avg_memory': np.mean(combo_mems),
                'entropy': best_next_score
            },
            'method': f'{method}_greedy'
        }


def visualize_selection(result: Dict):
    """Visualize the selection results."""
    print("=" * 80)
    print("OPTIMAL MEMORY SELECTION RESULTS")
    print("=" * 80)
    
    print(f"\nMethod: {result['method']}")
    print(f"Total Entropy: {result['total_entropy']:.4f}")
    print(f"Combinations Evaluated: {result.get('total_evaluated', 'N/A')}")
    
    metrics = result['metrics']
    print("\n" + "-" * 80)
    print("METRICS")
    print("-" * 80)
    print(f"Number of memories selected: {len(result['selected_indices'])}")
    print(f"Total similarity:  {metrics['total_similarity']:.4f}")
    print(f"Average similarity: {metrics['avg_similarity']:.4f}")
    print(f"Total memory:      {metrics['total_memory']} tokens")
    print(f"Average memory:    {metrics['avg_memory']:.1f} tokens")
    
    print("\n" + "-" * 80)
    print("SELECTED MEMORIES")
    print("-" * 80)
    
    for i, (mem, sim, size) in enumerate(zip(
        result['selected_memories'],
        metrics['similarities'],
        metrics['memory_sizes']
    ), 1):
        text_preview = mem['text'][:60] + "..." if len(mem['text']) > 60 else mem['text']
        print(f"\n{i}. Similarity: {sim:.3f} | Size: {size} tokens")
        print(f"   {text_preview}")


# Example usage
if __name__ == "__main__":
    # Create sample memories with varying similarity and sizes
    np.random.seed(42)
    
    memories = []
    for i in range(15):
        similarity = np.random.uniform(0.3, 1.0)
        size = np.random.randint(20, 150)
        text = " ".join([f"word_{j}" for j in range(size)])
        
        memories.append({
            'text': text,
            'similarity': similarity,
            'index': i
        })
    
    print("=" * 80)
    print("MEMORY POOL")
    print("=" * 80)
    print(f"Total memories: {len(memories)}")
    print(f"Similarity range: [{min(m['similarity'] for m in memories):.2f}, "
          f"{max(m['similarity'] for m in memories):.2f}]")
    print(f"Memory size range: [{min(len(m['text'].split()) for m in memories)}, "
          f"{max(len(m['text'].split()) for m in memories)}] tokens")
    
    # Filter by threshold
    filtered = [m for m in memories if m['similarity'] >= 0.5]
    print(f"\nAfter filtering (>=0.5 similarity): {len(filtered)} memories")
    
    # Create selector
    selector = JointEntropyMemorySelector(memories, similarity_threshold=0.5)
    
    print("\n" + "=" * 80)
    print("METHOD 1: JOINT ENTROPY (Exact)")
    print("=" * 80)
    
    # Try exact optimization for small sets
    result1 = selector.select_optimal_combination(
        max_size=5,
        method='joint_entropy',
        n_bins=5
    )
    visualize_selection(result1)
    
    print("\n" + "=" * 80)
    print("METHOD 2: WEIGHTED ENTROPY")
    print("=" * 80)
    
    result2 = selector.select_optimal_combination(
        max_size=5,
        method='weighted',
        sim_weight=0.7,
        mem_weight=0.3
    )
    visualize_selection(result2)
    
    print("\n" + "=" * 80)
    print("METHOD 3: EFFICIENCY SCORE")
    print("=" * 80)
    
    result3 = selector.select_optimal_combination(
        max_size=5,
        method='efficiency'
    )
    visualize_selection(result3)
    
    print("\n" + "=" * 80)
    print("METHOD 4: GREEDY APPROXIMATION (Fast)")
    print("=" * 80)
    
    result4 = selector.select_optimal_greedy(
        target_size=5,
        method='joint_entropy'
    )
    visualize_selection(result4)
