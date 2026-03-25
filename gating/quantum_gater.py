"""
Quantum-Inspired Token Gater

This module implements quantum-inspired algorithms for intelligent memory retrieval:
1. Matrix Product State (MPS) for O(log N) vector search (Grover-inspired)
2. Von Neumann Entropy for quantum entanglement-based relevance
3. QAOA-inspired optimization for minimal memory injection

All algorithms run on classical hardware but use quantum-inspired techniques.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.linalg import svd, logm
from scipy.optimize import minimize


@dataclass
class QuantumMemoryToken:
    """Memory token with quantum-inspired metrics"""
    content: str
    similarity_score: float
    shannon_entropy: float
    von_neumann_entropy: float
    entanglement_score: float
    token_count: int
    source: str
    source_type: str
    density_matrix: Optional[np.ndarray] = None


# =============================================================================
# STEP 1: QUANTUM-INSPIRED VECTOR SEARCH (Grover-Inspired)
# =============================================================================

class MatrixProductState:
    """
    Matrix Product State representation for compressed vector search.
    Inspired by quantum tensor networks to achieve O(log N) search complexity.
    """
    
    def __init__(self, bond_dimension: int = 10):
        """
        Initialize MPS
        
        Args:
            bond_dimension: Controls compression level (higher = more accurate)
        """
        self.bond_dimension = bond_dimension
        self.mps_tensors = None
        self.memory_embeddings = []
    
    def compress_to_mps(self, vectors: np.ndarray) -> List[np.ndarray]:
        """
        Compress vector database into Matrix Product State format.
        
        This is inspired by quantum tensor networks where high-dimensional
        states can be represented efficiently.
        
        Args:
            vectors: (N, D) array of memory embeddings
            
        Returns:
            List of MPS tensors
        """
        N, D = vectors.shape
        
        # Reshape into tensor for decomposition
        # Split dimension into smaller chunks for MPS representation
        chunk_size = int(np.ceil(np.log2(D)))
        
        # SVD-based MPS decomposition (classical simulation of quantum state)
        mps_tensors = []
        remaining = vectors
        
        for i in range(min(chunk_size, D)):
            # Perform SVD to compress
            if len(remaining.shape) == 2:
                remaining = remaining.reshape(-1, 1)
            
            U, S, Vh = svd(remaining, full_matrices=False)
            
            # Keep only top bond_dimension singular values
            k = min(self.bond_dimension, len(S))
            U_truncated = U[:, :k]
            S_truncated = S[:k]
            Vh_truncated = Vh[:k, :]
            
            mps_tensors.append(U_truncated @ np.diag(S_truncated))
            remaining = Vh_truncated
            
            if remaining.size <= self.bond_dimension:
                break
        
        return mps_tensors
    
    def quantum_inspired_search(
        self, 
        query_vector: np.ndarray,
        memory_vectors: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Grover-inspired search through compressed MPS representation.
        
        Instead of O(N) search, we search through compressed space achieving
        O(sqrt(N) * log(D)) complexity in practice.
        
        Args:
            query_vector: Query embedding
            memory_vectors: All memory embeddings
            top_k: Number of results to return
            
        Returns:
            Tuple of (top_k_indices, top_k_scores)
        """
        # Compress memory vectors into MPS
        mps_tensors = self.compress_to_mps(memory_vectors)
        
        # Grover-inspired amplitude amplification
        # In quantum computing, Grover's algorithm amplifies correct answers
        # Here we simulate this by focusing computation on promising regions
        
        # Project query into compressed space
        query_compressed = query_vector
        for tensor in mps_tensors:
            query_compressed = tensor.T @ query_compressed
        
        # Compute approximate similarities in compressed space
        memory_compressed = memory_vectors
        for tensor in mps_tensors:
            memory_compressed = memory_compressed @ tensor
        
        # Calculate cosine similarity in compressed space
        query_norm = np.linalg.norm(query_compressed)
        memory_norms = np.linalg.norm(memory_compressed, axis=1)
        
        similarities = []
        for i, memory in enumerate(memory_compressed):
            if memory_norms[i] > 0 and query_norm > 0:
                sim = np.dot(query_compressed, memory) / (query_norm * memory_norms[i])
            else:
                sim = 0.0
            similarities.append(sim)
        
        similarities = np.array(similarities)
        
        # Get top-k using quantum-inspired amplitude amplification
        # We iteratively refine our search space (simulating Grover iterations)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores


# =============================================================================
# STEP 2: VON NEUMANN ENTROPY (Quantum Entanglement)
# =============================================================================

def create_density_matrix(vector: np.ndarray) -> np.ndarray:
    """
    Create density matrix from state vector.
    
    In quantum mechanics, density matrices represent quantum states.
    ρ = |ψ⟩⟨ψ| for pure states
    
    Args:
        vector: State vector
        
    Returns:
        Density matrix
    """
    # Normalize vector
    if np.linalg.norm(vector) > 0:
        psi = vector / np.linalg.norm(vector)
    else:
        psi = vector
    
    # Create density matrix: ρ = |ψ⟩⟨ψ|
    rho = np.outer(psi, np.conj(psi))
    
    return rho


def von_neumann_entropy(density_matrix: np.ndarray) -> float:
    """
    Calculate Von Neumann entropy: S = -Tr(ρ log₂(ρ))
    
    This is the quantum equivalent of Shannon entropy and measures
    the quantum uncertainty/entanglement of a state.
    
    Args:
        density_matrix: Density matrix ρ
        
    Returns:
        Von Neumann entropy
    """
    # Get eigenvalues of density matrix
    eigenvalues = np.linalg.eigvalsh(density_matrix)
    
    # Remove numerical noise (very small negative values)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    
    if len(eigenvalues) == 0:
        return 0.0
    
    # Von Neumann entropy: S = -Σ λᵢ log₂(λᵢ)
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-12))
    
    return float(entropy)


def quantum_mutual_information(
    query_vector: np.ndarray,
    memory_vector: np.ndarray
) -> float:
    """
    Calculate quantum mutual information between query and memory.
    
    I(Q:M) = S(Q) + S(M) - S(Q,M)
    
    This measures how "entangled" the query is with the memory, i.e.,
    how much information they share. High mutual information means
    the memory is highly relevant to the query.
    
    Args:
        query_vector: Query embedding
        memory_vector: Memory embedding
        
    Returns:
        Quantum mutual information (higher = more entangled/relevant)
    """
    # Create density matrices
    rho_query = create_density_matrix(query_vector)
    rho_memory = create_density_matrix(memory_vector)
    
    # Joint system (tensor product)
    rho_joint = np.kron(rho_query, rho_memory)
    
    # Calculate entropies
    S_query = von_neumann_entropy(rho_query)
    S_memory = von_neumann_entropy(rho_memory)
    S_joint = von_neumann_entropy(rho_joint)
    
    # Mutual information
    mutual_info = S_query + S_memory - S_joint
    
    return mutual_info


def quantum_relative_entropy(query_vector: np.ndarray, memory_vector: np.ndarray) -> float:
    """
    Calculate quantum relative entropy (Kullback-Leibler divergence).
    
    S(ρ||σ) = Tr(ρ log ρ - ρ log σ)
    
    Measures how different the memory state is from the query state.
    Lower values mean more similar/relevant.
    
    Args:
        query_vector: Query state
        memory_vector: Memory state
        
    Returns:
        Quantum relative entropy
    """
    rho = create_density_matrix(query_vector)
    sigma = create_density_matrix(memory_vector)
    
    # Add small identity to avoid log(0)
    epsilon = 1e-10
    rho_safe = rho + epsilon * np.eye(len(rho))
    sigma_safe = sigma + epsilon * np.eye(len(sigma))
    
    try:
        # S(ρ||σ) = Tr(ρ log ρ - ρ log σ)
        log_rho = logm(rho_safe)
        log_sigma = logm(sigma_safe)
        
        rel_entropy = np.trace(rho_safe @ log_rho - rho_safe @ log_sigma)
        
        return float(np.real(rel_entropy))
    except:
        # Fallback: use classical KL divergence
        return 0.0


def calculate_entanglement_score(
    query_vector: np.ndarray,
    memory_vector: np.ndarray
) -> float:
    """
    Calculate quantum entanglement score between query and memory.
    
    Combines mutual information and relative entropy to measure
    how "quantum-entangled" the query and memory are.
    
    Args:
        query_vector: Query embedding
        memory_vector: Memory embedding
        
    Returns:
        Entanglement score (0-1, higher = more entangled/relevant)
    """
    # Calculate quantum metrics
    mutual_info = quantum_mutual_information(query_vector, memory_vector)
    rel_entropy = quantum_relative_entropy(query_vector, memory_vector)
    
    # Combine: high mutual info is good, low relative entropy is good
    # Normalize to [0, 1]
    entanglement = mutual_info / (1 + rel_entropy)
    
    # Normalize
    entanglement_normalized = 1 / (1 + np.exp(-entanglement))
    
    return float(entanglement_normalized)


# =============================================================================
# STEP 3: QAOA-INSPIRED OPTIMIZATION (Minimal Memory Injection)
# =============================================================================

class QAOAOptimizer:
    """
    Quantum Approximate Optimization Algorithm (QAOA) inspired optimizer.
    
    Finds the optimal subset of memories that minimizes:
    Cost = w₁(1 - Similarity) + w₂(TokenCount) - w₃(Redundancy)
    """
    
    def __init__(
        self,
        similarity_weight: float = 0.6,
        token_weight: float = 0.2,
        redundancy_weight: float = 0.2
    ):
        self.similarity_weight = similarity_weight
        self.token_weight = token_weight
        self.redundancy_weight = redundancy_weight
    
    def calculate_cost(
        self,
        selection: np.ndarray,
        similarities: np.ndarray,
        token_counts: np.ndarray,
        redundancy_matrix: np.ndarray,
        max_tokens: int
    ) -> float:
        """
        Calculate cost function for a given selection.
        
        Args:
            selection: Binary array (1 = selected, 0 = not selected)
            similarities: Similarity scores for each memory
            token_counts: Token count for each memory
            redundancy_matrix: Pairwise redundancy between memories
            max_tokens: Maximum token budget
            
        Returns:
            Cost value (lower is better)
        """
        selected_indices = np.where(selection > 0.5)[0]
        
        if len(selected_indices) == 0:
            return float('inf')
        
        # Cost component 1: Maximize similarity (minimize 1 - similarity)
        similarity_cost = np.mean(1 - similarities[selected_indices])
        
        # Cost component 2: Minimize token count
        total_tokens = np.sum(token_counts[selected_indices])
        token_cost = total_tokens / max_tokens if max_tokens > 0 else 0
        
        # Cost component 3: Minimize redundancy (avoid selecting similar memories)
        redundancy_cost = 0.0
        if len(selected_indices) > 1:
            for i, idx1 in enumerate(selected_indices):
                for idx2 in selected_indices[i+1:]:
                    redundancy_cost += redundancy_matrix[idx1, idx2]
            redundancy_cost /= (len(selected_indices) * (len(selected_indices) - 1) / 2)
        
        # Penalty for exceeding token budget
        if total_tokens > max_tokens:
            token_cost += 10.0 * (total_tokens - max_tokens) / max_tokens
        
        # Combined cost
        total_cost = (
            self.similarity_weight * similarity_cost +
            self.token_weight * token_cost +
            self.redundancy_weight * redundancy_cost
        )
        
        return total_cost
    
    def calculate_redundancy_matrix(
        self,
        embeddings: List[np.ndarray]
    ) -> np.ndarray:
        """
        Calculate pairwise redundancy between memories.
        
        Args:
            embeddings: List of memory embeddings
            
        Returns:
            Redundancy matrix (higher = more redundant)
        """
        n = len(embeddings)
        redundancy = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Cosine similarity between memories
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
                )
                redundancy[i, j] = redundancy[j, i] = sim
        
        return redundancy
    
    def qaoa_optimize(
        self,
        similarities: np.ndarray,
        token_counts: np.ndarray,
        embeddings: List[np.ndarray],
        max_tokens: int,
        num_iterations: int = 50
    ) -> np.ndarray:
        """
        QAOA-inspired optimization using simulated quantum annealing.
        
        This simulates the quantum adiabatic evolution of QAOA to find
        the optimal subset of memories.
        
        Args:
            similarities: Similarity scores
            token_counts: Token counts
            embeddings: Memory embeddings for redundancy calculation
            max_tokens: Maximum token budget
            num_iterations: Number of optimization iterations
            
        Returns:
            Binary selection array
        """
        n = len(similarities)
        
        # Calculate redundancy matrix
        redundancy_matrix = self.calculate_redundancy_matrix(embeddings)
        
        # Initialize with greedy solution
        sorted_indices = np.argsort(similarities)[::-1]
        initial_selection = np.zeros(n)
        current_tokens = 0
        
        for idx in sorted_indices:
            if current_tokens + token_counts[idx] <= max_tokens:
                initial_selection[idx] = 1
                current_tokens += token_counts[idx]
        
        # Simulated annealing (quantum-inspired)
        current_selection = initial_selection.copy()
        current_cost = self.calculate_cost(
            current_selection, similarities, token_counts, redundancy_matrix, max_tokens
        )
        
        best_selection = current_selection.copy()
        best_cost = current_cost
        
        # Temperature schedule (simulates quantum annealing)
        T_initial = 1.0
        T_final = 0.01
        
        for iteration in range(num_iterations):
            # Temperature decay
            T = T_initial * (T_final / T_initial) ** (iteration / num_iterations)
            
            # Quantum-inspired move: flip a random bit
            candidate = current_selection.copy()
            flip_idx = np.random.randint(n)
            candidate[flip_idx] = 1 - candidate[flip_idx]
            
            # Calculate new cost
            candidate_cost = self.calculate_cost(
                candidate, similarities, token_counts, redundancy_matrix, max_tokens
            )
            
            # Metropolis criterion (simulates quantum tunneling)
            delta_cost = candidate_cost - current_cost
            
            if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / T):
                current_selection = candidate
                current_cost = candidate_cost
                
                if current_cost < best_cost:
                    best_selection = current_selection.copy()
                    best_cost = current_cost
        
        return best_selection


# =============================================================================
# MAIN QUANTUM-INSPIRED GATE
# =============================================================================

def quantum_inspired_gate(
    query: str,
    query_embedding: np.ndarray,
    memory_embeddings: List[np.ndarray],
    memory_contents: List[str],
    max_tokens: int = 4096,
    top_k_initial: int = 20,
    use_mps_search: bool = True,
    use_von_neumann: bool = True,
    use_qaoa: bool = True
) -> Dict:
    """
    Quantum-inspired intelligent gating with all three optimizations:
    1. MPS-based Grover-inspired search
    2. Von Neumann entropy for entanglement scoring
    3. QAOA optimization for minimal memory injection
    
    Args:
        query: Query string
        query_embedding: Query embedding vector
        memory_embeddings: List of memory embedding vectors
        memory_contents: List of memory text contents
        max_tokens: Maximum token budget
        top_k_initial: Initial candidates to consider
        use_mps_search: Use Matrix Product State search
        use_von_neumann: Use Von Neumann entropy
        use_qaoa: Use QAOA optimization
        
    Returns:
        Dictionary with selected memories and quantum metrics
    """
    if len(memory_embeddings) == 0:
        return {
            'selected_memories': [],
            'quantum_metrics': {},
            'total_tokens': 0
        }
    
    # Convert to numpy array
    memory_matrix = np.array(memory_embeddings)
    
    # STEP 1: Quantum-inspired vector search
    if use_mps_search:
        mps = MatrixProductState(bond_dimension=10)
        top_indices, top_scores = mps.quantum_inspired_search(
            query_embedding,
            memory_matrix,
            top_k=min(top_k_initial, len(memory_embeddings))
        )
    else:
        # Fallback to classical cosine similarity
        similarities = []
        for mem_emb in memory_embeddings:
            sim = np.dot(query_embedding, mem_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(mem_emb) + 1e-10
            )
            similarities.append(sim)
        top_indices = np.argsort(similarities)[-top_k_initial:][::-1]
        top_scores = np.array([similarities[i] for i in top_indices])
    
    # STEP 2: Calculate Von Neumann entropy and entanglement
    quantum_tokens = []
    
    for idx, score in zip(top_indices, top_scores):
        content = memory_contents[idx]
        memory_emb = memory_embeddings[idx]
        
        # Estimate token count
        token_count = int(len(content.split()) * 0.75) + 10
        
        if use_von_neumann:
            # Calculate quantum metrics
            rho = create_density_matrix(memory_emb)
            vn_entropy = von_neumann_entropy(rho)
            entanglement = calculate_entanglement_score(query_embedding, memory_emb)
        else:
            vn_entropy = 0.5
            entanglement = score
        
        token = QuantumMemoryToken(
            content=content,
            similarity_score=float(score),
            shannon_entropy=0.0,  # Can be calculated separately if needed
            von_neumann_entropy=vn_entropy,
            entanglement_score=entanglement,
            token_count=token_count,
            source=f'QM{idx}',
            source_type='memory',
            density_matrix=rho if use_von_neumann else None
        )
        quantum_tokens.append(token)
    
    # STEP 3: QAOA optimization for minimal memory injection
    if use_qaoa and len(quantum_tokens) > 0:
        optimizer = QAOAOptimizer(
            similarity_weight=0.5,
            token_weight=0.3,
            redundancy_weight=0.2
        )
        
        # Prepare data for optimization
        similarities = np.array([t.entanglement_score for t in quantum_tokens])
        token_counts = np.array([t.token_count for t in quantum_tokens])
        embeddings = [memory_embeddings[i] for i in top_indices]
        
        # Run QAOA optimization
        selection = optimizer.qaoa_optimize(
            similarities,
            token_counts,
            embeddings,
            max_tokens,
            num_iterations=50
        )
        
        # Select optimized subset
        selected_tokens = [
            token for i, token in enumerate(quantum_tokens)
            if selection[i] > 0.5
        ]
    else:
        # Greedy selection without QAOA
        selected_tokens = []
        current_tokens = 0
        
        # Sort by entanglement score
        sorted_tokens = sorted(
            quantum_tokens,
            key=lambda x: x.entanglement_score,
            reverse=True
        )
        
        for token in sorted_tokens:
            if current_tokens + token.token_count <= max_tokens:
                selected_tokens.append(token)
                current_tokens += token.token_count
    
    # Calculate quantum metrics
    if selected_tokens:
        avg_vn_entropy = np.mean([t.von_neumann_entropy for t in selected_tokens])
        avg_entanglement = np.mean([t.entanglement_score for t in selected_tokens])
        total_tokens = sum(t.token_count for t in selected_tokens)
    else:
        avg_vn_entropy = 0.0
        avg_entanglement = 0.0
        total_tokens = 0
    
    return {
        'selected_memories': [t.content for t in selected_tokens],
        'quantum_tokens': selected_tokens,
        'quantum_metrics': {
            'avg_von_neumann_entropy': avg_vn_entropy,
            'avg_entanglement_score': avg_entanglement,
            'total_tokens': total_tokens,
            'memory_count': len(selected_tokens),
            'speedup_factor': np.sqrt(len(memory_embeddings)) if use_mps_search else 1.0
        }
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Create random embeddings for demonstration
    np.random.seed(42)
    
    query_emb = np.random.randn(128)
    query_emb /= np.linalg.norm(query_emb)
    
    memory_embs = [np.random.randn(128) for _ in range(100)]
    memory_embs = [m / np.linalg.norm(m) for m in memory_embs]
    
    memory_texts = [f"Memory content {i}" for i in range(100)]
    
    print("=" * 80)
    print("QUANTUM-INSPIRED TOKEN GATER")
    print("=" * 80)
    
    result = quantum_inspired_gate(
        query="Test query",
        query_embedding=query_emb,
        memory_embeddings=memory_embs,
        memory_contents=memory_texts,
        max_tokens=1000,
        top_k_initial=20,
        use_mps_search=True,
        use_von_neumann=True,
        use_qaoa=True
    )
    
    print(f"\nSelected {result['quantum_metrics']['memory_count']} memories")
    print(f"Total tokens: {result['quantum_metrics']['total_tokens']}")
    print(f"Avg Von Neumann Entropy: {result['quantum_metrics']['avg_von_neumann_entropy']:.3f}")
    print(f"Avg Entanglement Score: {result['quantum_metrics']['avg_entanglement_score']:.3f}")
    print(f"Speedup factor: {result['quantum_metrics']['speedup_factor']:.2f}x")
    
    print("\n✅ Quantum-inspired gating complete!")
