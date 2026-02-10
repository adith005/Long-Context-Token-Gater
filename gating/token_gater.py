import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from memory.history_db import retrieve_similar
from rag.retriever import retrieve


# =============================================================================
# ORIGINAL GATE FUNCTION (PRESERVED)
# =============================================================================

def gate(query, memory_weight=0.6, doc_weight=0.4, top_k=5):
    """
    Original simple gating function with weighted merging.
    
    Args:
        query: User query string
        memory_weight: Weight for memory results (default: 0.6)
        doc_weight: Weight for document results (default: 0.4)
        top_k: Number of results to return
        
    Returns:
        List of text strings from merged results
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


# =============================================================================
# NEW ENTROPY-BASED INTELLIGENT GATING
# =============================================================================

@dataclass
class MemoryToken:
    """Represents a memory segment with metadata"""
    content: str
    similarity_score: float
    entropy: float
    token_count: int
    source: str
    source_type: str  # 'memory' or 'document'


def calculate_shannon_entropy(text: str) -> float:
    """
    Calculate Shannon entropy to measure information content/uncertainty.
    
    Lower entropy = more predictable/certain = higher priority
    Higher entropy = more uncertain/random = lower priority
    
    Args:
        text: Text content to analyze
        
    Returns:
        Shannon entropy value
    """
    if not text:
        return float('inf')
    
    # Character-level entropy
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    total_chars = len(text)
    entropy = 0.0
    
    for count in char_counts.values():
        probability = count / total_chars
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy


def calculate_normalized_entropy(text: str) -> float:
    """
    Calculate normalized Shannon entropy (0-1 range).
    
    Args:
        text: Text content to analyze
        
    Returns:
        Normalized entropy value between 0 and 1
    """
    entropy = calculate_shannon_entropy(text)
    
    if entropy == float('inf'):
        return 1.0
    
    # Estimate based on typical character set
    vocab_size = 256  # ASCII + extended
    max_entropy = math.log2(vocab_size)
    
    if max_entropy == 0:
        return 0.0
    
    return min(entropy / max_entropy, 1.0)


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for a text segment.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Estimated token count
    """
    # Rough estimation: ~0.75 tokens per word
    words = len(text.split())
    return int(words * 0.75) + 10  # Add small buffer


def intelligent_gate(
    query: str,
    max_tokens: int = 4096,
    memory_top_k: int = 10,
    doc_top_k: int = 5,
    strategy: str = 'entropy_weighted',
    min_similarity: float = 0.0
) -> Dict:
    """
    Intelligent gating with entropy-based selection.
    
    This is the NEW method that:
    1. Searches memory with cosine similarity
    2. Calculates Shannon entropy for each result
    3. Creates context window with lowest uncertainty tokens
    
    Args:
        query: User query string
        max_tokens: Maximum tokens allowed in context window
        memory_top_k: Number of memories to retrieve initially
        doc_top_k: Number of documents to retrieve
        strategy: Selection strategy ('entropy_weighted', 'lowest_entropy', 'hybrid')
        min_similarity: Minimum similarity threshold
        
    Returns:
        Dictionary containing:
            - selected_memories: List of selected memory texts
            - selected_documents: List of selected document texts
            - total_tokens: Total token count
            - metadata: Additional information about selection
    """
    # Step 1: Retrieve similar memories using cosine similarity
    mem_results = retrieve_similar(query, k=memory_top_k)
    doc_results = retrieve(query, k=doc_top_k)
    
    # Step 2: Create MemoryToken objects and calculate entropy
    memory_tokens = []
    
    for idx, mem in enumerate(mem_results):
        content = mem.get("response", "")
        similarity = mem.get("similarity", 1.0)
        
        # Skip if below similarity threshold
        if similarity < min_similarity:
            continue
        
        # Calculate entropy
        entropy = calculate_normalized_entropy(content)
        token_count = estimate_token_count(content)
        
        memory_token = MemoryToken(
            content=content,
            similarity_score=similarity,
            entropy=entropy,
            token_count=token_count,
            source=f'M{idx + 1}',
            source_type='memory'
        )
        memory_tokens.append(memory_token)
    
    # Process documents similarly
    doc_tokens = []
    for idx, doc in enumerate(doc_results):
        content = doc if isinstance(doc, str) else doc.get("text", "")
        
        entropy = calculate_normalized_entropy(content)
        token_count = estimate_token_count(content)
        
        doc_token = MemoryToken(
            content=content,
            similarity_score=0.8,  # Default high score for retrieved docs
            entropy=entropy,
            token_count=token_count,
            source=f'D{idx + 1}',
            source_type='document'
        )
        doc_tokens.append(doc_token)
    
    # Step 3: Select tokens with lowest uncertainty based on strategy
    all_tokens = memory_tokens + doc_tokens
    
    if strategy == 'lowest_entropy':
        # Sort by entropy (ascending) - prefer low uncertainty
        sorted_tokens = sorted(all_tokens, key=lambda x: x.entropy)
    
    elif strategy == 'entropy_weighted':
        # Combine entropy and similarity: prefer high similarity + low entropy
        # Score = similarity * (1 - entropy)
        sorted_tokens = sorted(
            all_tokens,
            key=lambda x: x.similarity_score * (1 - x.entropy),
            reverse=True
        )
    
    elif strategy == 'hybrid':
        # Multi-factor ranking
        max_sim = max(t.similarity_score for t in all_tokens) if all_tokens else 1.0
        
        def hybrid_score(token):
            norm_similarity = token.similarity_score / max_sim if max_sim > 0 else 0
            certainty = 1 - token.entropy
            # Weighted combination: 60% similarity, 40% certainty
            return 0.6 * norm_similarity + 0.4 * certainty
        
        sorted_tokens = sorted(all_tokens, key=hybrid_score, reverse=True)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Select tokens that fit within context window
    selected_tokens = []
    current_token_count = 0
    
    for token in sorted_tokens:
        if current_token_count + token.token_count <= max_tokens:
            selected_tokens.append(token)
            current_token_count += token.token_count
        else:
            break
    
    # Separate memories and documents
    selected_memories = [t for t in selected_tokens if t.source_type == 'memory']
    selected_documents = [t for t in selected_tokens if t.source_type == 'document']
    
    return {
        'selected_memories': [m.content for m in selected_memories],
        'selected_documents': [d.content for d in selected_documents],
        'total_tokens': current_token_count,
        'metadata': {
            'memory_count': len(selected_memories),
            'document_count': len(selected_documents),
            'avg_memory_entropy': np.mean([m.entropy for m in selected_memories]) if selected_memories else 0,
            'avg_memory_similarity': np.mean([m.similarity_score for m in selected_memories]) if selected_memories else 0,
            'strategy': strategy,
            'memory_details': [
                {
                    'source': m.source,
                    'similarity': m.similarity_score,
                    'entropy': m.entropy,
                    'tokens': m.token_count
                }
                for m in selected_memories
            ]
        }
    }


def format_context_window(result: Dict, include_metadata: bool = False) -> str:
    """
    Format the intelligent_gate result into a context string for LLM.
    
    Args:
        result: Output from intelligent_gate()
        include_metadata: Whether to include metadata in output
        
    Returns:
        Formatted context string
    """
    parts = []
    
    # Add memories
    if result['selected_memories']:
        parts.append("=== RELEVANT CHAT HISTORY ===\n")
        for idx, memory in enumerate(result['selected_memories'], 1):
            if include_metadata:
                mem_meta = result['metadata']['memory_details'][idx - 1]
                parts.append(
                    f"[Memory {idx}] (Similarity: {mem_meta['similarity']:.3f}, "
                    f"Certainty: {1 - mem_meta['entropy']:.3f})\n"
                )
            parts.append(f"{memory}\n")
    
    # Add documents
    if result['selected_documents']:
        parts.append("\n=== EXTERNAL DOCUMENTS ===\n")
        for idx, doc in enumerate(result['selected_documents'], 1):
            parts.append(f"[Document {idx}] {doc}\n")
    
    return "\n".join(parts)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def gate_with_entropy(
    query: str,
    max_tokens: int = 4096,
    strategy: str = 'entropy_weighted'
) -> List[str]:
    """
    Simplified interface for entropy-based gating that returns just the texts.
    Similar signature to original gate() but with entropy selection.
    
    Args:
        query: User query string
        max_tokens: Maximum tokens in context
        strategy: Selection strategy
        
    Returns:
        List of selected text strings (memories + documents)
    """
    result = intelligent_gate(query, max_tokens=max_tokens, strategy=strategy)
    return result['selected_memories'] + result['selected_documents']

