# token_gater.py
# This file implements a query complexity gating system for RAG.

import time
from dataclasses import dataclass, field
from typing import List, Literal, Union, Iterator
from enum import Enum

# --- 1. Data Structures ---
# Represents a chunk of data from the vector DB
@dataclass
class Chunk:
    id: int
    content: str
    score: float = 0.0 # Bi-encoder similarity score
    relevance_score: float = 0.0 # Cross-encoder relevance score

# Enum to define the type of context pointer
class PointerType(Enum):
    HEURISTIC = "Heuristic"
    ATTESTED = "Attested"
    COMPRESSED = "Compressed"

# A "pointer" to a context, which can be of different types
@dataclass
class HybridContextPointer:
    type: PointerType
    chunk: Chunk = None    # Used for HEURISTIC or ATTESTED
    content: str = None  # Used for COMPRESSED
    score: float = 0.0     # The score (bi-encoder or cross-encoder)

# --- 2. Mock System Components (for Demonstration) ---
# These classes simulate the behavior of your models and DB.
# Replace these with your actual model/DB loading logic.

class MockQCC:
    """Mock Query Complexity Classifier"""
    def Classify(self, query: str) -> Literal['SIMPLE', 'COMPLEX', 'MODERATE']:
        print(f"🤖 QCC: Classifying query: '{query}'")
        # Simple mock logic
        if "complex" in query or "explain" in query or "how" in query:
            print("  -> Result: COMPLEX")
            return 'COMPLEX'
        print("  -> Result: SIMPLE")
        return 'SIMPLE'

class MockBiEncoder:
    """Mock Bi-Encoder"""
    def embed(self, query: str) -> List[float]:
        print(f"🤖 BiEncoder: Embedding query: '{query}'")
        return [0.1, 0.2, 0.3] # Dummy embedding

class MockCrossEncoder:
    """Mock Cross-Encoder for reranking"""
    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        print(f"🤖 CrossEncoder: Reranking {len(chunks)} chunks for query: '{query}'")
        # Mock reranking: just sort by ID and assign new scores
        reranked = sorted(chunks, key=lambda c: c.id, reverse=True)
        for i, chunk in enumerate(reranked):
            chunk.relevance_score = 0.99 - (i * 0.005)
        return reranked

class MockVectorDB:
    """Mock Vector Database"""
    def __init__(self):
        # Create a pool of 200 dummy chunks
        self.dummy_data = [
            Chunk(id=i, content=f"This is document chunk #{i}. It contains relevant information.") 
            for i in range(200)
        ]
        
    def search(self, embedding: List[float], k: int) -> List[Chunk]:
        print(f"🤖 VectorDB: Searching for top {k} chunks.")
        # Mock search: just return the first k chunks with dummy scores
        results = self.dummy_data[:k]
        for i, chunk in enumerate(results):
            # Simulate a bi-encoder score (higher is better)
            chunk.score = 0.95 - (i * 0.02) 
        return results

class MockLLM:
    """Mock Main LLM"""
    def generate(self, prompt: str) -> str:
        print(f"🤖 MainLLM: Generating response from prompt...")
        # (Uncomment the line below to see the full prompt in the console)
        # print("\n--- LLM PROMPT START ---\n" + prompt + "\n--- LLM PROMPT END ---\n")
        return f"This is the final AI-generated answer based on the provided context and query."

# --- 3. System Components (Initialized) ---
# In a real application, you would load your actual models here.
print("--- 1. Initializing System Components ---")
QCC = MockQCC()
BiEncoder = MockBiEncoder()
CrossEncoder = MockCrossEncoder()
VectorDB = MockVectorDB()
MainLLM = MockLLM()
print("--- System Ready ---\n")

# --- 4. Gating and Logic Functions ---

def ExecuteHeuristicGate(query: str, k: int, threshold: float) -> List[HybridContextPointer]:
    """Path A: Fast retrieval with heuristic pruning."""
    print(f"\n--- Path A: Executing Heuristic Gate (k={k}, threshold={threshold}) ---")
    query_embedding = BiEncoder.embed(query)
    candidate_chunks = VectorDB.search(query_embedding, k=k)
    
    final_pointers: List[HybridContextPointer] = []
    for chunk in candidate_chunks:
        # Fast heuristic pruning (the "gate")
        if chunk.score > threshold:
            print(f"  -> Heuristic PASS: Chunk {chunk.id} (Score: {chunk.score:.2f})")
            final_pointers.append(
                HybridContextPointer(
                    type=PointerType.HEURISTIC, 
                    chunk=chunk, 
                    score=chunk.score
                )
            )
        else:
            print(f"  -> Heuristic FAIL: Chunk {chunk.id} (Score: {chunk.score:.2f})")

    return final_pointers

def ExecuteLearnedGate(query: str, k_initial: int, k_final: int) -> List[HybridContextPointer]:
    """Path B: Expensive reranking for high-relevance context."""
    print(f"\n--- Path B: Executing Learned Gate (k_initial={k_initial}, k_final={k_final}) ---")
    
    # 1. Initial Retrieval (a subset of Path A)
    query_embedding = BiEncoder.embed(query)
    candidate_chunks = VectorDB.search(query_embedding, k=k_initial)
    print(f"  1. Retrieved {len(candidate_chunks)} initial candidates.")

    # 2. High-cost "Bookkeeping": Cross-Encoder Reranking
    reranked_chunks = CrossEncoder.rerank(query, candidate_chunks)
    print(f"  2. Reranked all candidates.")

    final_pointers: List[HybridContextPointer] = []
    # 3. Attest the top-N "Retrieval Head" chunks
    print(f"  3. Attesting top {k_final} chunks:")
    for chunk in reranked_chunks[:k_final]:
        print(f"  -> Attested PASS: Chunk {chunk.id} (Relevance: {chunk.relevance_score:.2f})")
        final_pointers.append(
            HybridContextPointer(
                type=PointerType.ATTESTED,
                chunk=chunk,
                score=chunk.relevance_score
            )
        )
    
    # Optional: "Streaming Head" logic to compress peripherals
    # if len(reranked_chunks) > k_final:
    #     print(f"  (Optional) Compressing {len(reranked_chunks) - k_final} peripheral chunks...")
    #     # compressed_content = SummarizerLLM.compress(reranked_chunks[k_final:])
    #     compressed_content = "This is a summary of the other less relevant chunks."
    #     final_pointers.append(
    #         HybridContextPointer(type=PointerType.COMPRESSED, content=compressed_content)
    #     )

    return final_pointers

def AssemblePrompt(query: str, pointers: List[HybridContextPointer]) -> str:
    """Builds the final prompt string from the list of context pointers."""
    print("\n--- 3. Assembling Final Prompt ---")
    
    # Using a list of strings and then joining is Python's "StringBuilder"
    prompt_builder: List[str] = []
    prompt_builder.append("--- Contextual Information ---")
    
    if not pointers:
        prompt_builder.append("No context found.")

    for ptr in pointers:
        if ptr.type == PointerType.HEURISTIC or ptr.type == PointerType.ATTESTED:
            # "Serialize" the chunk content directly into the prompt buffer
            prompt_builder.append(f"\n[Source: Chunk {ptr.chunk.id} | Type: {ptr.type.value} | Score: {ptr.score:.3f}]")
            prompt_builder.append(ptr.chunk.content)
            
        elif ptr.type == PointerType.COMPRESSED:
            prompt_builder.append(f"\n[Source: Compressed Summary | Type: {ptr.type.value}]")
            prompt_builder.append(ptr.content)
            
        prompt_builder.append("---")

    prompt_builder.append("\n\n--- Query ---")
    prompt_builder.append(query)
    
    print(f"  -> Assembled prompt with {len(pointers)} context items.")
    return "\n".join(prompt_builder)

def GenerateResponse(query: str) -> str:
    """Main entry point for the RAG pipeline."""
    
    print(f"\n{'='*50}")
    print(f"Processing query: '{query}'")
    print(f"{'='*50}")
    
    # 1. Gating: The "512-byte threshold" analogue
    print("--- 1. Gating: Classifying query complexity ---")
    complexity = QCC.Classify(query)

    # 2. Path Selection
    print("\n--- 2. Path Selection: Retrieving context ---")
    context_pointers: List[HybridContextPointer]
    if complexity == 'SIMPLE':
        # Path A: The "Copy" Path (Low compute, high token noise)
        context_pointers = ExecuteHeuristicGate(query, k=20, threshold=0.7) # Note: threshold is 0.7
    else: # 'COMPLEX' or 'MODERATE'
        # Path B: The "Zero-Copy" Path (High compute, low token noise)
        context_pointers = ExecuteLearnedGate(query, k_initial=150, k_final=10)

    # 3. Semantic Serialization
    final_prompt = AssemblePrompt(query, context_pointers)
    
    # 4. Final I/O (Transmission to LLM)
    print("\n--- 4. Final Generation ---")
    response = MainLLM.generate(final_prompt)
    
    print(f"\n✅ Final Response: {response}")
    return response

# --- 5. Example Usage ---
if __name__ == "__main__":
    
    # Example 1: Simple Query (should trigger Path A)
    # The threshold is 0.7, and the mock DB scores are > 0.7 for most
    simple_query = "What is the project status?"
    GenerateResponse(simple_query)

    print("\n\n" + "="*50 + "\n\n")

    # Example 2: Complex Query (should trigger Path B)
    # This will retrieve 150, rerank them, and take the top 10
    complex_query = "How does the complex reranking system work?"
    GenerateResponse(complex_query)