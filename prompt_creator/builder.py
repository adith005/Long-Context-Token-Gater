"""
prompt_creator/builder.py

Formats retrieved context items into a single prompt string.
"""

def build_prompt(context_items: list[dict], query: str) -> str:
    """
    Constructs a RAG prompt using the provided context items and user query.
    """
    
    context_str = ""
    for idx, item in enumerate(context_items):
        # Handle both document and memory item formats
        content = item.get("sentence") or item.get("content") or ""
        source = item.get("doc_name") or item.get("source") or "Unknown"
        
        context_str += f"[{idx+1}] Source: {source}\n{content}\n\n"
    
    prompt = f"""Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

CONTEXT:
{context_str}

QUESTION: {query}

ANSWER:"""
    
    return prompt
