"""
llm_handler/handler.py

Handles communication with the LLM provider.
Currently configured for LM Studio (local OpenAI-compatible API).
"""

import requests
import time
from config.settings import LLM_PROVIDER, LLM_MODEL

def call_llm(prompt: str) -> dict:
    """
    Calls the configured LLM provider and returns the response with token counts.
    """
    
    # LM Studio default endpoint
    url = "http://localhost:1234/v1/chat/completions"
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer the user query."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        elapsed = time.time() - start_time
        
        return {
            "response_text":    data["choices"][0]["message"]["content"],
            "prompt_tokens":    data["usage"]["prompt_tokens"],
            "completion_tokens":data["usage"]["completion_tokens"],
            "total_time_sec":   elapsed,
            "error":            None,
        }
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return {
            "response_text": f"Error: Could not connect to LLM ({e})",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_time_sec": time.time() - start_time
        }