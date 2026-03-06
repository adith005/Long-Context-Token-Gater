from typing import List
from .models import MemoryEntry, GateConfig
from .utils import estimate_tokens, total_tokens

class PromptBuilder:
    def __init__(self, token_budget: int = 1024):
        self.token_budget = token_budget

    def build(self, query: str, memories: List[MemoryEntry]) -> str:
        # simple greedy pack by token estimate
        parts = [f"User: {query}\n"]
        remaining = self.token_budget - estimate_tokens(parts[0])
        for m in memories:
            t = estimate_tokens(m.text)
            if t <= remaining:
                parts.append(f"Memory: {m.text}\n")
                remaining -= t
            else:
                # skip overly large memories
                continue
        parts.append("Assistant:")
        return "\n".join(parts)

    def estimate_prompt_tokens(self, prompt_text: str) -> int:
        return estimate_tokens(prompt_text)
