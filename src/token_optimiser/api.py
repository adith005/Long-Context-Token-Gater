from .retriever import Retriever
from .prompt import PromptBuilder
from .heuristic import top_k_by_similarity
from .attention import attention_score_stub
from .models import GateConfig, GateResult
from .utils import total_tokens
from typing import List

class TokenOptimiser:
    def __init__(self, retriever: Retriever, token_budget: int = 1024):
        self.retriever = retriever
        self.prompt_builder = PromptBuilder(token_budget=token_budget)

    def gate(self, query: str, config: GateConfig):
        if config.strategy == "heuristic":
            candidates = top_k_by_similarity(self.retriever, query, top_k=config.top_k)
            selected = candidates
        elif config.strategy == "attention":
            candidates = top_k_by_similarity(self.retriever, query, top_k=config.top_k*2)
            scored = attention_score_stub(query, candidates)
            selected = [e for e, s in scored[: config.top_k]]
        else:
            # learned - not implemented
            raise NotImplementedError("Learned strategy not implemented")

        prompt = self.prompt_builder.build(query, selected)
        token_est = self.prompt_builder.estimate_prompt_tokens(prompt)
        return GateResult(selected=selected, token_estimate=token_est, meta={"strategy": config.strategy})
