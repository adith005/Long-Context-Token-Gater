# minimal harness to test gating and print token savings ratio vs baseline
from token_optimiser.retriever import Retriever
from token_optimiser.api import TokenOptimiser
from token_optimiser.models import GateConfig
from token_optimiser.prompt import PromptBuilder

def run_one(query):
    r = Retriever()
    r.load_from_jsonl("data/sample_memory.jsonl")
    opt = TokenOptimiser(r, token_budget=1024)

    baseline_cfg = GateConfig(strategy="heuristic", top_k=50, token_budget=1024)
    g = opt.gate(query, baseline_cfg)
    print("selected:", len(g.selected), "tokens:", g.token_estimate)

if __name__ == "__main__":
    run_one("What is the user's preferred language for communication?")
