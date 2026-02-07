from fastapi import FastAPI
from pydantic import BaseModel
from token_optimiser.retriever import Retriever
from token_optimiser.api import TokenOptimiser
from token_optimiser.models import GateConfig, GateResult
import uvicorn

app = FastAPI(title="TokenOptimiser - Memory API")

# init singletons (mute in production)
retriever = Retriever()
# optionally load data file:
# retriever.load_from_jsonl("data/sample_memory.jsonl")
optimiser = TokenOptimiser(retriever=retriever)

class GateRequest(BaseModel):
    query: str
    strategy: str = "heuristic"
    top_k: int = 10
    token_budget: int = 1024

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/gate", response_model=GateResult)
def gate(req: GateRequest):
    cfg = GateConfig(strategy=req.strategy, top_k=req.top_k, token_budget=req.token_budget)
    res = optimiser.gate(req.query, cfg)
    return res

# to run: uvicorn app.server:app --reload --port 8000
