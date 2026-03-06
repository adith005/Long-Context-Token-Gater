from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

class MemoryEntry(BaseModel):
    id: str
    text: str
    metadata: dict = {}
    created_at: Optional[datetime] = None

class GateConfig(BaseModel):
    strategy: str = "heuristic"  # heuristic | attention | learned
    top_k: int = 10
    token_budget: int = 1024

class GateResult(BaseModel):
    selected: List[MemoryEntry]
    token_estimate: int
    meta: dict = {}
