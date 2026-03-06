from .models import MemoryEntry, GateConfig
from typing import List

class LearnedGater:
    """
    Placeholder for learned gating model. Collects labeled tuples for future training.
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self._is_trained = False

    def predict(self, query: str, candidates: List[MemoryEntry]):
        raise NotImplementedError("Learned gater is future work. Use human-in-the-loop logs to bootstrap.")

    def collect_label(self, query: str, memory: MemoryEntry, label: int):
        # append to local dataset for bootstrapping
        # expected format: (query, memory_id, label)
        pass
