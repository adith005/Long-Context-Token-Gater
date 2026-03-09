"""
tracer.py  —  Pipeline Step Tracer
===================================

Collects timestamped events from each pipeline step.
Passed into run_pipeline() as an optional argument.
Zero overhead when not provided (tracer=None path is a no-op).

Each event:
    {
        "step":    int,           # 1-8
        "name":    str,           # human label
        "status":  "ok"|"error",
        "data":    dict,          # step-specific payload
        "elapsed": float,         # seconds since pipeline start
    }

Usage
-----
    from tracer import PipelineTracer

    tracer = PipelineTracer()
    result = run_pipeline(query, gating_mode="entropy", tracer=tracer)

    for event in tracer.events:
        print(event)
"""

import time
import json
import numpy as np
from dataclasses import dataclass, field


class _NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalars and arrays to native Python types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _sanitize(data):
    """Recursively convert numpy types in a dict/list to JSON-safe types."""
    return json.loads(json.dumps(data, cls=_NumpyEncoder))


@dataclass
class PipelineTracer:
    events: list = field(default_factory=list)
    _start: float = field(default_factory=time.time)

    def log(self, step: int, name: str, data: dict, status: str = "ok"):
        self.events.append({
            "step":    step,
            "name":    name,
            "status":  status,
            "data":    _sanitize(data),
            "elapsed": round(time.time() - self._start, 3),
        })

    def reset(self):
        self.events.clear()
        self._start = time.time()