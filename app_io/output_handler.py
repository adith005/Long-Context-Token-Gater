"""
output_handler.py

Formats pipeline results for frontend display.
No retrieval. No gating. No LLM calls.
"""

from typing import Dict


def process_output(
    gated_response: Dict,
    non_gated_response: Dict,
    gating_stats: Dict | None = None
) -> dict:
    """
    Wrap responses into a clean structure for frontend.
    """

    return {
        "gated": {
            "response_text": gated_response.get("response_text"),
            "prompt_tokens": gated_response.get("prompt_tokens"),
            "completion_tokens": gated_response.get("completion_tokens"),
            "total_time_sec": gated_response.get("total_time_sec"),
        },
        "non_gated": {
            "response_text": non_gated_response.get("response_text"),
            "prompt_tokens": non_gated_response.get("prompt_tokens"),
            "completion_tokens": non_gated_response.get("completion_tokens"),
            "total_time_sec": non_gated_response.get("total_time_sec"),
        },
        "gating_stats": gating_stats
    }