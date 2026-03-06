"""
input_handler.py

Normalizes frontend inputs into structured events.
"""

import uuid
import datetime
from typing import Optional


def process_chat_input(user_query: str) -> dict:
    return {
        "type": "chat",
        "request_id": str(uuid.uuid4()),
        "user_query": user_query.strip(),
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    }


def process_document_input(file_name: str, file_path: str) -> dict:
    return {
        "type": "document",
        "doc_id": str(uuid.uuid4()),
        "file_name": file_name,
        "file_path": file_path,
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    }