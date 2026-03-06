"""
embedding.py

Central embedding layer.
Loaded once. Shared everywhere.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

# Load model once at import time
_model = SentenceTransformer(MODEL_NAME)


def embed(text: str) -> np.ndarray:
    """
    Returns L2-normalized float32 vector.
    """
    return _model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0].astype(np.float32)