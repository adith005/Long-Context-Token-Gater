import redis
import numpy as np
import uuid
import time
from typing import List, Dict, Tuple
from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB
from utils.embedding import embed

# ============================================================
# Redis Memory Store Class
# ============================================================

class RedisMemoryStore:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, dim=384):
        self.dim = dim
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=False)

    def _serialize_vector(self, vector: np.ndarray) -> bytes:
        return vector.astype(np.float32).tobytes()

    def _deserialize_vector(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def store_cluster_centroid(self, cluster_id: int, centroid: np.ndarray):
        key = f"cluster:{cluster_id}:centroid"
        self.redis.set(key, self._serialize_vector(centroid))

    def get_top_k_clusters(self, query_vec: np.ndarray, k: int) -> List[int]:
        cluster_keys = self.redis.keys("cluster:*:centroid")
        similarities = []
        for key in cluster_keys:
            try:
                cluster_id = int(key.decode().split(":")[1])
                centroid = self._deserialize_vector(self.redis.get(key))
                sim = self._cosine_similarity(query_vec, centroid)
                similarities.append((cluster_id, sim))
            except:
                continue
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in similarities[:k]]

    def ingest_memory(self, content: str, embedding: np.ndarray, cluster_id: int, token_len: int, source: str = "chat"):
        memory_id = str(uuid.uuid4())
        key = f"mem:{cluster_id}:{memory_id}"
        pipeline = self.redis.pipeline()
        pipeline.hset(key, mapping={
            "content": content,
            "cluster_id": cluster_id,
            "token_len": token_len,
            "timestamp": int(time.time()),
            "usefulness": 0.5,
            "access_count": 0,
            "source": source
        })
        pipeline.hset(key, "embedding", self._serialize_vector(embedding))
        pipeline.execute()
        return memory_id

    def retrieve_all(self, query_vec: np.ndarray, top_k: int = 10) -> List[Dict]:
        # Fallback to simple search if clusters aren't used/set up
        keys = self.redis.keys("mem:*")
        results = []
        for key in keys:
            data = self.redis.hgetall(key)
            if b"embedding" not in data: continue
            vec = self._deserialize_vector(data[b"embedding"])
            sim = self._cosine_similarity(query_vec, vec)
            
            results.append({
                "content": data[b"content"].decode(),
                "confidence": sim * 100, # Normalize to percentage for gater
                "source": data[b"source"].decode(),
                "timestamp": int(data[b"timestamp"])
            })
        
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0: return 0
        return np.dot(a, b) / (norm_a * norm_b)

# ── Global Instance & Helpers ────────────────────────────────────────────────

_store = RedisMemoryStore()

def retrieve_memory(query_vec: np.ndarray) -> List[Dict]:
    return _store.retrieve_all(query_vec)

def remember(source: str, content: str):
    vec = embed(content)
    # Using a default cluster 0 for simplicity in this stage
    _store.ingest_memory(content, vec, cluster_id=0, token_len=len(content)//4, source=source)
