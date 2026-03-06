import redis
import numpy as np
import uuid
import time
from typing import List, Dict, Optional


class RedisMemoryStore:
    """
    Redis-backed clustered memory store.

    Responsibilities:
    - Store memory chunks
    - Store cluster centroids
    - Retrieve memories via cluster-pruned vector search
    - Provide basic update/delete utilities

    No gating logic is implemented here.
    """

    def __init__(self, host="localhost", port=6379, dim=768):
        self.dim = dim
        self.redis = redis.Redis(host=host, port=port, decode_responses=False)

    # ============================================================
    # ---------------- VECTOR SERIALIZATION ----------------------
    # ============================================================

    def _to_bytes(self, vector: np.ndarray) -> bytes:
        """Convert float32 numpy vector to raw bytes."""
        return vector.astype(np.float32).tobytes()

    def _from_bytes(self, blob: bytes) -> np.ndarray:
        """Convert raw bytes back to numpy float32 vector."""
        return np.frombuffer(blob, dtype=np.float32)

    # ============================================================
    # ---------------- CLUSTER OPERATIONS ------------------------
    # ============================================================

    def store_cluster_centroid(self, cluster_id: int, centroid: np.ndarray):
        """
        Store a cluster centroid vector.
        """
        key = f"cluster:{cluster_id}:centroid"
        self.redis.set(key, self._to_bytes(centroid))

    def get_cluster_centroid(self, cluster_id: int) -> Optional[np.ndarray]:
        """
        Retrieve a cluster centroid.
        """
        key = f"cluster:{cluster_id}:centroid"
        blob = self.redis.get(key)
        if blob is None:
            return None
        return self._from_bytes(blob)

    def list_cluster_ids(self) -> List[int]:
        """
        List all stored cluster IDs.
        """
        keys = self.redis.keys("cluster:*:centroid")
        return [int(k.decode().split(":")[1]) for k in keys]

    # ============================================================
    # ---------------- MEMORY INGESTION --------------------------
    # ============================================================

    def ingest_memory(
        self,
        content: str,
        embedding: np.ndarray,
        cluster_id: int,
        token_len: int,
        source: str = "chat"
    ) -> str:
        """
        Store a memory chunk in Redis.

        Returns:
            memory_id (str)
        """
        memory_id = str(uuid.uuid4())
        key = f"mem:{cluster_id}:{memory_id}"

        pipe = self.redis.pipeline()

        # Store metadata fields
        pipe.hset(key, mapping={
            "content": content,
            "cluster_id": cluster_id,
            "token_len": token_len,
            "timestamp": int(time.time()),
            "usefulness": 0.5,      # initial neutral value
            "access_count": 0,
            "last_access": 0,
            "source": source
        })

        # Store embedding separately inside same hash
        pipe.hset(key, "embedding", self._to_bytes(embedding))

        pipe.execute()

        return memory_id

    # ============================================================
    # ---------------- VECTOR RETRIEVAL --------------------------
    # ============================================================

    def search_within_clusters(
        self,
        query_vector: np.ndarray,
        cluster_ids: List[int],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Perform KNN search restricted to specific cluster IDs.

        Returns:
            List of memory records (metadata only, no embedding).
        """

        if not cluster_ids:
            return []

        # Build cluster filter expression
        cluster_filter = " | ".join(
            [f"@cluster_id:[{cid} {cid}]" for cid in cluster_ids]
        )

        # RediSearch KNN query
        query = f"({cluster_filter})=>[KNN {top_k} @embedding $vec]"

        result = self.redis.execute_command(
            "FT.SEARCH",
            "mem_idx",
            query,
            "PARAMS", 2,
            "vec", self._to_bytes(query_vector),
            "RETURN", 6,
            "content",
            "cluster_id",
            "token_len",
            "timestamp",
            "usefulness",
            "access_count",
            "DIALECT", 2
        )

        return self._parse_search_result(result)

    # ============================================================
    # ---------------- MEMORY FETCHING ---------------------------
    # ============================================================

    def get_memory(self, memory_key: str) -> Dict:
        """
        Retrieve full memory hash by key.
        """
        data = self.redis.hgetall(memory_key)
        if not data:
            return {}

        parsed = {}
        for k, v in data.items():
            key_str = k.decode()
            if key_str == "embedding":
                continue  # do not return embedding by default
            parsed[key_str] = v.decode()

        parsed["key"] = memory_key
        return parsed

    # ============================================================
    # ---------------- MEMORY UPDATES ----------------------------
    # ============================================================

    def increment_access(self, memory_key: str):
        """
        Increment access counter and update last access timestamp.
        """
        pipe = self.redis.pipeline()
        pipe.hincrby(memory_key, "access_count", 1)
        pipe.hset(memory_key, "last_access", int(time.time()))
        pipe.execute()

    def update_usefulness(self, memory_key: str, new_value: float):
        """
        Directly set usefulness value.
        """
        self.redis.hset(memory_key, "usefulness", new_value)

    # ============================================================
    # ---------------- MEMORY DELETION ---------------------------
    # ============================================================

    def delete_memory(self, memory_key: str):
        """
        Delete a memory entry.
        """
        self.redis.delete(memory_key)

    def delete_cluster(self, cluster_id: int):
        """
        Delete all memories belonging to a cluster.
        """
        keys = self.redis.keys(f"mem:{cluster_id}:*")
        if keys:
            self.redis.delete(*keys)

    # ============================================================
    # ---------------- SEARCH RESULT PARSER ----------------------
    # ============================================================

    def _parse_search_result(self, raw) -> List[Dict]:
        """
        Convert RediSearch raw output to structured dict list.
        """
        if not raw or len(raw) < 2:
            return []

        results = []

        for i in range(1, len(raw), 2):
            key = raw[i].decode()
            fields = raw[i + 1]

            entry = {"key": key}

            for j in range(0, len(fields), 2):
                field = fields[j].decode()
                value = fields[j + 1]
                entry[field] = value.decode() if isinstance(value, bytes) else value

            results.append(entry)

        return results