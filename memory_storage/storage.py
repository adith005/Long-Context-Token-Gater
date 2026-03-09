"""
memory_storage/storage.py
==========================

Redis-backed memory store with full retrieval intelligence.

Features
--------
  1. Deduplication on write   — exact skip / near-dup merge
  2. Access-count weighting   — frequently retrieved memories rank higher
  3. Recency decay            — recent memories get a freshness boost
  4. Usefulness feedback      — pipeline signals which memories were actually used
  5. TTL expiry               — stale, low-usefulness entries auto-expire
  6. Source weighting         — document memories outrank chat memories
  7. Min-similarity floor     — irrelevant memories never enter the context window

Scoring formula (retrieve_all)
-------------------------------
  raw_score = (SIM_W   × cosine_sim)
            + (ACCESS_W × norm_access_count)
            + (RECENCY_W × recency_score)       recency = e^(-λ × days_old)
            + (SOURCE_W  × source_weight)        per-source multiplier
  confidence = raw_score × usefulness × 100

  Entries below MIN_SIM_FLOOR are discarded before scoring.

All weights and thresholds live in config/settings.py.
"""

import math
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import redis

from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB
from utils.embedding import embed

# ── Settings with fallbacks ───────────────────────────────────────────────────
def _cfg(name, default):
    try:
        from config import settings
        return getattr(settings, name, default)
    except Exception:
        return default

# Dedup
MEMORY_EXACT_THRESHOLD  = _cfg("MEMORY_EXACT_THRESHOLD",  0.97)
MEMORY_MERGE_THRESHOLD  = _cfg("MEMORY_MERGE_THRESHOLD",  0.85)

# Retrieval weights  (should sum to 1.0)
RETRIEVAL_SIM_WEIGHT    = _cfg("RETRIEVAL_SIM_WEIGHT",    0.50)
RETRIEVAL_ACCESS_WEIGHT = _cfg("RETRIEVAL_ACCESS_WEIGHT", 0.15)
RETRIEVAL_RECENCY_WEIGHT= _cfg("RETRIEVAL_RECENCY_WEIGHT",0.20)
RETRIEVAL_SOURCE_WEIGHT = _cfg("RETRIEVAL_SOURCE_WEIGHT", 0.15)

# Recency decay — λ in e^(-λ × days).  0.1 → ~half-life ≈ 7 days
RECENCY_DECAY_LAMBDA    = _cfg("RECENCY_DECAY_LAMBDA",    0.1)

# Min cosine similarity — entries below this are never returned
MIN_SIM_FLOOR           = _cfg("MIN_SIM_FLOOR",           0.30)

# TTL — entries older than this with usefulness < TTL_USEFULNESS_FLOOR get expired
MEMORY_TTL_DAYS         = _cfg("MEMORY_TTL_DAYS",         30)
MEMORY_TTL_USEFULNESS   = _cfg("MEMORY_TTL_USEFULNESS",   0.3)

# Source weights — how much to trust each source type
SOURCE_WEIGHTS: Dict[str, float] = {
    "exchange": 0.8,   # conversational Q+A
    "chat":     0.6,   # raw chat turn
    "document": 1.0,   # ingested document (most authoritative)
}
DEFAULT_SOURCE_WEIGHT = 0.7


# ═════════════════════════════════════════════════════════════════════════════
# Redis Memory Store
# ═════════════════════════════════════════════════════════════════════════════

class RedisMemoryStore:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, dim=384):
        self.dim   = dim
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=False)

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def _serialize_vector(self, v: np.ndarray) -> bytes:
        return v.astype(np.float32).tobytes()

    def _deserialize_vector(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0

    # ── Scoring components ────────────────────────────────────────────────────

    def _recency_score(self, timestamp: int) -> float:
        """Exponential decay: score=1 when brand new, drops toward 0 over days."""
        days_old = (time.time() - timestamp) / 86400.0
        return math.exp(-RECENCY_DECAY_LAMBDA * days_old)

    def _source_weight(self, source: str) -> float:
        return SOURCE_WEIGHTS.get(source, DEFAULT_SOURCE_WEIGHT)

    # ── Cluster helpers ───────────────────────────────────────────────────────

    def store_cluster_centroid(self, cluster_id: int, centroid: np.ndarray):
        self.redis.set(f"cluster:{cluster_id}:centroid", self._serialize_vector(centroid))

    def get_top_k_clusters(self, query_vec: np.ndarray, k: int) -> List[int]:
        cluster_keys = self.redis.keys("cluster:*:centroid")
        sims = []
        for key in cluster_keys:
            try:
                cid = int(key.decode().split(":")[1])
                c   = self._deserialize_vector(self.redis.get(key))
                sims.append((cid, self._cosine_similarity(query_vec, c)))
            except Exception:
                continue
        sims.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in sims[:k]]

    # ── Dedup helpers ─────────────────────────────────────────────────────────

    def _find_most_similar(
        self, embedding: np.ndarray
    ) -> Optional[Tuple[bytes, float, dict]]:
        keys = self.redis.keys("mem:*")
        if not keys:
            return None
        best_key, best_sim, best_data = None, -1.0, None
        for key in keys:
            data = self.redis.hgetall(key)
            if b"embedding" not in data:
                continue
            sim = self._cosine_similarity(embedding, self._deserialize_vector(data[b"embedding"]))
            if sim > best_sim:
                best_key, best_sim, best_data = key, sim, data
        return (best_key, best_sim, best_data) if best_key else None

    def _merge_into(self, existing_key: bytes, new_embedding: np.ndarray, new_content: str):
        data         = self.redis.hgetall(existing_key)
        old_vec      = self._deserialize_vector(data[b"embedding"])
        merged_vec   = (old_vec + new_embedding) / 2.0
        norm         = np.linalg.norm(merged_vec)
        if norm > 0:
            merged_vec /= norm
        old_content  = data[b"content"].decode()
        keep_content = new_content if len(new_content) > len(old_content) else old_content
        old_count    = int(data.get(b"access_count", b"0"))
        pipe = self.redis.pipeline()
        pipe.hset(existing_key, "content",      keep_content)
        pipe.hset(existing_key, "access_count", old_count + 1)
        pipe.hset(existing_key, "embedding",    self._serialize_vector(merged_vec))
        pipe.execute()

    # ── Ingest ────────────────────────────────────────────────────────────────

    def _dedup_check(self, embedding: np.ndarray, content: str) -> bool:
        """
        Returns True if the entry should be stored (novel enough).
        Side-effects: skips or merges as appropriate.
        """
        match = self._find_most_similar(embedding)
        if not match:
            return True
        key, sim, data = match
        if sim >= MEMORY_EXACT_THRESHOLD:
            self.redis.hset(key, "access_count", int(data.get(b"access_count", b"0")) + 1)
            return False
        if sim >= MEMORY_MERGE_THRESHOLD:
            self._merge_into(key, embedding, content)
            return False
        return True

    def ingest_memory(
        self,
        content:    str,
        embedding:  np.ndarray,
        cluster_id: int,
        token_len:  int,
        source:     str = "chat",
    ) -> Optional[str]:
        if not self._dedup_check(embedding, content):
            return None
        memory_id = str(uuid.uuid4())
        key       = f"mem:{cluster_id}:{memory_id}".encode()
        pipe      = self.redis.pipeline()
        pipe.hset(key, mapping={
            "content":      content,
            "cluster_id":   cluster_id,
            "token_len":    token_len,
            "timestamp":    int(time.time()),
            "usefulness":   0.5,
            "access_count": 0,
            "source":       source,
        })
        pipe.hset(key, "embedding", self._serialize_vector(embedding))
        pipe.execute()
        return memory_id

    def ingest_exchange(
        self,
        user_content:      str,
        assistant_content: str,
        cluster_id:        int = 0,
    ) -> Optional[str]:
        """Store a Q+A exchange as one deduplicated memory unit."""
        combined  = f"Q: {user_content}\nA: {assistant_content}"
        embedding = embed(combined)
        if not self._dedup_check(embedding, combined):
            return None
        memory_id = str(uuid.uuid4())
        key       = f"mem:{cluster_id}:{memory_id}".encode()
        pipe      = self.redis.pipeline()
        pipe.hset(key, mapping={
            "content":           combined,
            "user_content":      user_content,
            "assistant_content": assistant_content,
            "cluster_id":        cluster_id,
            "token_len":         len(combined) // 4,
            "timestamp":         int(time.time()),
            "usefulness":        0.5,
            "access_count":      0,
            "source":            "exchange",
        })
        pipe.hset(key, "embedding", self._serialize_vector(embedding))
        pipe.execute()
        return memory_id

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve_all(self, query_vec: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Retrieve top-k memories using the full weighted scoring formula.

        Scoring
        -------
          raw  = SIM_W×sim + ACCESS_W×norm_access + RECENCY_W×recency + SOURCE_W×src_w
          conf = raw × usefulness × 100

        Entries below MIN_SIM_FLOOR are dropped before scoring.
        """
        keys = self.redis.keys("mem:*")
        candidates = []

        for key in keys:
            data = self.redis.hgetall(key)
            if b"embedding" not in data:
                continue

            vec = self._deserialize_vector(data[b"embedding"])
            sim = self._cosine_similarity(query_vec, vec)

            # ── 7. Min-similarity floor ────────────────────────────────────
            if sim < MIN_SIM_FLOOR:
                continue

            candidates.append({
                "key":          key,
                "content":      data[b"content"].decode(),
                "source":       data[b"source"].decode(),
                "sim":          sim,
                "access_count": int(data.get(b"access_count", b"0")),
                "timestamp":    int(data.get(b"timestamp",    b"0")),
                "usefulness":   float(data.get(b"usefulness", b"0.5")),
            })

        if not candidates:
            return []

        # ── Normalise access_count across candidates ───────────────────────
        max_access = max(c["access_count"] for c in candidates) or 1

        for c in candidates:
            norm_access = c["access_count"] / max_access

            # ── 3. Recency decay ───────────────────────────────────────────
            recency = self._recency_score(c["timestamp"])

            # ── 6. Source weighting ────────────────────────────────────────
            src_w = self._source_weight(c["source"])

            raw = (
                RETRIEVAL_SIM_WEIGHT     * c["sim"]     +
                RETRIEVAL_ACCESS_WEIGHT  * norm_access  +
                RETRIEVAL_RECENCY_WEIGHT * recency       +
                RETRIEVAL_SOURCE_WEIGHT  * src_w
            )

            # ── 4. Usefulness scaling ──────────────────────────────────────
            c["confidence"] = raw * c["usefulness"] * 100

        candidates.sort(key=lambda x: x["confidence"], reverse=True)

        # Strip internal fields before returning
        results = []
        for c in candidates[:top_k]:
            results.append({
                "content":      c["content"],
                "confidence":   round(c["confidence"], 4),
                "source":       c["source"],
                "timestamp":    c["timestamp"],
                "access_count": c["access_count"],
                "usefulness":   c["usefulness"],
            })
        return results

    # ── Usefulness feedback ───────────────────────────────────────────────────

    def update_usefulness(self, used_contents: List[str], delta: float = 0.1):
        """
        Called by the pipeline after an LLM response is generated.

        For every memory whose content appears in the response:
          usefulness = min(1.0, usefulness + delta)

        For every memory that was retrieved but NOT used:
          usefulness = max(0.1, usefulness - delta * 0.5)

        Parameters
        ----------
        used_contents : list of 'content' strings that were actually in the response
        delta         : how much to nudge usefulness per call (default 0.1)
        """
        if not used_contents:
            return

        used_set = set(c.strip().lower() for c in used_contents)
        keys     = self.redis.keys("mem:*")

        pipe = self.redis.pipeline()
        for key in keys:
            data = self.redis.hgetall(key)
            if b"content" not in data:
                continue
            content     = data[b"content"].decode().strip().lower()
            usefulness  = float(data.get(b"usefulness", b"0.5"))
            if content in used_set:
                new_u = min(1.0, usefulness + delta)
            else:
                new_u = max(0.1, usefulness - delta * 0.5)
            pipe.hset(key, "usefulness", new_u)
        pipe.execute()

    # ── TTL expiry ────────────────────────────────────────────────────────────

    def expire_stale(
        self,
        ttl_days:         float = None,
        usefulness_floor: float = None,
    ) -> Dict:
        """
        Delete entries that are both old AND low-usefulness.

        An entry is expired when:
          age_days >= ttl_days  AND  usefulness < usefulness_floor

        Returns { "scanned": int, "expired": int }
        """
        ttl_days         = ttl_days         or MEMORY_TTL_DAYS
        usefulness_floor = usefulness_floor or MEMORY_TTL_USEFULNESS
        now              = time.time()
        cutoff           = ttl_days * 86400

        keys    = self.redis.keys("mem:*")
        expired = 0
        pipe    = self.redis.pipeline()

        for key in keys:
            data = self.redis.hgetall(key)
            ts         = int(data.get(b"timestamp",  b"0"))
            usefulness = float(data.get(b"usefulness", b"0.5"))
            age        = now - ts
            if age >= cutoff and usefulness < usefulness_floor:
                pipe.delete(key)
                expired += 1

        pipe.execute()
        return {"scanned": len(keys), "expired": expired}

    # ── Full dedup sweep ──────────────────────────────────────────────────────

    def dedup_all(self, threshold: float = None) -> Dict:
        """
        Collapse all pairs above threshold. Keeper = higher access_count.
        Returns { scanned, merged, deleted, remaining }.
        """
        threshold = threshold if threshold is not None else MEMORY_MERGE_THRESHOLD
        keys      = self.redis.keys("mem:*")
        entries   = []
        for key in keys:
            data = self.redis.hgetall(key)
            if b"embedding" not in data:
                continue
            entries.append({
                "key":          key,
                "vec":          self._deserialize_vector(data[b"embedding"]),
                "content":      data[b"content"].decode(),
                "access_count": int(data.get(b"access_count", b"0")),
            })

        to_delete = set()
        merged    = 0

        for i in range(len(entries)):
            if entries[i]["key"] in to_delete:
                continue
            for j in range(i + 1, len(entries)):
                if entries[j]["key"] in to_delete:
                    continue
                sim = self._cosine_similarity(entries[i]["vec"], entries[j]["vec"])
                if sim < threshold:
                    continue

                keep, drop = (i, j) if entries[i]["access_count"] >= entries[j]["access_count"] else (j, i)

                merged_vec = (entries[keep]["vec"] + entries[drop]["vec"]) / 2.0
                norm = np.linalg.norm(merged_vec)
                if norm > 0:
                    merged_vec /= norm
                entries[keep]["vec"] = merged_vec

                keep_content = (
                    entries[keep]["content"]
                    if len(entries[keep]["content"]) >= len(entries[drop]["content"])
                    else entries[drop]["content"]
                )
                entries[keep]["content"]      = keep_content
                entries[keep]["access_count"] += entries[drop]["access_count"]

                pipe = self.redis.pipeline()
                pipe.hset(entries[keep]["key"], "embedding",    self._serialize_vector(merged_vec))
                pipe.hset(entries[keep]["key"], "content",      keep_content)
                pipe.hset(entries[keep]["key"], "access_count", entries[keep]["access_count"])
                pipe.execute()

                to_delete.add(entries[drop]["key"])
                merged += 1

        if to_delete:
            pipe = self.redis.pipeline()
            for key in to_delete:
                pipe.delete(key)
            pipe.execute()

        return {
            "scanned":   len(entries),
            "merged":    merged,
            "deleted":   len(to_delete),
            "remaining": len(entries) - len(to_delete),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Global Instance & Module-level Helpers
# ═════════════════════════════════════════════════════════════════════════════

_store = RedisMemoryStore()


def retrieve_memory(query_vec: np.ndarray) -> List[Dict]:
    return _store.retrieve_all(query_vec)


def remember(source: str, content: str):
    vec = embed(content)
    _store.ingest_memory(content, vec, cluster_id=0, token_len=len(content) // 4, source=source)


def remember_exchange(user_content: str, assistant_content: str):
    """Store a full Q+A turn as one deduplicated memory unit."""
    _store.ingest_exchange(user_content, assistant_content)


def update_memory_usefulness(used_contents: List[str], delta: float = 0.1):
    """
    Signal which retrieved memories were actually useful in the last response.
    Call this from pipeline.py after the LLM call.
    """
    _store.update_usefulness(used_contents, delta)


def dedup_memory(threshold: float = None) -> Dict:
    """Run a full deduplication sweep. Returns { scanned, merged, deleted, remaining }."""
    return _store.dedup_all(threshold)


def expire_stale_memory(ttl_days: float = None, usefulness_floor: float = None) -> Dict:
    """Delete old, low-usefulness entries. Returns { scanned, expired }."""
    return _store.expire_stale(ttl_days, usefulness_floor)
