REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DOC_STORE_PATH  = "rag_storage"
DOC_JSON_PATH   = "doc_storage/json_store"    # persisted JSON files for uploaded docs


# Memory deduplication thresholds
MEMORY_EXACT_THRESHOLD  = 0.97   # cosine sim above this → skip (exact duplicate)
MEMORY_MERGE_THRESHOLD  = 0.85   # cosine sim above this → merge (near-duplicate)

# Retrieval ranking weights (should sum to 1.0)
RETRIEVAL_SIM_WEIGHT     = 0.50  # cosine similarity
RETRIEVAL_ACCESS_WEIGHT  = 0.15  # access frequency
RETRIEVAL_RECENCY_WEIGHT = 0.20  # recency decay
RETRIEVAL_SOURCE_WEIGHT  = 0.15  # source authority

# Recency decay — λ in e^(-λ × days).  0.1 → half-life ≈ 7 days
RECENCY_DECAY_LAMBDA     = 0.1

# Min cosine similarity floor — entries below this never enter context
MIN_SIM_FLOOR            = 0.30

# TTL expiry — entries older than this with low usefulness are deleted
MEMORY_TTL_DAYS          = 30
MEMORY_TTL_USEFULNESS    = 0.3

LLM_PROVIDER = "lmstudio"
LLM_MODEL    = "lfm2.5-1.2b-thinking"

