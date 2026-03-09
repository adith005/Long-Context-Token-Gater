"""
Document Store — Redis + JSON
==============================

Pipeline (strict order):
  1. Load      : read file(s) — .pdf / .txt / .md
  2. Persist   : save full sentences as a JSON file on disk
  3. Summarise : build an extractive summary (first 5 sentences)
  4. Registry  : store { filename, summary, json_path, sentence_count }
                 in Redis under  doc:registry:<doc_name>
  5. Embed     : encode sentences with all-MiniLM-L6-v2
  6. Chunk     : split embedding matrix into fixed-size chunks
  7. Store     : push chunks into Redis  <doc_name>:chunk:N
  8. Meta      : store cluster metadata  <doc_name>:meta
  9. Retrieve  : semantic similarity search

Redis layout
────────────
  doc:registry:<doc_name>   →  JSON  { doc_name, original_filename,
                                       summary, json_path, sentence_count,
                                       ingested_at }
  <doc_name>:meta           →  JSON  { created_at, access_count, … }
  <doc_name>:chunk:0        →  JSON  { sentences: […], vectors: [[…], …] }
  …

Disk layout
───────────
  doc_storage/json_store/<doc_name>.json
      { doc_name, original_filename, ingested_at, sentences: […] }
"""

import os
import re
import json
import datetime
from typing import Optional

import numpy as np
import redis
from pypdf import PdfReader

from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB, DOC_JSON_PATH
from utils.embedding import embed, MODEL_NAME

# ── Config ─────────────────────────────────────────────────────────────────────
SENTENCES_PER_CHUNK  = 10
SUMMARY_SENTENCES    = 5      # how many sentences to use for the extractive summary

# ── Singletons ─────────────────────────────────────────────────────────────────
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
os.makedirs(DOC_JSON_PATH, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ═══════════════════════════════════════════════════════════════════════════════

def load_documents(storage_path: str) -> dict[str, list[str]]:
    """
    Load all supported files from a directory (or a single file path).

    Returns
    -------
    { doc_name: [sentence1, sentence2, …] }

    Supported formats: .pdf, .txt, .md
    The doc_name is the filename without extension.
    """
    if os.path.isfile(storage_path):
        paths = [storage_path]
    elif os.path.isdir(storage_path):
        paths = [
            os.path.join(storage_path, f)
            for f in os.listdir(storage_path)
            if f.lower().endswith((".pdf", ".txt", ".md"))
        ]
    else:
        raise FileNotFoundError(f"Path not found: {storage_path}")

    documents = {}
    for path in paths:
        doc_name = os.path.splitext(os.path.basename(path))[0]
        text     = _read_file(path)
        sentences = _split_sentences(text)
        if sentences:
            documents[doc_name] = sentences
            print(f"[load] '{doc_name}'  →  {len(sentences)} sentences")
        else:
            print(f"[load] Skipped '{doc_name}' (no extractable text)")

    return documents


def _read_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    else:                             # .txt / .md
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def _split_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter — splits on '.', '!', '?'
    Filters out blank lines and very short fragments.
    """
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — PERSIST JSON TO DISK
# ═══════════════════════════════════════════════════════════════════════════════

def persist_json(doc_name: str, original_filename: str, sentences: list[str]) -> str:
    """
    Save the full sentence list as a JSON file on disk.
    Returns the file path so it can be stored in the registry.
    """
    payload = {
        "doc_name":          doc_name,
        "original_filename": original_filename,
        "ingested_at":       _now(),
        "sentence_count":    len(sentences),
        "sentences":         sentences,
    }
    path = os.path.join(DOC_JSON_PATH, f"{doc_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[json]     '{doc_name}'  →  {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — EXTRACTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def build_summary(sentences: list[str], n: int = SUMMARY_SENTENCES) -> str:
    """
    Lightweight extractive summary — first N non-trivial sentences joined.
    No LLM required; keeps the registry fast and lightweight.
    """
    picked = [s for s in sentences if len(s.split()) >= 6][:n]
    return " ".join(picked)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — REDIS REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

def register_document(
    doc_name:          str,
    original_filename: str,
    summary:           str,
    json_path:         str,
    sentence_count:    int,
) -> None:
    """
    Store a human-readable registry entry in Redis.

    Key:   doc:registry:<doc_name>
    Value: JSON { doc_name, original_filename, summary,
                  json_path, sentence_count, ingested_at }

    Separate from <doc_name>:meta (chunk-level index).
    Browse with list_registry().
    """
    entry = {
        "doc_name":          doc_name,
        "original_filename": original_filename,
        "summary":           summary,
        "json_path":         json_path,
        "sentence_count":    sentence_count,
        "ingested_at":       _now(),
    }
    r.set(f"doc:registry:{doc_name}", json.dumps(entry))
    print(f"[registry] '{doc_name}'  →  registered in Redis")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — EMBED
# ═══════════════════════════════════════════════════════════════════════════════

def embed_sentences(sentences: list[str]) -> np.ndarray:
    """
    Embed a list of sentences.
    """
    print(f"[embed] Encoding {len(sentences)} sentences …")
    vectors = []
    for s in sentences:
        vectors.append(embed(s))
    return np.array(vectors).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 + 4 — CHUNK & STORE IN REDIS
# ═══════════════════════════════════════════════════════════════════════════════

def store_document(
    doc_name: str,
    sentences: list[str],
    vectors: np.ndarray,
) -> None:
    """
    Chunk the embedding matrix and store everything under the cluster
    named after the document.

    Redis keys
    ----------
    <doc_name>:meta       — cluster metadata  (Step 5)
    <doc_name>:chunk:0    — first N sentences + their vectors
    <doc_name>:chunk:1    — next N sentences + their vectors
    …
    """
    n           = len(sentences)
    chunk_size  = SENTENCES_PER_CHUNK
    num_chunks  = (n + chunk_size - 1) // chunk_size  # ceiling division

    pipe = r.pipeline()

    # ── Step 5: Metadata ──────────────────────────────────────────────────────
    meta = {
        "doc_name"      : doc_name,
        "created_at"    : _now(),
        "access_count"  : 0,
        "last_accessed" : None,
        "chunk_count"   : num_chunks,
        "total_sentences": n,
        "model"         : MODEL_NAME,
    }
    pipe.set(f"{doc_name}:meta", json.dumps(meta))

    # ── Chunk loop ────────────────────────────────────────────────────────────
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end   = min(start + chunk_size, n)

        chunk_sentences = sentences[start:end]
        chunk_vectors   = vectors[start:end].tolist()   # list of lists (JSON-serialisable)

        payload = {
            "sentences": chunk_sentences,
            "vectors"  : chunk_vectors,
        }
        pipe.set(f"{doc_name}:chunk:{chunk_idx}", json.dumps(payload))

    pipe.execute()
    print(f"[store] '{doc_name}'  →  {num_chunks} chunks stored in Redis cluster '{doc_name}:*'")


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PIPELINE (convenience wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

def ingest(storage_path: str) -> list[str]:
    """
    Run the full pipeline for every document found at storage_path.

    Steps per document
    ------------------
    1. Load sentences from file
    2. Persist sentences as JSON to disk  (doc_storage/json_store/<name>.json)
    3. Build extractive summary
    4. Register { filename, summary, json_path } in Redis  (doc:registry:<name>)
    5. Embed sentences
    6. Chunk + store embeddings in Redis  (<name>:chunk:N, <name>:meta)

    Returns list of doc_names stored.
    """
    documents = load_documents(storage_path)
    stored    = []

    for doc_name, sentences in documents.items():
        original_filename = os.path.basename(storage_path)

        # Steps 2–4: persist, summarise, register
        json_path = persist_json(doc_name, original_filename, sentences)
        summary   = build_summary(sentences)
        register_document(doc_name, original_filename, summary, json_path, len(sentences))

        # Steps 5–6: embed + chunk store
        vectors = embed_sentences(sentences)
        store_document(doc_name, sentences, vectors)

        stored.append(doc_name)
        print()

    return stored


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — RETRIEVE (semantic similarity search)
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve(
    query: str,
    doc_name: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Embed the user query and return the top-k most similar sentences
    from the specified document cluster.

    Parameters
    ----------
    query    : free-text question or search string
    doc_name : name of the stored document cluster
    top_k    : number of results to return

    Returns
    -------
    List of dicts sorted by similarity (highest first):
        { rank, score, sentence, chunk_index, sentence_index }
    """
    # Check document exists
    meta_raw = r.get(f"{doc_name}:meta")
    if not meta_raw:
        print(f"[retrieve] No cluster found for '{doc_name}'.")
        return []

    meta        = json.loads(meta_raw)
    num_chunks  = meta["chunk_count"]

    # Embed query  (same model, same normalisation)
    q_vec = embed(query)

    # Load all chunks and score every sentence
    all_scores = []   # (global_sentence_idx, chunk_idx, local_idx, score, sentence)

    pipe = r.pipeline()
    for i in range(num_chunks):
        pipe.get(f"{doc_name}:chunk:{i}")
    raw_chunks = pipe.execute()

    global_idx = 0
    for chunk_idx, raw in enumerate(raw_chunks):
        if raw is None:
            continue
        payload   = json.loads(raw)
        sentences = payload["sentences"]
        vectors   = payload["vectors"]

        for local_idx, (sentence, vec_list) in enumerate(zip(sentences, vectors)):
            vec   = np.array(vec_list, dtype=np.float32)
            score = float(np.dot(q_vec, vec))           # cosine sim (unit vecs)
            all_scores.append((global_idx, chunk_idx, local_idx, score, sentence))
            global_idx += 1

    # Sort by score descending
    all_scores.sort(key=lambda x: x[3], reverse=True)
    top = all_scores[:top_k]

    # Update access metadata
    meta["access_count"]  += 1
    meta["last_accessed"]  = _now()
    r.set(f"{doc_name}:meta", json.dumps(meta))

    # Format results
    results = [
        {
            "rank"            : rank + 1,
            "score"           : round(score, 4),
            "sentence"        : sentence,
            "chunk_index"     : chunk_idx,
            "sentence_index"  : global_idx,
        }
        for rank, (global_idx, chunk_idx, _, score, sentence) in enumerate(top)
    ]

    return results


def retrieve_all(query: str, top_k: int = 5) -> list[dict]:
    """
    Search across ALL stored document clusters.

    Returns the globally top-k sentences with their source doc_name.
    """
    # Find all clusters via meta keys
    all_meta_keys = r.keys("*:meta")
    doc_names     = [k.replace(":meta", "") for k in all_meta_keys]

    q_vec = embed(query)

    all_scores = []

    for doc_name in doc_names:
        meta_raw = r.get(f"{doc_name}:meta")
        if not meta_raw:
            continue
        meta       = json.loads(meta_raw)
        num_chunks = meta["chunk_count"]

        pipe = r.pipeline()
        for i in range(num_chunks):
            pipe.get(f"{doc_name}:chunk:{i}")
        raw_chunks = pipe.execute()

        for chunk_idx, raw in enumerate(raw_chunks):
            if raw is None:
                continue
            payload = json.loads(raw)
            for local_idx, (sentence, vec_list) in enumerate(
                zip(payload["sentences"], payload["vectors"])
            ):
                vec   = np.array(vec_list, dtype=np.float32)
                score = float(np.dot(q_vec, vec))
                all_scores.append((doc_name, chunk_idx, local_idx, score, sentence))

    all_scores.sort(key=lambda x: x[3], reverse=True)

    results = [
        {
            "rank"        : i + 1,
            "doc_name"    : doc_name,
            "chunk_index" : chunk_idx,
            "score"       : round(score, 4),
            "sentence"    : sentence,
        }
        for i, (doc_name, chunk_idx, _, score, sentence) in enumerate(all_scores[:top_k])
    ]

    # Update access counts
    seen = set()
    for doc_name, *_ in all_scores[:top_k]:
        if doc_name not in seen:
            meta_raw = r.get(f"{doc_name}:meta")
            if meta_raw:
                meta = json.loads(meta_raw)
                meta["access_count"] += 1
                meta["last_accessed"] = _now()
                r.set(f"{doc_name}:meta", json.dumps(meta))
            seen.add(doc_name)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════════════

def get_meta(doc_name: str) -> Optional[dict]:
    raw = r.get(f"{doc_name}:meta")
    return json.loads(raw) if raw else None


def list_documents() -> list[dict]:
    return [
        json.loads(r.get(key))
        for key in r.keys("*:meta")
        if r.get(key)
    ]


def list_registry() -> list[dict]:
    """
    Return all registry entries — one per ingested document.
    Each entry has: doc_name, original_filename, summary, json_path,
                    sentence_count, ingested_at.
    """
    keys = r.keys("doc:registry:*")
    entries = []
    for key in keys:
        raw = r.get(key)
        if raw:
            entries.append(json.loads(raw))
    return sorted(entries, key=lambda x: x.get("ingested_at", ""), reverse=True)


def get_registry_entry(doc_name: str) -> Optional[dict]:
    """Return the registry entry for a single document, or None if not found."""
    raw = r.get(f"doc:registry:{doc_name}")
    return json.loads(raw) if raw else None


def load_json_store(doc_name: str) -> Optional[dict]:
    """
    Load the persisted JSON file for a document from disk.
    Returns the full payload including all sentences, or None if file missing.
    """
    path = os.path.join(DOC_JSON_PATH, f"{doc_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _now() -> str:
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ═══════════════════════════════════════════════════════════════════════════════
# CLI / DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:  python document_store.py <path>  [query]")
        print("        path  = file or folder containing .pdf / .txt / .md files")
        print("        query = optional search query (default: 'main topic')")
        sys.exit(1)

    storage_path = sys.argv[1]
    query        = sys.argv[2] if len(sys.argv) > 2 else "What is the main topic?"

    # ── Ingest ────────────────────────────────────────────────────────────────
    stored_docs = ingest(storage_path)

    # ── Inspect metadata ──────────────────────────────────────────────────────
    for name in stored_docs:
        print(json.dumps(get_meta(name), indent=2))

    # ── Retrieve: single doc ──────────────────────────────────────────────────
    if stored_docs:
        target  = stored_docs[0]
        results = retrieve(query, target, top_k=3)
        print(f'\n── Top results from "{target}" for: "{query}" ──')
        for res in results:
            print(f"  #{res['rank']}  score={res['score']}  |  {res['sentence'][:120]}")

    # ── Retrieve: all docs ────────────────────────────────────────────────────
    print(f'\n── Global search: "{query}" ──')
    for res in retrieve_all(query, top_k=5):
        print(f"  #{res['rank']}  [{res['doc_name']}]  score={res['score']}  |  {res['sentence'][:100]}")