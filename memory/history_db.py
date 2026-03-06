import json
from redis import Redis
from datetime import datetime
from uuid import uuid4
from sentence_transformers import SentenceTransformer
import numpy as np

from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB, EMBED_MODEL

r = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
model = SentenceTransformer(EMBED_MODEL)

def create_entry(query, response, tags=None):
    eid = uuid4().hex
    entry = {
        "id": eid,
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "response": response,
        "tags": tags or []
    }
    r.set(f"memory:{eid}", json.dumps(entry))
    emb = model.encode(query + " " + response).tolist()
    r.set(f"embedding:{eid}", json.dumps(emb))
    return eid

def retrieve_similar(query, k=5):
    q_emb = model.encode(query)
    results = []
    for key in r.keys("embedding:*"):
        eid = key.split(":")[1]
        emb = np.array(json.loads(r.get(key)))
        sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
        entry = json.loads(r.get(f"memory:{eid}"))
        results.append((entry, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return [e for e, _ in results[:k]]
