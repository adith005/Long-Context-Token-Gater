import os, json, numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import EMBED_MODEL, DOC_STORE_PATH

model = SentenceTransformer(EMBED_MODEL)

def retrieve(query, k=5):
    q_emb = model.encode(query)
    results = []
    for file in os.listdir(DOC_STORE_PATH):
        data = json.load(open(f"{DOC_STORE_PATH}/{file}"))
        for chunk, emb in zip(data["chunks"], data["embeddings"]):
            emb = np.array(emb)
            sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb))
            results.append((chunk, sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in results[:k]]
