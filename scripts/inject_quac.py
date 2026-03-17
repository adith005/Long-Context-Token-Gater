import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from memory_storage.storage import _store
from utils.embedding import embed
import uuid, time, random

SEED = 42
random.seed(SEED)

def load_quac(path: str) -> list:
    with open(path) as f:
        raw = json.load(f)

    pairs = []
    for article in raw["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                question = qa["question"].strip()
                # skip unanswerable
                if not qa["answers"]:
                    continue
                answer = qa["answers"][0]["text"].strip()
                if answer == "CANNOTANSWER" or len(answer) < 5:
                    continue
                pairs.append({
                    "user":      question,
                    "assistant": answer,
                    "topic":     article.get("title", "unknown"),
                })
    return pairs


def inject_quac(path: str, max_entries: int = None, flush: bool = False):
    if flush:
        keys = _store.redis.keys("mem:*")
        if keys:
            _store.redis.delete(*keys)
            print(f"Flushed {len(keys)} existing keys.")

    pairs = load_quac(path)
    random.shuffle(pairs)
    if max_entries is not None:
        pairs = pairs[:max_entries]

    print(f"Injecting {len(pairs)} QuAC pairs...")

    for i, pair in enumerate(pairs):
        content   = f"Q: {pair['user']}\nA: {pair['assistant']}"
        vec       = embed(content)
        timestamp = int(time.time()) - random.randint(0, 60 * 86400)

        entry = {
            "content":           content,
            "user_content":      pair["user"],
            "assistant_content": pair["assistant"],
            "source":            "exchange",
            "embedding":         _store._serialize_vector(vec),
            "cluster_id":        pair["topic"],
            "token_len":         str(len(content.split())),
            "timestamp":         str(timestamp),
            "usefulness":        str(round(random.uniform(0.1, 1.0), 3)),
            "access_count":      str(random.randint(1, 20)),
        }

        _store.redis.hset(f"mem:quac:{uuid.uuid4().hex}", mapping=entry)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pairs)} done...")

    print(f"Done. Total injected: {len(pairs)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",  required=True, help="Path to QuAC JSON file")
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--flush", action="store_true")
    args = parser.parse_args()

    inject_quac(args.file, args.max, args.flush)