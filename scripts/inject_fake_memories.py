"""
scripts/inject_fake_memories.py
=================================

Injects synthetic Q&A memory entries directly into Redis.

Can load from:
  (a) --file path/to/synthetic_qa.json   ← preferred, use generate_synthetic_qa.py first
  (b) built-in QA pool                   ← fallback if no file given

Usage
-----
  # Load from JSON file (recommended)
  python scripts/inject_fake_memories.py --file data/synthetic_qa.json --flush

  # Use built-in pool, inject 500
  python scripts/inject_fake_memories.py --count 500 --flush

  # Dry run
  python scripts/inject_fake_memories.py --file data/synthetic_qa.json --dry-run

  # Full 7500 from file, wipe first
  python scripts/inject_fake_memories.py --file data/synthetic_qa.json --flush

Requirements
------------
  Redis running: docker run -d --name redis -p 6379:6379 redis:7
"""

import argparse
import json
import os
import sys
import random
import time
import uuid

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.embedding import embed
from memory_storage.storage import _store

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# BUILT-IN QA POOL  (fallback when no --file given)
# =============================================================================

QA_POOL = [
    ("What is the difference between supervised and unsupervised learning?",
     "Supervised learning uses labelled data to train a model to predict outputs. Unsupervised learning finds patterns in unlabelled data without predefined targets."),
    ("How does backpropagation work?",
     "Backpropagation computes the gradient of the loss with respect to each weight by applying the chain rule layer by layer from output to input."),
    ("What is overfitting and how do you prevent it?",
     "Overfitting is when a model learns noise in training data and performs poorly on new data. Prevention methods include dropout, regularisation, early stopping, and data augmentation."),
    ("Explain the transformer architecture.",
     "Transformers use self-attention mechanisms to weigh the importance of each token relative to all others, consisting of encoder and decoder stacks with multi-head attention."),
    ("What is the vanishing gradient problem?",
     "In deep networks, gradients shrink exponentially as they propagate back through layers, making early layers learn very slowly. Solutions include ReLU activations and residual connections."),
    ("What is a language model?",
     "A language model assigns probabilities to sequences of tokens. Modern LLMs are trained to predict the next token given previous context using vast amounts of text data."),
    ("What is fine-tuning in the context of LLMs?",
     "Fine-tuning adapts a pre-trained model to a specific task by continuing training on a smaller, task-specific dataset with a lower learning rate."),
    ("What is RAG — retrieval augmented generation?",
     "RAG combines a retrieval system with a language model. Relevant documents are fetched and injected into the prompt as context before the LLM generates a response."),
    ("What is Shannon entropy?",
     "Shannon entropy H = -Σ p_i log2(p_i) measures the average information content of a probability distribution. Higher entropy means more uncertainty and diversity."),
    ("How does cosine similarity work?",
     "Cosine similarity measures the angle between two vectors, equal to the dot product divided by the product of their magnitudes, returning a value between -1 and 1."),
    ("What is a Pixhawk flight controller?",
     "Pixhawk is an open-source autopilot hardware platform for drones. It runs ArduPilot or PX4 firmware and supports GPS, IMU, and communication modules."),
    ("What is PID control in drones?",
     "PID control adjusts motor outputs based on proportional, integral, and derivative terms of orientation error, maintaining drone stability."),
    ("What is LoRa communication?",
     "LoRa is a long-range, low-power wireless modulation technique providing kilometre-range telemetry links at low data rates, used for drone control."),
    ("What is a REST API?",
     "A REST API uses HTTP methods like GET, POST, PUT, DELETE to expose resources. It is stateless, meaning each request contains all information needed to process it."),
    ("Explain Docker containers.",
     "Docker packages applications and dependencies into containers that run consistently across environments, sharing the host OS kernel, lighter than VMs."),
    ("What is the speed of light?",
     "The speed of light in a vacuum is approximately 299,792,458 metres per second. It is denoted c and is the universal speed limit."),
    ("What is quantum entanglement?",
     "Quantum entanglement is a phenomenon where two particles become correlated such that measuring one instantly determines the state of the other regardless of distance."),
    ("What is DNA replication?",
     "DNA replication unwinds the double helix and uses each strand as a template to synthesise a complementary strand, resulting in two identical DNA molecules."),
    ("What is compound interest?",
     "Compound interest calculates interest on both the principal and accumulated interest: A = P(1+r/n)^(nt), causing exponential growth over time."),
    ("What is the difference between TCP and UDP?",
     "TCP is connection-oriented and guarantees delivery and order. UDP is connectionless and faster but does not guarantee delivery, used for streaming and gaming."),
]

NEAR_DUPLICATE_VARIANTS = [
    ("Can you explain what backpropagation does?",
     "Backpropagation applies the chain rule to compute gradients, propagating error from output back through each layer to update weights."),
    ("How does back propagation work in neural networks?",
     "During backprop, the error signal flows backward from the loss through each layer, accumulating gradients for weight updates via gradient descent."),
    ("What does the Pixhawk flight controller do?",
     "Pixhawk is an autopilot hardware board for drones running ArduPilot or PX4, handling stabilisation, GPS navigation, and sensor interfaces."),
    ("Tell me about LoRa wireless technology.",
     "LoRa provides long-range wireless communication at low power and low data rate, popular for IoT and drone telemetry requiring km-range links."),
    ("How is cosine similarity calculated?",
     "Cosine similarity is the dot product of two normalised vectors, measuring directional similarity and returning 1 for identical direction."),
]


# =============================================================================
# USEFULNESS / TIMESTAMP / ACCESS SAMPLING
# =============================================================================

def sample_usefulness() -> float:
    tier = random.random()
    if tier < 0.15:
        return round(random.uniform(0.01, 0.09), 3)
    elif tier < 0.40:
        return round(random.uniform(0.10, 0.39), 3)
    elif tier < 0.80:
        return round(random.uniform(0.40, 0.70), 3)
    else:
        return round(random.uniform(0.71, 1.00), 3)


def sample_timestamp(days_back_max: int = 60) -> int:
    return int(time.time()) - random.randint(0, days_back_max * 86400)


def sample_access_count() -> int:
    return max(1, int(np.random.exponential(scale=5)) + 1)


# =============================================================================
# INJECTION
# =============================================================================

def build_memory_entry(user_content, assistant_content, usefulness, timestamp, access_count):
    content = f"Q: {user_content}\nA: {assistant_content}"
    vec     = embed(content)
    return {
        "content":           content,
        "user_content":      user_content,
        "assistant_content": assistant_content,
        "source":            "exchange",
        "embedding":         _store._serialize_vector(vec),
        "cluster_id":        "injected",
        "token_len":         str(len(content.split())),
        "timestamp":         str(timestamp),
        "usefulness":        str(usefulness),
        "access_count":      str(access_count),
    }


def load_qa_from_file(path: str) -> list:
    """
    Load Q&A pairs from a JSON file.

    Supports two formats:
      (a) generate_synthetic_qa.py output:
            [ {"user": "...", "assistant": "...", "topic": "..."}, ... ]
      (b) inject_quac.py style:
            [ {"user": "...", "assistant": "...", "topic": "..."}, ... ]

    Both are the same shape so one loader handles both.
    """
    with open(path) as f:
        data = json.load(f)

    pairs = []
    for item in data:
        u = item.get("user") or item.get("question") or ""
        a = item.get("assistant") or item.get("answer") or ""
        if u.strip() and a.strip():
            pairs.append((u.strip(), a.strip()))

    return pairs


def inject_memories(
    count:     int  = None,
    file_path: str  = None,
    dry_run:   bool = False,
    flush:     bool = False,
    verbose:   bool = True,
) -> dict:
    """
    Inject memories from a JSON file or the built-in QA pool.

    Parameters
    ----------
    count     : how many to inject. None = all entries in the file.
    file_path : path to synthetic_qa.json. None = use built-in QA_POOL.
    dry_run   : print without writing to Redis.
    flush     : wipe existing mem:* keys first.
    verbose   : print progress.
    """
    if flush and not dry_run:
        existing = _store.redis.keys("mem:*")
        if existing:
            _store.redis.delete(*existing)
            if verbose:
                print(f"  Flushed {len(existing)} existing memory keys.")

    # ── Load source QA ────────────────────────────────────────────────────────
    if file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        all_qa = load_qa_from_file(file_path)
        if verbose:
            print(f"  Loaded {len(all_qa)} pairs from {file_path}")
    else:
        # Built-in pool — repeat and shuffle to hit count
        pool   = list(QA_POOL)
        target = count or 500
        all_qa = (pool * (target // len(pool) + 2))[:target - len(NEAR_DUPLICATE_VARIANTS)]
        random.shuffle(all_qa)
        all_qa += NEAR_DUPLICATE_VARIANTS
        if verbose:
            print(f"  Using built-in QA pool ({len(QA_POOL)} base pairs).")

    # ── Apply count cap ───────────────────────────────────────────────────────
    if count is not None:
        random.shuffle(all_qa)
        all_qa = all_qa[:count]

    total = len(all_qa)

    injected         = 0
    usefulness_dist  = {"near_zero": 0, "low": 0, "medium": 0, "high": 0}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Memory Injector")
        print(f"  Source     : {file_path or 'built-in pool'}")
        print(f"  Injecting  : {total} entries")
        print(f"  Dry run    : {dry_run}")
        print(f"{'='*60}\n")

    for i, pair in enumerate(all_qa):
        # Handle both tuple (built-in) and dict (file) formats
        if isinstance(pair, tuple):
            user_q, asst_a = pair
        else:
            user_q, asst_a = pair["user"], pair["assistant"]

        usefulness   = sample_usefulness()
        timestamp    = sample_timestamp(days_back_max=60)
        access_count = sample_access_count()

        if usefulness < 0.10:
            usefulness_dist["near_zero"] += 1
        elif usefulness < 0.40:
            usefulness_dist["low"] += 1
        elif usefulness < 0.71:
            usefulness_dist["medium"] += 1
        else:
            usefulness_dist["high"] += 1

        if dry_run:
            print(f"  [{i+1:04d}] use={usefulness:.3f}  acc={access_count:3d}  "
                  f"Q: {user_q[:70]}")
            injected += 1
            continue

        entry = build_memory_entry(user_q, asst_a, usefulness, timestamp, access_count)
        _store.redis.hset(f"mem:injected:{uuid.uuid4().hex}", mapping=entry)
        injected += 1

        if verbose and (i + 1) % 500 == 0:
            print(f"  Injected {i+1}/{total}...")

    total_keys = len(_store.redis.keys("mem:*")) if not dry_run else total

    if verbose:
        print(f"\n{'='*60}")
        print(f"  INJECTION COMPLETE")
        print(f"{'='*60}")
        print(f"  Injected  : {injected}")
        print(f"  Total mem:* keys in Redis: {total_keys}")
        print(f"\n  Usefulness distribution:")
        for tier, n in usefulness_dist.items():
            pct = n / injected * 100 if injected else 0
            print(f"    {tier:<12} : {n:5d}  ({pct:.1f}%)")
        print(f"\n  Next steps:")
        print(f"    python needlebench.py --seed 42 --output results.json")
        print(f"    python scripts/mass_query_runner.py --output paper_results.json")
        print(f"{'='*60}\n")

    return {
        "injected":         injected,
        "total_redis_keys": total_keys,
        "usefulness_dist":  usefulness_dist,
        "source":           file_path or "built-in",
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject fake memories into Redis")
    parser.add_argument("--file",    type=str,  default=None,
                        help="Path to synthetic_qa.json (from generate_synthetic_qa.py). "
                             "If not given, uses the built-in QA pool.")
    parser.add_argument("--count",   type=int,  default=None,
                        help="Max entries to inject. Default: all entries in file, "
                             "or 500 for built-in pool.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print entries without writing to Redis.")
    parser.add_argument("--flush",   action="store_true",
                        help="Wipe existing mem:* keys before injecting.")
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    inject_memories(
        count     = args.count,
        file_path = args.file,
        dry_run   = args.dry_run,
        flush     = args.flush,
        verbose   = not args.quiet,
    )