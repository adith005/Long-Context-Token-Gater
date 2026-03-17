"""
scripts/inject_fake_memories.py
=================================

Injects 500 synthetic Q&A memory entries directly into Redis
to simulate a long conversation history.

Stress test focus: USEFULNESS SCORE VARIANCE
  - ~20% entries: high usefulness (0.8–1.0)  — frequently helpful memories
  - ~40% entries: medium usefulness (0.4–0.7) — occasionally useful
  - ~25% entries: low usefulness (0.1–0.39)  — rarely surfaced in responses
  - ~15% entries: near-zero usefulness (0.01–0.09) — candidates for expiry

Additional variance:
  - Timestamps spread over past 60 days (tests recency decay)
  - Access counts vary 1–50 (tests access weight)
  - 30 deliberate near-duplicate pairs (tests dedup on future writes)
  - Mixed topics: AI/ML, drones, coding, science, history, health, finance

Usage
-----
  # From project root with venv active:
  python scripts/inject_fake_memories.py

  # Dry run (no Redis writes, just print what would be injected):
  python scripts/inject_fake_memories.py --dry-run

  # Custom count:
  python scripts/inject_fake_memories.py --count 200

  # Wipe existing memories first:
  python scripts/inject_fake_memories.py --flush

Requirements
------------
  Redis must be running: docker run -d --name redis -p 6379:6379 redis:7
"""

import argparse
import os
import sys
import random
import time
import uuid

import numpy as np

# ── Path fix ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.embedding import embed
from memory_storage.storage import _store   # direct access to MemoryStore instance
from config.settings import (
    MEMORY_EXACT_THRESHOLD,
    MEMORY_MERGE_THRESHOLD,
)

# ── Seed for reproducibility ──────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =============================================================================
# SYNTHETIC Q&A POOL  —  mixed topics
# =============================================================================

QA_POOL = [

    # ── AI / ML ───────────────────────────────────────────────────────────────
    ("What is the difference between supervised and unsupervised learning?",
     "Supervised learning uses labelled data to train a model to predict outputs. Unsupervised learning finds patterns in unlabelled data without predefined targets."),
    ("How does backpropagation work?",
     "Backpropagation computes the gradient of the loss with respect to each weight by applying the chain rule layer by layer from output to input."),
    ("What is overfitting and how do you prevent it?",
     "Overfitting is when a model learns noise in training data and performs poorly on new data. Prevention methods include dropout, regularisation, early stopping, and data augmentation."),
    ("Explain the transformer architecture.",
     "Transformers use self-attention mechanisms to weigh the importance of each token relative to all others. They consist of encoder and decoder stacks with multi-head attention and feed-forward layers."),
    ("What is the vanishing gradient problem?",
     "In deep networks, gradients shrink exponentially as they propagate back through layers, making early layers learn very slowly. Solutions include ReLU activations, batch normalisation, and residual connections."),
    ("What is a language model?",
     "A language model assigns probabilities to sequences of tokens. Modern LLMs are trained to predict the next token given previous context using vast amounts of text data."),
    ("What is fine-tuning in the context of LLMs?",
     "Fine-tuning adapts a pre-trained model to a specific task or domain by continuing training on a smaller, task-specific dataset with a lower learning rate."),
    ("What is RAG — retrieval augmented generation?",
     "RAG combines a retrieval system with a language model. Relevant documents are fetched from a store and injected into the prompt as context before the LLM generates a response."),
    ("What is Shannon entropy?",
     "Shannon entropy H = -Σ p_i log2(p_i) measures the average information content of a probability distribution. Higher entropy means more uncertainty and diversity."),
    ("How does cosine similarity work?",
     "Cosine similarity measures the angle between two vectors. It equals the dot product divided by the product of their magnitudes, returning a value between -1 and 1."),
    ("What is sentence-transformers?",
     "sentence-transformers is a Python library for generating dense vector embeddings of sentences using pre-trained transformer models like all-MiniLM-L6-v2."),
    ("What is Redis and why use it for memory storage?",
     "Redis is an in-memory key-value store. It provides fast reads and writes, making it suitable for storing conversation memory and vector embeddings that need low-latency retrieval."),
    ("What is BM25?",
     "BM25 is a bag-of-words retrieval function that ranks documents by term frequency and inverse document frequency, with length normalisation. It is a strong IR baseline."),
    ("What is NDCG?",
     "NDCG — Normalised Discounted Cumulative Gain — measures ranking quality by giving more credit to relevant items appearing at higher positions in the result list."),
    ("What is the difference between precision and recall?",
     "Precision is the fraction of retrieved items that are relevant. Recall is the fraction of relevant items that were retrieved. They trade off against each other."),
    ("Explain attention mechanisms.",
     "Attention allows a model to focus on relevant parts of the input by computing a weighted sum of values, where weights are determined by similarity between queries and keys."),
    ("What is a vector database?",
     "A vector database stores high-dimensional embeddings and supports approximate nearest-neighbour search. Examples include Pinecone, Weaviate, and Chroma."),
    ("What is the purpose of softmax?",
     "Softmax converts a vector of raw scores into a probability distribution by exponentiating each value and dividing by the sum, ensuring all outputs are positive and sum to 1."),
    ("What is knowledge distillation?",
     "Knowledge distillation trains a smaller student model to mimic the outputs of a larger teacher model, transferring knowledge without requiring the student to be as large."),
    ("What is prompt engineering?",
     "Prompt engineering involves crafting input text to guide an LLM toward desired behaviour, including techniques like chain-of-thought, few-shot examples, and role specification."),

    # ── Drones & Robotics ─────────────────────────────────────────────────────
    ("What is a Pixhawk flight controller?",
     "Pixhawk is an open-source autopilot hardware platform used in drones. It runs ArduPilot or PX4 firmware and supports GPS, IMU, and communication modules."),
    ("What is PID control in drones?",
     "PID control adjusts motor outputs based on the error between desired and actual orientation. P corrects current error, I corrects accumulated error, D dampens oscillation."),
    ("How does a quadcopter achieve yaw rotation?",
     "Yaw is achieved by varying the speed of diagonal motor pairs. Since opposite motors spin in opposite directions, speeding one pair relative to the other creates net torque."),
    ("What is MAVLink?",
     "MAVLink is a lightweight messaging protocol for communicating with drones and ground stations. It defines packet structures for telemetry, commands, and status messages."),
    ("What is LoRa communication?",
     "LoRa is a long-range, low-power wireless modulation technique. It is used for telemetry links in drones where low data rate but long range is acceptable."),
    ("How does GPS work on a drone?",
     "The drone's GPS receiver triangulates its position using signals from at least four satellites. Position, velocity, and altitude data are fed to the flight controller."),
    ("What is a Raspberry Pi used for in a drone?",
     "A Raspberry Pi acts as a companion computer, handling tasks too computationally heavy for the flight controller such as computer vision, object detection, and high-level mission logic."),
    ("What is ArduPilot?",
     "ArduPilot is an open-source autopilot software platform supporting planes, copters, rovers, and boats. It runs on Pixhawk and supports autonomous missions via MAVLink."),
    ("What is obstacle avoidance in drone navigation?",
     "Obstacle avoidance uses sensors like lidar, ultrasonic, or depth cameras to detect objects in the flight path and reroute or halt the drone before collision."),
    ("What is geofencing for drones?",
     "Geofencing defines a virtual boundary within which a drone must stay. If the drone approaches or crosses the boundary, the flight controller can automatically return home."),

    # ── General Tech / Coding ────────────────────────────────────────────────
    ("What is the difference between TCP and UDP?",
     "TCP is connection-oriented and guarantees delivery and order. UDP is connectionless and faster but does not guarantee delivery or ordering, used for streaming and gaming."),
    ("What is a REST API?",
     "A REST API uses HTTP methods like GET, POST, PUT, DELETE to expose resources. It is stateless, meaning each request contains all information needed to process it."),
    ("Explain Docker containers.",
     "Docker packages applications and their dependencies into containers that run consistently across environments. Containers share the host OS kernel, making them lighter than VMs."),
    ("What is Git rebase vs merge?",
     "Merge creates a merge commit preserving branch history. Rebase rewrites commits on top of another branch, producing a linear history without merge commits."),
    ("What is an async function in Python?",
     "An async function is defined with async def and can use await to pause execution without blocking the thread, allowing other coroutines to run in the event loop."),
    ("What is a hash map?",
     "A hash map stores key-value pairs using a hash function to compute indices. It provides average O(1) lookup, insertion, and deletion."),
    ("What is time complexity?",
     "Time complexity measures how an algorithm's runtime scales with input size. O(n) grows linearly, O(log n) grows logarithmically, O(n²) grows quadratically."),
    ("What is a binary search tree?",
     "A BST stores nodes where each left child is smaller and each right child is larger than the parent. This allows O(log n) search, insertion, and deletion on balanced trees."),
    ("What is a neural network activation function?",
     "Activation functions introduce non-linearity. Common ones are ReLU (max(0,x)), sigmoid (1/(1+e^-x)), and tanh. Without them, networks reduce to linear models."),
    ("What is recursion?",
     "Recursion is a technique where a function calls itself with a smaller subproblem until a base case is reached. Examples include factorial, Fibonacci, and tree traversal."),
    ("How does HTTPS work?",
     "HTTPS encrypts HTTP traffic using TLS. The client and server perform a handshake to exchange certificates and negotiate a session key, then communicate symmetrically."),
    ("What is a virtual environment in Python?",
     "A virtual environment isolates Python packages per project, preventing version conflicts. Created with python -m venv and activated with source venv/bin/activate."),
    ("What is Streamlit?",
     "Streamlit is a Python library for building data apps and dashboards without writing HTML or JavaScript. Components are defined in plain Python and rendered in a browser."),
    ("What is JSON?",
     "JSON — JavaScript Object Notation — is a lightweight data interchange format using key-value pairs. It is human-readable and widely used for APIs and configuration files."),
    ("What is an API rate limit?",
     "A rate limit restricts how many requests a client can make in a time window. It prevents abuse and ensures fair resource allocation across users."),

    # ── Science ───────────────────────────────────────────────────────────────
    ("What is the speed of light?",
     "The speed of light in a vacuum is approximately 299,792,458 metres per second. It is a fundamental constant denoted c and is the universal speed limit."),
    ("What is the Higgs boson?",
     "The Higgs boson is an elementary particle in the Standard Model that gives other particles mass through the Higgs mechanism. It was confirmed at CERN in 2012."),
    ("Explain quantum entanglement.",
     "Quantum entanglement is a phenomenon where two particles become correlated such that measuring one instantly determines the state of the other, regardless of distance."),
    ("What is DNA replication?",
     "DNA replication unwinds the double helix and uses each strand as a template to synthesise a complementary strand, resulting in two identical DNA molecules."),
    ("What is the greenhouse effect?",
     "The greenhouse effect occurs when atmospheric gases trap heat radiated from Earth's surface. CO2, methane, and water vapour are the primary greenhouse gases."),
    ("What is dark matter?",
     "Dark matter is a hypothetical form of matter that does not interact with light but exerts gravitational effects. It accounts for approximately 27% of the universe's mass-energy."),
    ("What is CRISPR?",
     "CRISPR-Cas9 is a gene editing tool that uses a guide RNA to direct the Cas9 protein to cut specific DNA sequences, enabling precise genetic modifications."),
    ("How does a vaccine work?",
     "A vaccine introduces an antigen — weakened pathogen, protein, or mRNA — that triggers an immune response, training the immune system to recognise and fight the real pathogen later."),

    # ── History ───────────────────────────────────────────────────────────────
    ("When did the First World War begin?",
     "World War I began on 28 July 1914 following the assassination of Archduke Franz Ferdinand of Austria-Hungary. It ended on 11 November 1918."),
    ("What was the Manhattan Project?",
     "The Manhattan Project was a US-led research programme during World War II that developed the first nuclear weapons. It resulted in the atomic bombings of Hiroshima and Nagasaki in 1945."),
    ("What was the Cold War?",
     "The Cold War was a geopolitical tension between the United States and the Soviet Union from 1947 to 1991, characterised by proxy wars, nuclear arms race, and ideological competition."),
    ("Who was Alan Turing?",
     "Alan Turing was a British mathematician and pioneer of computer science. He formulated the concept of the Turing machine and contributed to breaking the Enigma cipher in World War II."),
    ("What was the Space Race?",
     "The Space Race was a competition between the US and USSR to achieve milestones in spaceflight. It began with Sputnik in 1957 and culminated in the Apollo 11 moon landing in 1969."),

    # ── Health ────────────────────────────────────────────────────────────────
    ("What is the difference between aerobic and anaerobic exercise?",
     "Aerobic exercise uses oxygen for sustained energy, like running or cycling. Anaerobic exercise uses stored glycogen for short bursts of high intensity, like sprinting or lifting."),
    ("What is sleep hygiene?",
     "Sleep hygiene refers to habits that promote consistent, quality sleep. This includes a regular sleep schedule, dark cool room, limiting screens before bed, and avoiding caffeine late."),
    ("What causes muscle soreness after exercise?",
     "Delayed onset muscle soreness is caused by micro-tears in muscle fibres from eccentric contractions. The inflammation and repair process causes soreness 24–72 hours later."),
    ("What is intermittent fasting?",
     "Intermittent fasting cycles between periods of eating and fasting. Common protocols include 16:8 (16 hours fasting, 8 hour eating window) and 5:2 (two low-calorie days per week)."),
    ("How much protein should you eat per day?",
     "General recommendations for muscle maintenance are 0.8g per kg of bodyweight. For muscle building, 1.6–2.2g per kg is commonly cited in sports science literature."),

    # ── Finance ───────────────────────────────────────────────────────────────
    ("What is compound interest?",
     "Compound interest calculates interest on both the principal and accumulated interest. The formula is A = P(1 + r/n)^(nt). It causes exponential growth over time."),
    ("What is diversification in investing?",
     "Diversification spreads investments across different assets to reduce risk. If one asset underperforms, others may offset losses. It does not eliminate market risk entirely."),
    ("What is inflation?",
     "Inflation is the rate at which the general price level of goods and services rises, eroding purchasing power. Central banks target around 2% annual inflation in most economies."),
    ("What is a hedge fund?",
     "A hedge fund is a pooled investment vehicle that uses diverse strategies — including leverage, short-selling, and derivatives — to generate returns regardless of market direction."),
    ("What is the difference between stocks and bonds?",
     "Stocks represent ownership in a company with variable returns. Bonds are debt instruments with fixed interest payments. Bonds are generally lower risk but offer lower returns."),
]


# =============================================================================
# NEAR-DUPLICATE VARIANTS  —  stress test dedup
# =============================================================================

NEAR_DUPLICATE_VARIANTS = [
    # Paraphrased versions of existing QA pairs — similar but not identical
    ("Can you explain what backpropagation does?",
     "Backpropagation applies the chain rule to compute gradients of the loss function with respect to weights, propagating error from output back through each layer."),
    ("How does back propagation work in neural networks?",
     "During backprop, the error signal flows backward from the loss through each layer. Gradients are accumulated and weights updated using gradient descent."),
    ("What does the Pixhawk flight controller do?",
     "Pixhawk is an autopilot hardware board for drones running ArduPilot or PX4. It handles stabilisation, GPS navigation, and interfaces with sensors and motors."),
    ("Tell me about LoRa wireless technology.",
     "LoRa provides long-range wireless communication at low power and low data rate. It is popular for IoT and drone telemetry applications requiring km-range links."),
    ("How is cosine similarity calculated?",
     "Cosine similarity is the dot product of two normalised vectors. It measures directional similarity between vectors, returning 1 for identical direction and 0 for orthogonal."),
]


# =============================================================================
# USEFULNESS TIER DISTRIBUTION
# =============================================================================

def sample_usefulness() -> float:
    """
    Returns a usefulness score drawn from a multi-modal distribution.

    Tier distribution:
      15% → near-zero   (0.01–0.09)  stale / never recalled
      25% → low         (0.10–0.39)  rarely used
      40% → medium      (0.40–0.70)  occasionally useful
      20% → high        (0.71–1.00)  frequently recalled
    """
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
    """Random timestamp within the past N days."""
    seconds_back = random.randint(0, days_back_max * 86400)
    return int(time.time()) - seconds_back


def sample_access_count() -> int:
    """Access count weighted toward lower values, occasional spikes."""
    return max(1, int(np.random.exponential(scale=5)) + 1)


# =============================================================================
# INJECTION
# =============================================================================

def build_memory_entry(
    user_content: str,
    assistant_content: str,
    usefulness: float,
    timestamp: int,
    access_count: int,
    source: str = "exchange",
) -> dict:
    """
    Build a Redis hash entry matching the exact format in memory_storage/storage.py.
    """
    content = f"Q: {user_content}\nA: {assistant_content}"
    vec     = embed(content)

    return {
        "content":        content,
        "user_content":   user_content,
        "assistant_content": assistant_content,
        "source":         source,
        "embedding":      _store._serialize_vector(vec),
        "cluster_id":     "injected",
        "token_len":      str(len(content.split())),
        "timestamp":      str(timestamp),
        "usefulness":     str(usefulness),
        "access_count":   str(access_count),
    }


def inject_memories(
    count:    int  = 500,
    dry_run:  bool = False,
    flush:    bool = False,
    verbose:  bool = True,
) -> dict:
    """
    Main injection function.

    Parameters
    ----------
    count   : total number of memories to inject
    dry_run : if True, print entries without writing to Redis
    flush   : if True, wipe all existing mem:* keys before injecting
    verbose : print progress

    Returns dict with injection stats.
    """
    if flush and not dry_run:
        existing = _store.redis.keys("mem:*")
        if existing:
            _store.redis.delete(*existing)
            print(f"  Flushed {len(existing)} existing memory keys.")

    # ── Build full QA list ────────────────────────────────────────────────────
    # Repeat pool to reach count, shuffle for variety
    all_qa = list(QA_POOL) * (count // len(QA_POOL) + 2)
    all_qa = all_qa[:count - len(NEAR_DUPLICATE_VARIANTS)]
    random.shuffle(all_qa)

    # Append near-duplicates at the end
    all_qa += NEAR_DUPLICATE_VARIANTS

    injected    = 0
    skipped     = 0
    usefulness_dist = {"near_zero": 0, "low": 0, "medium": 0, "high": 0}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Memory Injector — {count} entries")
        print(f"  Dry run: {dry_run}")
        print(f"  Near-duplicate pairs: {len(NEAR_DUPLICATE_VARIANTS)}")
        print(f"{'='*60}\n")

    for i, (user_q, asst_a) in enumerate(all_qa):
        usefulness   = sample_usefulness()
        timestamp    = sample_timestamp(days_back_max=60)
        access_count = sample_access_count()

        # Track distribution
        if usefulness < 0.10:
            usefulness_dist["near_zero"] += 1
        elif usefulness < 0.40:
            usefulness_dist["low"] += 1
        elif usefulness < 0.71:
            usefulness_dist["medium"] += 1
        else:
            usefulness_dist["high"] += 1

        if dry_run:
            print(f"  [{i+1:03d}] use={usefulness:.3f}  acc={access_count:3d}  "
                  f"days_ago={int((time.time()-timestamp)/86400):3d}  "
                  f"Q: {user_q[:60]}")
            injected += 1
            continue

        entry = build_memory_entry(
            user_content      = user_q,
            assistant_content = asst_a,
            usefulness        = usefulness,
            timestamp         = timestamp,
            access_count      = access_count,
        )

        key = f"mem:injected:{uuid.uuid4().hex}"
        _store.redis.hset(key, mapping=entry)
        injected += 1

        if verbose and (i + 1) % 50 == 0:
            print(f"  Injected {i+1}/{len(all_qa)}  ...")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_keys = len(_store.redis.keys("mem:*")) if not dry_run else count

    print(f"\n{'='*60}")
    print(f"  INJECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Injected  : {injected}")
    print(f"  Skipped   : {skipped}")
    print(f"  Total mem:* keys in Redis: {total_keys}")
    print(f"\n  Usefulness distribution:")
    print(f"    Near-zero  (0.01–0.09) : {usefulness_dist['near_zero']:4d}  "
          f"({usefulness_dist['near_zero']/injected*100:.1f}%)")
    print(f"    Low        (0.10–0.39) : {usefulness_dist['low']:4d}  "
          f"({usefulness_dist['low']/injected*100:.1f}%)")
    print(f"    Medium     (0.40–0.70) : {usefulness_dist['medium']:4d}  "
          f"({usefulness_dist['medium']/injected*100:.1f}%)")
    print(f"    High       (0.71–1.00) : {usefulness_dist['high']:4d}  "
          f"({usefulness_dist['high']/injected*100:.1f}%)")
    print(f"\n  Suggested next steps:")
    print(f"    python needlebench.py                    # run evaluation")
    print(f"    # In Streamlit: Memory Dedup sweep       # collapse near-dupes")
    print(f"    # In Streamlit: Expire Stale Memories    # TTL=30, floor=0.1")
    print(f"{'='*60}\n")

    return {
        "injected":          injected,
        "skipped":           skipped,
        "total_redis_keys":  total_keys,
        "usefulness_dist":   usefulness_dist,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject fake memories into Redis")
    parser.add_argument("--count",   type=int,  default=500,
                        help="Number of memories to inject (default: 500)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print entries without writing to Redis")
    parser.add_argument("--flush",   action="store_true",
                        help="Wipe existing mem:* keys before injecting")
    parser.add_argument("--quiet",   action="store_true")
    args = parser.parse_args()

    inject_memories(
        count   = args.count,
        dry_run = args.dry_run,
        flush   = args.flush,
        verbose = not args.quiet,
    )
