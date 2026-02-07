from memory.history_db import create_entry

# General academic context
create_entry(
    "Explain quantum entanglement in simple terms.",
    "Quantum entanglement occurs when two particles share a state such that measuring one instantly determines the other's state, regardless of distance.",
    tags=["physics", "quantum"]
)

create_entry(
    "How does the Large Hadron Collider detect collisions?",
    "The LHC uses layers of detectors such as calorimeters and muon chambers to record particle traces from high-energy collisions.",
    tags=["CERN", "particle_physics"]
)

create_entry(
    "What is the purpose of the ATLAS experiment?",
    "ATLAS studies the fundamental particles of the universe using proton-proton collisions, focusing on Higgs boson and supersymmetry searches.",
    tags=["CERN", "experiments"]
)

# Software & computation
create_entry(
    "How do you implement backpropagation in Python?",
    "Use libraries like PyTorch or TensorFlow to define computational graphs and automatically compute gradients for parameter updates.",
    tags=["AI", "ML", "Python"]
)

create_entry(
    "What is gradient descent?",
    "Gradient descent is an optimization algorithm that minimizes loss by iteratively adjusting weights in the opposite direction of the gradient.",
    tags=["ML", "optimization"]
)

# Research & data storage (CERN-relevant)
create_entry(
    "How does CERN manage petabytes of collision data?",
    "CERN uses a distributed computing model called the Worldwide LHC Computing Grid (WLCG) to store and analyze petabytes of collision data across global centers.",
    tags=["CERN", "data_management", "grid_computing"]
)

create_entry(
    "What are the main challenges in particle data storage?",
    "High data throughput, real-time filtering, redundancy management, and distributed consistency are major issues.",
    tags=["data_storage", "physics"]
)

# General AI context for gating differentiation
create_entry(
    "What are attention mechanisms in transformers?",
    "Attention mechanisms allow models to focus on relevant parts of input sequences by computing weighted relationships between tokens.",
    tags=["AI", "transformers"]
)

create_entry(
    "How can embeddings be used for information retrieval?",
    "Embeddings map text to vector space; similarity between vectors indicates semantic relatedness, enabling effective retrieval.",
    tags=["AI", "IR", "embeddings"]
)

# Irrelevant / distractor context
create_entry(
    "What are the best beaches in Kerala?",
    "Varkala and Kovalam are two of the most popular beaches known for their cliffs and calm waters.",
    tags=["travel", "kerala"]
)

create_entry(
    "How do plants perform photosynthesis?",
    "Plants convert sunlight, carbon dioxide, and water into glucose and oxygen via chlorophyll in chloroplasts.",
    tags=["biology", "plants"]
)

print("✅ Redis memory populated successfully!")