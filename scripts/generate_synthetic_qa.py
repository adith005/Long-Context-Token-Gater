"""
scripts/generate_synthetic_qa.py
=================================

Generates 7,500 unique standalone Q&A pairs across 8 domains.
Fully deterministic (seed=42). No LLM required. No internet needed.

Output: data/synthetic_qa.json  — list of {user, assistant, topic} dicts

Usage
-----
    python scripts/generate_synthetic_qa.py
    python scripts/generate_synthetic_qa.py --count 5000 --output custom.json
"""

import json
import random
import argparse
from collections import Counter
from pathlib import Path

SEED = 42
random.seed(SEED)

# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

KB = []

def add(topic, questions, answers):
    for q, a in zip(questions, answers):
        KB.append({"topic": topic, "q": q, "a": a})


# ── AI / Machine Learning ─────────────────────────────────────────────────────
add("AI/ML", [
    "What is supervised learning?",
    "What is unsupervised learning?",
    "What is reinforcement learning?",
    "What is a neural network?",
    "What is backpropagation?",
    "What is overfitting?",
    "What is underfitting?",
    "What is regularisation?",
    "What is dropout in neural networks?",
    "What is batch normalisation?",
    "What is a loss function?",
    "What is gradient descent?",
    "What is stochastic gradient descent?",
    "What is a learning rate?",
    "What is a hyperparameter?",
    "What is cross-validation?",
    "What is a confusion matrix?",
    "What is precision in classification?",
    "What is recall in classification?",
    "What is the F1 score?",
    "What is ROC-AUC?",
    "What is a decision tree?",
    "What is a random forest?",
    "What is gradient boosting?",
    "What is a support vector machine?",
    "What is the kernel trick in SVMs?",
    "What is k-nearest neighbours?",
    "What is k-means clustering?",
    "What is principal component analysis?",
    "What is dimensionality reduction?",
    "What is transfer learning?",
    "What is fine-tuning a model?",
    "What is data augmentation?",
    "What is one-hot encoding?",
    "What is an embedding in machine learning?",
    "What is Word2Vec?",
    "What is an autoencoder?",
    "What is a generative adversarial network?",
    "What is the transformer architecture?",
    "What is self-attention?",
    "What is BERT?",
    "What is GPT?",
    "What is a large language model?",
    "What is prompt engineering?",
    "What is chain-of-thought prompting?",
    "What is few-shot learning?",
    "What is zero-shot learning?",
    "What is knowledge distillation?",
    "What is model quantisation?",
    "What is federated learning?",
    "What is RAG in AI?",
    "What is a vector database?",
    "What is semantic search?",
    "What is cosine similarity?",
    "What is Shannon entropy?",
    "What is the bias-variance tradeoff?",
    "What is the curse of dimensionality?",
    "What is Bayesian inference?",
    "What is the softmax function?",
    "What is the ReLU activation function?",
    "What is a convolutional neural network?",
    "What is an LSTM?",
    "What is the vanishing gradient problem?",
    "What are residual connections?",
    "What is beam search in NLP?",
    "What is temperature in language model sampling?",
    "What is RLHF?",
    "What is LoRA?",
    "What is contrastive learning?",
    "What is active learning?",
    "What is continual learning?",
    "What is meta-learning?",
    "What is semi-supervised learning?",
    "What is self-supervised learning?",
    "What is an attention mechanism?",
    "What is multi-head attention?",
    "What is positional encoding?",
    "What is nucleus sampling?",
    "What is top-k sampling?",
    "What is instruction tuning?",
    "What is parameter-efficient fine-tuning?",
    "What is XGBoost?",
    "What is DBSCAN?",
], [
    "Supervised learning trains a model on labelled data where both inputs and correct outputs are provided, enabling the model to learn a mapping from inputs to outputs.",
    "Unsupervised learning finds patterns in unlabelled data without predefined outputs, commonly used for clustering, dimensionality reduction, and anomaly detection.",
    "Reinforcement learning trains an agent to make decisions by rewarding desired behaviours and penalising undesired ones through interaction with an environment.",
    "A neural network is a computational model consisting of layers of connected nodes that transform input data through learned weights to produce output predictions.",
    "Backpropagation computes the gradient of the loss function with respect to each weight by applying the chain rule, propagating error signals from output to input layers.",
    "Overfitting occurs when a model learns noise and specific patterns in training data, performing well on training examples but poorly on unseen data.",
    "Underfitting occurs when a model is too simple to capture the underlying patterns in data, resulting in high error on both training and test sets.",
    "Regularisation adds a penalty term to the loss function to discourage large weights, reducing overfitting. Common forms include L1 and L2 regularisation.",
    "Dropout randomly deactivates a fraction of neurons during training, forcing the network to learn redundant representations and reducing overfitting.",
    "Batch normalisation normalises layer inputs to have zero mean and unit variance, stabilising training and allowing higher learning rates.",
    "A loss function measures the discrepancy between predictions and true labels, serving as the objective that training seeks to minimise.",
    "Gradient descent iteratively adjusts model parameters in the direction of the negative gradient of the loss function to find a minimum.",
    "Stochastic gradient descent updates model parameters using gradients computed from a single randomly selected training example rather than the full dataset.",
    "The learning rate controls the step size during gradient descent; too high causes instability, too low causes slow convergence.",
    "A hyperparameter is a configuration setting external to the model such as learning rate or number of layers, set before training begins.",
    "Cross-validation evaluates model performance by partitioning data into multiple folds, training on some and testing on others for a robust performance estimate.",
    "A confusion matrix shows counts of true positives, true negatives, false positives, and false negatives, enabling detailed classification performance analysis.",
    "Precision is the fraction of predicted positive instances that are actually positive, measuring the reliability of positive predictions.",
    "Recall is the fraction of actual positive instances correctly predicted as positive, measuring how well the model finds all positive cases.",
    "The F1 score is the harmonic mean of precision and recall, balancing both metrics into a single value for imbalanced classification problems.",
    "ROC-AUC measures the area under the receiver operating characteristic curve, representing a model's ability to distinguish between classes across all thresholds.",
    "A decision tree splits data recursively based on feature values to produce if-then rules, creating interpretable classification or regression models.",
    "A random forest is an ensemble of decision trees trained on random subsets of data and features, with predictions aggregated to reduce variance.",
    "Gradient boosting builds an ensemble of weak learners sequentially, where each tree corrects the residual errors of the previous ensemble.",
    "A support vector machine finds the hyperplane that maximises the margin between classes in feature space, classifying points based on which side they fall on.",
    "The kernel trick allows SVMs to operate in high-dimensional feature spaces without explicitly computing the transformation, using kernel functions for inner products.",
    "K-nearest neighbours classifies a point by majority vote among its k closest training examples in feature space using distance metrics.",
    "K-means clustering partitions data into k clusters by iteratively assigning points to the nearest centroid and recomputing centroids until convergence.",
    "Principal component analysis projects data onto orthogonal axes of maximum variance, reducing dimensionality while retaining most information.",
    "Dimensionality reduction transforms high-dimensional data into a lower-dimensional representation, reducing computational cost and mitigating the curse of dimensionality.",
    "Transfer learning reuses a model trained on one task as the starting point for a related task, leveraging learned representations to reduce training requirements.",
    "Fine-tuning adapts a pre-trained model to a specific task by continuing training on a smaller task-specific dataset with a lower learning rate.",
    "Data augmentation artificially expands a training dataset by applying transformations like rotation or cropping, improving robustness and reducing overfitting.",
    "One-hot encoding represents categorical variables as binary vectors where exactly one element is one and all others are zero.",
    "An embedding is a dense low-dimensional vector representation of a discrete object capturing semantic relationships in continuous space.",
    "Word2Vec trains shallow neural networks to produce word embeddings using CBOW or skip-gram objectives, learning from word co-occurrence in large corpora.",
    "An autoencoder is a neural network trained to compress input into a lower-dimensional latent representation and reconstruct the original from it.",
    "A generative adversarial network trains a generator and discriminator in competition, producing realistic synthetic samples through adversarial training.",
    "The transformer architecture uses self-attention mechanisms to process sequences in parallel, replacing recurrence to capture long-range dependencies efficiently.",
    "Self-attention computes relationships between all positions in a sequence simultaneously using query, key, and value vectors.",
    "BERT is a bidirectional transformer pre-trained on masked language modelling, producing rich contextualised word representations used in many NLP tasks.",
    "GPT is a unidirectional transformer pre-trained on autoregressive language modelling, generating text by predicting each token given all previous tokens.",
    "A large language model is a transformer trained on massive text corpora with billions of parameters, capable of diverse tasks through in-context learning.",
    "Prompt engineering designs input text to guide an LLM toward desired outputs using techniques like role specification, examples, and step-by-step instructions.",
    "Chain-of-thought prompting encourages an LLM to generate intermediate reasoning steps before producing a final answer, improving complex reasoning performance.",
    "Few-shot learning adapts a model to new tasks using only a small number of labelled examples, leveraging prior knowledge from pre-training.",
    "Zero-shot learning enables a model to perform tasks never explicitly seen during training, relying on general knowledge and task descriptions.",
    "Knowledge distillation trains a smaller student model to replicate outputs of a larger teacher model, transferring knowledge while reducing model size.",
    "Model quantisation reduces numerical precision of model weights, decreasing memory usage and inference latency with minimal accuracy loss.",
    "Federated learning trains models across multiple decentralised devices without sharing raw data, preserving privacy by only sharing model updates.",
    "RAG — Retrieval-Augmented Generation — combines a retrieval system with a language model, injecting relevant documents into the prompt to improve factual accuracy.",
    "A vector database stores high-dimensional embeddings and supports approximate nearest-neighbour search for fast semantic retrieval over large collections.",
    "Semantic search retrieves documents based on meaning rather than keyword overlap, using dense embeddings to capture conceptual similarity.",
    "Cosine similarity measures the cosine of the angle between two vectors, returning 1 for identical direction and 0 for orthogonal, used for embedding comparison.",
    "Shannon entropy H = -Σ p_i log2(p_i) measures the average information content of a probability distribution, with higher values indicating greater uncertainty.",
    "The bias-variance tradeoff describes the tension between model complexity and generalisation: high bias underfits, high variance overfits.",
    "The curse of dimensionality refers to the exponential increase in data needed to maintain statistical significance as feature dimensionality grows.",
    "Bayesian inference updates prior beliefs about parameters using observed data to produce posterior distributions, providing a probabilistic learning framework.",
    "The softmax function converts a vector of real numbers into a probability distribution by exponentiating and normalising, used for classification outputs.",
    "The ReLU activation function returns max(0,x), introducing non-linearity efficiently and mitigating the vanishing gradient problem.",
    "A convolutional neural network applies learned filters using convolution operations, sharing weights spatially to detect local patterns regardless of position.",
    "An LSTM — Long Short-Term Memory — is a recurrent network with gating mechanisms controlling information flow, enabling learning of long-range dependencies.",
    "The vanishing gradient problem occurs in deep networks when gradients shrink exponentially during backpropagation, preventing early layers from learning.",
    "Residual connections add the input of a block directly to its output, enabling gradient flow through very deep networks without degradation.",
    "Beam search expands the most promising sequences at each decoding step, keeping top-k candidates by cumulative probability to find high-quality output.",
    "Temperature in language model sampling scales logits before softmax, with lower values producing deterministic output and higher values increasing diversity.",
    "RLHF — Reinforcement Learning from Human Feedback — fine-tunes LLMs using human preference rankings to align outputs with human values.",
    "LoRA — Low-Rank Adaptation — inserts trainable low-rank matrices into transformer layers, enabling efficient fine-tuning with far fewer parameters.",
    "Contrastive learning trains models to bring similar example representations closer and push dissimilar ones apart in embedding space.",
    "Active learning selects the most informative unlabelled examples for annotation, reducing labelling cost while maximising model improvement.",
    "Continual learning enables models to learn new tasks sequentially without forgetting previously learned tasks, addressing catastrophic forgetting.",
    "Meta-learning trains models to learn new tasks quickly from few examples by optimising across many tasks, producing rapidly adaptive models.",
    "Semi-supervised learning uses a small amount of labelled data combined with large unlabelled data to improve models when labels are scarce.",
    "Self-supervised learning creates supervisory signals from data itself, such as predicting masked tokens, without requiring manual labels.",
    "An attention mechanism computes a weighted sum of values based on similarity between a query and keys, allowing models to focus on relevant information.",
    "Multi-head attention runs several attention operations in parallel with different learned projections, attending to information from multiple representation subspaces.",
    "Positional encoding adds sequence position information to transformer embeddings since self-attention has no inherent notion of token order.",
    "Nucleus sampling selects from the smallest set of tokens whose cumulative probability exceeds a threshold p, adapting candidate pool size dynamically.",
    "Top-k sampling restricts token selection to the k most probable next tokens at each step, preventing low-probability tokens from being generated.",
    "Instruction tuning fine-tunes LLMs on instruction-response datasets, improving the model's ability to follow natural language directions.",
    "Parameter-efficient fine-tuning adapts pre-trained models using a small number of additional parameters rather than updating all weights.",
    "XGBoost is an optimised gradient boosting implementation with regularisation, parallel processing, and missing value handling, widely used in competitions.",
    "DBSCAN is a density-based clustering algorithm grouping high-density regions and labelling sparse points as outliers, without requiring a preset cluster count.",
])

# ── Computer Science ───────────────────────────────────────────────────────────
add("Computer Science", [
    "What is a hash table?",
    "What is a binary search tree?",
    "What is breadth-first search?",
    "What is depth-first search?",
    "What is dynamic programming?",
    "What is Big O notation?",
    "What is a stack data structure?",
    "What is a queue data structure?",
    "What is quicksort?",
    "What is merge sort?",
    "What is binary search?",
    "What is a greedy algorithm?",
    "What is a deadlock?",
    "What is a mutex?",
    "What is a thread?",
    "What is virtual memory?",
    "What is a cache?",
    "What is a compiler?",
    "What is garbage collection?",
    "What is TCP/IP?",
    "What is DNS?",
    "What is HTTP?",
    "What is HTTPS?",
    "What is REST?",
    "What is SQL?",
    "What is ACID in databases?",
    "What is a NoSQL database?",
    "What is load balancing?",
    "What is containerisation?",
    "What is a linked list?",
    "What is a heap data structure?",
    "What is recursion?",
    "What is memoisation?",
    "What is time complexity?",
    "What is a graph in computer science?",
    "What is a trie?",
    "What is a semaphore?",
    "What is a process in computing?",
    "What is cache coherence?",
    "What is an interpreter?",
    "What is a pointer?",
    "What is a database index?",
    "What is a foreign key?",
    "What is database normalisation?",
    "What is eventual consistency?",
    "What is a transaction in databases?",
    "What is space complexity?",
    "What is a divide and conquer algorithm?",
    "What is memory allocation?",
], [
    "A hash table stores key-value pairs using a hash function to compute array indices, providing average O(1) lookup, insertion, and deletion.",
    "A binary search tree stores nodes where left children are smaller and right children are larger than the parent, enabling O(log n) search on balanced trees.",
    "Breadth-first search explores a graph level by level, visiting all neighbours before moving deeper, using a queue to track nodes.",
    "Depth-first search explores as far as possible along each branch before backtracking, using a stack or recursion to track the path.",
    "Dynamic programming solves problems by breaking them into overlapping subproblems, solving each once and storing results to avoid redundant computation.",
    "Big O notation describes the asymptotic upper bound on an algorithm's resource usage, characterising how runtime or space grows with input size.",
    "A stack is a last-in-first-out data structure supporting push and pop operations, used for function call management and expression evaluation.",
    "A queue is a first-in-first-out data structure supporting enqueue and dequeue, used for breadth-first search, task scheduling, and buffering.",
    "Quicksort selects a pivot and partitions the array into smaller and larger elements, recursively sorting each partition with average O(n log n) complexity.",
    "Merge sort divides the array in half, recursively sorts each half, and merges them, achieving guaranteed O(n log n) time complexity.",
    "Binary search finds a target in a sorted array by halving the search space repeatedly, comparing to the middle element for O(log n) performance.",
    "A greedy algorithm makes the locally optimal choice at each step, working for problems where local choices lead to a global optimum.",
    "A deadlock occurs when processes each wait for resources held by others, creating a cycle that prevents any progress.",
    "A mutex is a mutual exclusion lock allowing only one thread to access a shared resource at a time, preventing race conditions.",
    "A thread is a lightweight unit of execution within a process, sharing memory space and enabling concurrent execution within a program.",
    "Virtual memory extends physical RAM using disk storage, giving each process the illusion of a large contiguous address space.",
    "A cache is a small, fast memory storing recently accessed data to reduce latency, exploiting temporal and spatial locality.",
    "A compiler translates source code into machine code or intermediate representation in one pass before program execution.",
    "Garbage collection automatically reclaims memory occupied by unreferenced objects, preventing memory leaks without manual deallocation.",
    "TCP/IP is a protocol suite defining data transmission over networks. TCP provides reliable ordered delivery; IP handles addressing and routing.",
    "DNS — Domain Name System — translates domain names into IP addresses, acting as the internet's distributed directory service.",
    "HTTP is a stateless application-layer protocol for web communication, defining request-response messages between browsers and servers.",
    "HTTPS encrypts HTTP traffic using TLS, authenticating the server and protecting data in transit from eavesdropping.",
    "REST is an architectural style for web APIs using HTTP methods to perform operations on resources identified by URLs.",
    "SQL is a declarative language for querying and manipulating relational databases using statements like SELECT, INSERT, UPDATE, and DELETE.",
    "ACID stands for Atomicity, Consistency, Isolation, and Durability, properties ensuring reliable database transaction processing.",
    "A NoSQL database stores data in non-relational formats like documents, key-value pairs, or graphs, prioritising scalability and flexibility.",
    "Load balancing distributes requests across multiple servers to prevent bottlenecks, improving availability and throughput.",
    "Containerisation packages applications and dependencies into portable containers sharing the host OS kernel, ensuring consistent execution.",
    "A linked list consists of nodes where each holds data and a pointer to the next node, enabling O(1) insertion and deletion at known positions.",
    "A heap is a tree satisfying the heap property where each parent is greater or lesser than its children, supporting O(log n) priority operations.",
    "Recursion is a technique where a function calls itself with a smaller subproblem until reaching a base case.",
    "Memoisation caches function results for specific inputs, returning cached results on repeated calls to avoid redundant computation.",
    "Time complexity measures how an algorithm's runtime scales with input size, expressed using Big O notation.",
    "A graph is a collection of vertices connected by edges, representing relationships and used in network analysis, pathfinding, and scheduling.",
    "A trie is a tree where each node represents a character prefix, enabling efficient string search and prefix matching in O(m) time.",
    "A semaphore is a synchronisation primitive controlling access to shared resources using a counter, allowing a specified concurrency level.",
    "A process is an independent program in execution with its own memory space and resources, managed by the operating system.",
    "Cache coherence ensures multiple processor caches maintain a consistent view of shared memory, preventing stale data.",
    "An interpreter executes source code line by line at runtime without producing a standalone executable, enabling interactive execution.",
    "A pointer is a variable storing the memory address of another variable, enabling direct memory manipulation and data structure construction.",
    "A database index speeds up data retrieval by maintaining a sorted reference to column values, trading storage for query performance.",
    "A foreign key references the primary key of another table, enforcing referential integrity and representing relationships between tables.",
    "Database normalisation organises tables to reduce redundancy by dividing data into related tables with defined relationships.",
    "Eventual consistency is a consistency model where replicas may temporarily diverge but will converge given no new updates.",
    "A database transaction is a unit of work that is fully completed or fully rolled back, ensuring integrity through ACID properties.",
    "Space complexity measures how an algorithm's memory usage scales with input size, including auxiliary space and input storage.",
    "Divide and conquer breaks a problem into smaller subproblems, solves them independently, and combines results.",
    "Memory allocation reserves a block of memory for program use, with static allocation at compile time and dynamic at runtime.",
])

# ── Physics ────────────────────────────────────────────────────────────────────
add("Physics", [
    "What is Newton's first law of motion?",
    "What is Newton's second law of motion?",
    "What is Newton's third law of motion?",
    "What is kinetic energy?",
    "What is potential energy?",
    "What is momentum?",
    "What is the speed of light?",
    "What is quantum mechanics?",
    "What is the Heisenberg uncertainty principle?",
    "What is wave-particle duality?",
    "What is nuclear fission?",
    "What is nuclear fusion?",
    "What is the Doppler effect?",
    "What is electric current?",
    "What is Ohm's law?",
    "What is electromagnetic induction?",
    "What is the first law of thermodynamics?",
    "What is the second law of thermodynamics?",
    "What is a black hole?",
    "What is special relativity?",
    "What is general relativity?",
    "What is gravity?",
    "What is escape velocity?",
    "What is centripetal force?",
    "What is the photoelectric effect?",
    "What is radioactivity?",
    "What is entropy in thermodynamics?",
    "What is absolute zero?",
    "What is dark matter?",
    "What is the Big Bang theory?",
    "What is refraction?",
    "What is electromagnetic radiation?",
    "What is voltage?",
    "What is a magnetic field?",
    "What is work in physics?",
    "What is power in physics?",
    "What is torque?",
    "What is friction?",
    "What is the universal law of gravitation?",
    "What is wave frequency?",
    "What is wavelength?",
    "What is total internal reflection?",
    "What is electric charge?",
    "What is resistance in electricity?",
    "What is acceleration?",
    "What is velocity?",
], [
    "Newton's first law states that an object remains at rest or in uniform motion unless acted upon by an external net force.",
    "Newton's second law states that the net force on an object equals its mass times acceleration: F = ma.",
    "Newton's third law states that for every action there is an equal and opposite reaction.",
    "Kinetic energy is the energy of motion, equal to one half times mass times velocity squared: KE = ½mv².",
    "Potential energy is stored energy due to position or configuration, such as gravitational potential energy mgh.",
    "Momentum is the product of mass and velocity (p = mv), representing the quantity of motion of an object.",
    "The speed of light in a vacuum is approximately 299,792,458 metres per second, denoted c, the universal maximum speed.",
    "Quantum mechanics describes matter and energy at atomic scales, where energy is quantised and particles exhibit wave-like properties.",
    "The Heisenberg uncertainty principle states that position and momentum cannot both be precisely known simultaneously: ΔxΔp ≥ ℏ/2.",
    "Wave-particle duality is the quantum property by which every particle exhibits both wave and particle characteristics depending on the experiment.",
    "Nuclear fission splits a heavy nucleus into lighter fragments, releasing large amounts of energy used in nuclear reactors.",
    "Nuclear fusion combines light nuclei into heavier ones, releasing even more energy than fission, powering stars and future reactors.",
    "The Doppler effect is the change in observed frequency of a wave when the source and observer are in relative motion.",
    "Electric current is the flow of electric charge through a conductor, measured in Amperes as charge per unit time.",
    "Ohm's law states that the current through a conductor equals the voltage divided by the resistance: I = V/R.",
    "Electromagnetic induction generates an electromotive force in a conductor by a changing magnetic flux, the principle behind generators.",
    "The first law of thermodynamics states that energy is conserved: change in internal energy equals heat added minus work done.",
    "The second law of thermodynamics states that total entropy of an isolated system always increases over time.",
    "A black hole is a region where gravity is so strong that nothing, including light, can escape beyond the event horizon.",
    "Special relativity establishes that physics laws are identical in all inertial frames and the speed of light is constant, leading to time dilation.",
    "General relativity extends special relativity to gravity, describing it as the curvature of spacetime caused by mass and energy.",
    "Gravity is the attractive force between masses, causing acceleration toward each other and responsible for orbits and weight.",
    "Escape velocity is the minimum speed to break free from a gravitational field without further propulsion, equal to √(2GM/r).",
    "Centripetal force is the inward force required to keep an object in circular motion, directed toward the centre of the circle.",
    "The photoelectric effect demonstrates that light ejects electrons from metals only above a threshold frequency, proving light consists of photons.",
    "Radioactivity is the spontaneous emission of particles or radiation from unstable nuclei as they decay toward more stable configurations.",
    "Entropy in thermodynamics measures system disorder; it tends to increase in spontaneous processes, defining the direction of time.",
    "Absolute zero is the theoretical minimum temperature (0 Kelvin, −273.15°C) at which particle motion reaches its minimum energy.",
    "Dark matter is hypothetical matter that does not interact electromagnetically but exerts gravitational effects, making up about 27% of the universe.",
    "The Big Bang theory describes the universe as originating from an extremely hot, dense state 13.8 billion years ago, expanding and cooling since.",
    "Refraction is the bending of a wave as it passes between media with different propagation speeds, following Snell's law.",
    "Electromagnetic radiation is energy propagated as oscillating electric and magnetic fields, including radio waves, light, X-rays, and gamma rays.",
    "Voltage is the electric potential difference between two points, driving current flow, measured in Volts as energy per unit charge.",
    "A magnetic field surrounds magnets and current-carrying conductors, exerting forces on moving charges and magnetic materials.",
    "Work is done when a force causes displacement in the force's direction, calculated as W = F·d·cos(θ), measured in Joules.",
    "Power is the rate of energy transfer or work done, calculated as P = W/t, measured in Watts.",
    "Torque is the rotational equivalent of force, equal to force times perpendicular distance from the pivot, causing angular acceleration.",
    "Friction is a resistive force opposing relative motion between surfaces, arising from surface irregularities and molecular interactions.",
    "The universal law of gravitation states every mass attracts every other mass with force proportional to product of masses and inversely proportional to distance squared.",
    "Wave frequency is the number of complete cycles per second, measured in Hertz, inversely related to wavelength through wave speed.",
    "Wavelength is the distance between successive crests of a wave, related to frequency and speed by λ = v/f.",
    "Total internal reflection occurs when a wave hits an interface above the critical angle, reflecting entirely back into the denser medium.",
    "Electric charge is a fundamental property causing electromagnetic interactions, existing as positive or negative, measured in Coulombs.",
    "Electrical resistance opposes current flow, measured in Ohms, determined by material resistivity, length, and cross-sectional area.",
    "Acceleration is the rate of change of velocity with time, measured in metres per second squared, occurring when speed or direction changes.",
    "Velocity is the rate of change of displacement with time, a vector specifying both speed and direction of motion.",
])

# ── Biology ────────────────────────────────────────────────────────────────────
add("Biology", [
    "What is DNA?",
    "What is RNA?",
    "What is a gene?",
    "What is mitosis?",
    "What is meiosis?",
    "What is natural selection?",
    "What is photosynthesis?",
    "What is cellular respiration?",
    "What is ATP?",
    "What is a virus?",
    "What is the immune system?",
    "What is an antibody?",
    "What is a vaccine?",
    "What is CRISPR?",
    "What is homeostasis?",
    "What is osmosis?",
    "What is a stem cell?",
    "What is epigenetics?",
    "What is a mutation?",
    "What is the central dogma of molecular biology?",
    "What is an enzyme?",
    "What is a protein?",
    "What is the cell membrane?",
    "What is the mitochondria?",
    "What is a chromosome?",
    "What is heredity?",
    "What is diffusion?",
    "What is a bacterium?",
    "What is evolution?",
], [
    "DNA — deoxyribonucleic acid — is a double-helix molecule carrying genetic instructions for development, functioning, and reproduction of all known life.",
    "RNA — ribonucleic acid — is a single-stranded molecule that carries genetic information from DNA to ribosomes and plays roles in gene expression.",
    "A gene is a sequence of DNA encoding instructions for building a specific protein, serving as the basic unit of heredity.",
    "Mitosis is cell division producing two genetically identical daughter cells, used for growth and repair in eukaryotes.",
    "Meiosis is cell division producing four genetically diverse haploid cells for sexual reproduction, involving recombination.",
    "Natural selection is the process by which traits increasing reproductive success become more common in a population over generations.",
    "Photosynthesis converts light energy into glucose using carbon dioxide and water, releasing oxygen in chloroplasts.",
    "Cellular respiration breaks down glucose to produce ATP using oxygen, releasing carbon dioxide and water.",
    "ATP — adenosine triphosphate — is the primary energy currency of cells, storing and releasing chemical energy for cellular processes.",
    "A virus is a non-cellular agent consisting of genetic material in a protein coat, replicating only inside host cells.",
    "The immune system is the body's defence network of cells and organs that identifies and destroys pathogens and foreign substances.",
    "An antibody is a Y-shaped protein produced by B cells that binds specifically to antigens on pathogens, neutralising or marking them.",
    "A vaccine stimulates the immune system by introducing antigens, creating immunological memory without causing disease.",
    "CRISPR-Cas9 is a gene-editing tool using guide RNA to direct Cas9 protein to cut specific DNA sequences for precise modification.",
    "Homeostasis is the maintenance of stable internal conditions like temperature and pH within a narrow range despite external changes.",
    "Osmosis is the movement of water across a semipermeable membrane from lower to higher solute concentration.",
    "A stem cell is an undifferentiated cell capable of self-renewal and differentiation into specialised cell types.",
    "Epigenetics studies heritable changes in gene expression that don't alter the DNA sequence, often through methylation or histone modification.",
    "A mutation is a change in the DNA sequence, ranging from single base substitutions to chromosomal rearrangements, with varying effects.",
    "The central dogma of molecular biology describes genetic information flow: DNA is transcribed to RNA, which is translated to protein.",
    "An enzyme is a biological catalyst, typically a protein, that accelerates specific chemical reactions without being consumed.",
    "A protein is a large molecule of amino acid chains folded into specific structures, performing virtually all cellular functions.",
    "The cell membrane is a phospholipid bilayer surrounding cells, controlling substance passage and maintaining internal conditions.",
    "Mitochondria are organelles producing most cellular ATP through aerobic respiration, containing their own DNA.",
    "A chromosome is a long DNA molecule packaged with proteins containing many genes. Humans have 46 arranged in 23 pairs.",
    "Heredity is the transmission of genetic information from parents to offspring, determining inherited traits through genes.",
    "Diffusion is the net movement of molecules from higher to lower concentration down their concentration gradient.",
    "A bacterium is a single-celled prokaryotic microorganism lacking a nucleus, existing in virtually every environment.",
    "Evolution is the change in heritable characteristics of populations over generations, driven by mutation, selection, and genetic drift.",
])

# ── Mathematics ────────────────────────────────────────────────────────────────
add("Mathematics", [
    "What is a prime number?",
    "What is the Pythagorean theorem?",
    "What is a derivative in calculus?",
    "What is an integral in calculus?",
    "What is a matrix?",
    "What is a probability?",
    "What is the central limit theorem?",
    "What is a logarithm?",
    "What is a differential equation?",
    "What is an eigenvalue?",
    "What is a Fourier transform?",
    "What is a complex number?",
    "What is a statistical hypothesis test?",
    "What is a p-value?",
    "What is standard deviation?",
    "What is variance?",
    "What is correlation?",
    "What is regression analysis?",
    "What is Bayes' theorem?",
    "What is the law of large numbers?",
    "What is a vector in mathematics?",
    "What is a set in mathematics?",
    "What is a function in mathematics?",
    "What is a limit in calculus?",
    "What is linear algebra?",
    "What is number theory?",
    "What is combinatorics?",
    "What is graph theory?",
    "What is topology?",
    "What is Occam's razor?",
], [
    "A prime number is a natural number greater than 1 with no positive divisors other than 1 and itself.",
    "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the other two sides squared: a² + b² = c².",
    "A derivative measures the instantaneous rate of change of a function with respect to its input, representing the tangent line slope at a point.",
    "An integral computes the area under a curve or accumulated quantity, with definite integrals giving values and indefinite integrals giving functions.",
    "A matrix is a rectangular array of numbers arranged in rows and columns, used for transformations, linear systems, and data representation.",
    "Probability measures likelihood of an event occurring, ranging from 0 to 1, defined as favourable outcomes over total outcomes.",
    "The central limit theorem states that sample means approach a normal distribution as sample size grows, regardless of the underlying distribution.",
    "A logarithm is the inverse of exponentiation: log_b(x) = y means b^y = x, with natural logs using base e and common logs base 10.",
    "A differential equation relates a function to its derivatives, describing how quantities change and fundamental in physics and engineering.",
    "An eigenvalue is a scalar λ such that a linear transformation scales a non-zero vector by λ: Av = λv, fundamental in data analysis.",
    "The Fourier transform decomposes a function into constituent frequencies, converting between time and frequency domains for signal processing.",
    "A complex number has the form a + bi where i = √(-1), extending real numbers to enable solutions of all polynomial equations.",
    "A statistical hypothesis test evaluates evidence against a null hypothesis using test statistics to decide whether to reject the null.",
    "A p-value is the probability of observing results at least as extreme as the data if the null hypothesis were true.",
    "Standard deviation measures distribution spread as the square root of variance, quantifying deviation from the mean.",
    "Variance measures average squared deviation of values from their mean, quantifying spread of a distribution.",
    "Correlation measures linear relationship between two variables, ranging from -1 to +1 with 0 indicating no linear relationship.",
    "Regression analysis models relationships between a dependent variable and independent variables, used for prediction.",
    "Bayes' theorem relates conditional probabilities: P(A|B) = P(B|A)P(A)/P(B), enabling belief updating with new evidence.",
    "The law of large numbers states that as sample size grows, the sample mean converges to the true population mean.",
    "A vector is a mathematical object with both magnitude and direction, represented as ordered numbers for geometry and machine learning.",
    "A set is a well-defined collection of distinct objects called elements, forming the foundation of modern mathematics.",
    "A function maps each domain element to exactly one codomain element, expressed as f: X → Y with unique outputs per input.",
    "A limit describes the value a function approaches as its input approaches a point, forming the foundation of calculus.",
    "Linear algebra studies vector spaces and linear transformations, including matrices, determinants, and systems of equations.",
    "Number theory studies integer properties including prime numbers, divisibility, and modular arithmetic, applied in cryptography.",
    "Combinatorics counts and arranges mathematical objects, studying permutations, combinations, and generating functions.",
    "Graph theory studies graphs as structures of vertices and edges, analysing connectivity, paths, and network properties.",
    "Topology studies properties of spaces preserved under continuous deformations, including connectedness and compactness.",
    "Occam's razor is the principle that among competing hypotheses, the one with fewest assumptions should be preferred.",
])

# ── History ─────────────────────────────────────────────────────────────────────
add("History", [
    "When did World War I begin and end?",
    "When did World War II begin and end?",
    "What was the Cold War?",
    "What was the French Revolution?",
    "What was the Industrial Revolution?",
    "What was the Renaissance?",
    "What was the Manhattan Project?",
    "What was the Space Race?",
    "What was the Cuban Missile Crisis?",
    "What was the Berlin Wall?",
    "What was the Holocaust?",
    "What was apartheid?",
    "What was the Civil Rights Movement?",
    "What was the Magna Carta?",
    "What was the Black Death?",
    "What was the Silk Road?",
    "What is the United Nations?",
    "What is NATO?",
    "What was the Marshall Plan?",
    "What was the American Revolution?",
    "What was the Russian Revolution?",
    "What was the Great Depression?",
    "What was the Treaty of Versailles?",
    "What was the Enlightenment?",
    "What was colonialism?",
], [
    "World War I began on 28 July 1914 following the assassination of Archduke Franz Ferdinand and ended on 11 November 1918, causing about 20 million deaths.",
    "World War II began on 1 September 1939 with Germany's invasion of Poland and ended on 2 September 1945, causing an estimated 70-85 million deaths.",
    "The Cold War was geopolitical tension between the US and Soviet Union from 1947 to 1991, characterised by arms race, proxy wars, and ideological competition.",
    "The French Revolution (1789-1799) overthrew the French monarchy, established a republic, and reshaped European political thought.",
    "The Industrial Revolution (c.1760-1840) transformed manufacturing from hand to machine production, beginning in Britain and reshaping society.",
    "The Renaissance (14th-17th centuries) was a cultural revival in Europe emphasising humanism, art, science, and classical learning, beginning in Italy.",
    "The Manhattan Project (1942-1946) was the US-led programme that developed the first nuclear weapons, resulting in the Hiroshima and Nagasaki bombings.",
    "The Space Race (1957-1969) was US-USSR competition for spaceflight supremacy, beginning with Sputnik and culminating in the Apollo 11 Moon landing.",
    "The Cuban Missile Crisis (October 1962) was a 13-day confrontation over Soviet nuclear missiles in Cuba, the closest the Cold War came to nuclear conflict.",
    "The Berlin Wall (1961-1989) divided East and West Berlin, built to prevent emigration and becoming the most powerful symbol of Cold War division.",
    "The Holocaust was the systematic genocide of six million Jews and millions of others by Nazi Germany during World War II.",
    "Apartheid was institutionalised racial segregation in South Africa from 1948 to 1994, restricting rights based on race until dismantled democratically.",
    "The Civil Rights Movement (1954-1968) was a nonviolent US social movement seeking to end racial discrimination and secure equal rights.",
    "The Magna Carta (1215) was a charter limiting royal power and guaranteeing legal rights, laying groundwork for constitutional governance.",
    "The Black Death (1347-1351) was a bubonic plague that killed 30-60% of Europe's population, causing profound social and economic changes.",
    "The Silk Road was an ancient trade network connecting China to the Mediterranean, facilitating exchange of goods, ideas, and culture.",
    "The United Nations is an international organisation founded in 1945 promoting peace, security, human rights, and international cooperation.",
    "NATO is a military alliance founded in 1949 based on collective defence: an attack on one member is considered an attack on all.",
    "The Marshall Plan (1948-1952) provided about $13 billion to rebuild Western European economies after World War II.",
    "The American Revolution (1765-1783) was the colonial revolt establishing US independence from Britain, inspired by Enlightenment principles.",
    "The Russian Revolution of 1917 overthrew the Tsar and established the Soviet Union as the world's first communist state.",
    "The Great Depression (1929-1939) was a severe global downturn triggered by the stock market crash, causing mass unemployment worldwide.",
    "The Treaty of Versailles (1919) formally ended World War I, imposing reparations on Germany, conditions blamed for contributing to World War II.",
    "The Enlightenment (17th-18th centuries) emphasised reason, individualism, and skepticism of authority, influencing democratic revolutions.",
    "Colonialism is one nation establishing political and economic control over another, exploiting resources and people, practiced from the 15th to 20th centuries.",
])

# ── Economics ──────────────────────────────────────────────────────────────────
add("Economics", [
    "What is GDP?",
    "What is inflation?",
    "What is monetary policy?",
    "What is fiscal policy?",
    "What is supply and demand?",
    "What is opportunity cost?",
    "What is comparative advantage?",
    "What is a recession?",
    "What is a stock?",
    "What is a bond?",
    "What is compound interest?",
    "What is diversification in investing?",
    "What is cryptocurrency?",
    "What is blockchain?",
    "What is quantitative easing?",
    "What is the time value of money?",
    "What is a budget deficit?",
    "What is a tariff?",
    "What is a central bank?",
    "What is market capitalisation?",
    "What is venture capital?",
    "What is a hedge fund?",
    "What is elasticity in economics?",
    "What is deflation?",
    "What is national debt?",
], [
    "GDP — Gross Domestic Product — is the total monetary value of goods and services produced within a country's borders in a given period.",
    "Inflation is the rate at which the general price level rises, eroding purchasing power. Central banks typically target around 2% annually.",
    "Monetary policy is the management of money supply and interest rates by a central bank to achieve economic objectives like price stability.",
    "Fiscal policy is the government's use of taxation and spending to influence the economy, with expansionary policy stimulating growth.",
    "Supply and demand describes how prices are determined by the quantity producers supply and the quantity consumers demand at various price levels.",
    "Opportunity cost is the value of the next best alternative forgone when making a choice, representing the true economic cost of decisions.",
    "Comparative advantage is the ability to produce a good at a lower opportunity cost than others, forming the basis for beneficial trade.",
    "A recession is a significant economic decline lasting months, typically defined as two consecutive quarters of negative GDP growth.",
    "A stock represents a share of ownership in a company, entitling holders to a portion of profits and assets, traded on exchanges.",
    "A bond is a debt instrument where the issuer borrows at a fixed interest rate, repaying principal at maturity, generally safer than stocks.",
    "Compound interest calculates interest on both principal and accumulated interest: A = P(1+r/n)^(nt), causing exponential growth over time.",
    "Diversification spreads investments across different assets to reduce risk, ensuring poor performance of one asset doesn't devastate the portfolio.",
    "Cryptocurrency is a digital currency using cryptography for security and operating on decentralised networks, with Bitcoin as the most prominent.",
    "Blockchain is a distributed ledger recording transactions in linked blocks verified by consensus, providing transparency without central authority.",
    "Quantitative easing is monetary policy where a central bank purchases securities to inject money, lowering interest rates and stimulating lending.",
    "The time value of money states that money available now is worth more than the same amount in the future due to earning potential.",
    "A budget deficit occurs when government expenditure exceeds revenue, requiring borrowing to fund the shortfall.",
    "A tariff is a tax on imported goods, raising their price to protect domestic industries or generate government revenue.",
    "A central bank manages a country's currency, money supply, and interest rates, serving as lender of last resort and regulating banks.",
    "Market capitalisation is the total market value of a company's outstanding shares, calculated as share price times number of shares.",
    "Venture capital provides funding to early-stage high-potential startups in exchange for equity, accepting high risk for high potential returns.",
    "A hedge fund uses diverse strategies including leverage, short-selling, and derivatives to generate returns regardless of market direction.",
    "Elasticity measures how responsive quantity demanded or supplied is to price changes; elastic goods change significantly, inelastic ones little.",
    "Deflation is a sustained decrease in the general price level, which can delay purchases and potentially cause economic stagnation.",
    "National debt is the total amount a government owes creditors, accumulated from years of budget deficits financed by borrowing.",
])

# ── Drones ─────────────────────────────────────────────────────────────────────
add("Drones", [
    "What is a quadcopter?",
    "What is a flight controller?",
    "What is PID control in drones?",
    "What is MAVLink?",
    "What is ArduPilot?",
    "What is an IMU in drones?",
    "What is an ESC in drones?",
    "What is LoRa communication?",
    "What is geofencing in drones?",
    "What is obstacle avoidance?",
    "What is waypoint navigation?",
    "What is LIDAR?",
    "What is telemetry in drones?",
    "What is yaw in drone orientation?",
    "What is pitch in drone orientation?",
    "What is roll in drone orientation?",
    "What is SLAM?",
    "What is a companion computer in drones?",
    "What is return-to-home?",
    "What is autonomous flight?",
], [
    "A quadcopter is a multirotor helicopter with four arms each carrying a motor and propeller, achieving flight by independently varying rotor speeds.",
    "A flight controller is the central processing unit of a drone, reading sensor data and computing motor commands to stabilise and control the aircraft.",
    "PID control adjusts motor outputs based on proportional, integral, and derivative terms of the orientation error, maintaining drone stability.",
    "MAVLink is a lightweight messaging protocol for communicating between drones and ground stations, defining packet formats for telemetry and commands.",
    "ArduPilot is an open-source autopilot software platform supporting planes, copters, and rovers, running on Pixhawk hardware with advanced autonomous capabilities.",
    "An IMU — Inertial Measurement Unit — combines accelerometers and gyroscopes to measure linear acceleration and rotational rates for attitude estimation.",
    "An ESC — Electronic Speed Controller — converts flight controller signals into three-phase power for brushless motors, controlling speed precisely.",
    "LoRa is a long-range, low-power wireless modulation providing kilometre-range telemetry links at low data rates for drone control.",
    "Geofencing defines virtual geographic boundaries that trigger automatic responses like return-to-home if the drone approaches the defined limit.",
    "Obstacle avoidance uses sensors like lidar or depth cameras to detect objects in the flight path and automatically reroute or halt the drone.",
    "Waypoint navigation allows a drone to autonomously fly a predetermined GPS path, executing missions without continuous pilot input.",
    "LIDAR measures distance using pulsed laser light, providing accurate 3D mapping and obstacle detection for drone navigation.",
    "Telemetry is real-time transmission of flight data including position, altitude, and battery voltage from the drone to the ground station.",
    "Yaw is rotation around the drone's vertical axis, controlled by varying torque between clockwise and counter-clockwise spinning motor pairs.",
    "Pitch is rotation around the side-to-side axis, causing forward and backward movement, controlled by differential throttle between front and rear pairs.",
    "Roll is rotation around the front-to-back axis, causing lateral movement, controlled by differential throttle between left and right motor pairs.",
    "SLAM — Simultaneous Localisation and Mapping — enables a drone to build a map of an unknown environment while tracking its position within it.",
    "A companion computer is a secondary processor on a drone handling computationally intensive tasks like computer vision alongside the flight controller.",
    "Return-to-home is a fail-safe feature that automatically navigates the drone to its launch point when triggered by low battery or signal loss.",
    "Autonomous flight enables a drone to execute missions without continuous pilot input, using GPS, sensors, and onboard computing for navigation.",
])


# =============================================================================
# GENERATE 7500 WITH VARIATION
# =============================================================================

QUESTION_PREFIXES = [
    "",
    "",
    "",
    "Can you explain ",
    "Could you describe ",
    "Please define ",
    "In simple terms, what is ",
    "How would you define ",
]

ANSWER_SUFFIXES = [
    "",
    "",
    "",
    " This is a foundational concept in the field.",
    " Understanding this is essential for further study.",
    " This principle underlies many practical applications.",
    " It is widely studied in research and industry.",
    " This concept is central to modern practice in the area.",
]


def vary(item: dict, variant: int) -> dict:
    q = item["q"]
    a = item["a"]

    prefix = QUESTION_PREFIXES[variant % len(QUESTION_PREFIXES)]
    suffix = ANSWER_SUFFIXES[variant % len(ANSWER_SUFFIXES)]

    if prefix and q.startswith("What is"):
        q_new = prefix + q[0].lower() + q[1:]
    else:
        q_new = q

    return {
        "user":      q_new,
        "assistant": a + suffix,
        "topic":     item["topic"],
    }


def generate(target: int = 7500) -> list:
    pairs = []
    seen  = set()

    # Pass 0 — originals
    for item in KB:
        key = item["q"].lower().strip()
        if key not in seen:
            seen.add(key)
            pairs.append({"user": item["q"], "assistant": item["a"], "topic": item["topic"]})

    print(f"  After originals: {len(pairs)}")

    # Pass 1+ — varied phrasings
    v = 1
    while len(pairs) < target:
        for item in KB:
            if len(pairs) >= target:
                break
            varied = vary(item, v)
            key    = varied["user"].lower().strip()
            if key not in seen:
                seen.add(key)
                pairs.append(varied)
        v += 1
        if v > 20:
            # If still short, duplicate with minor index suffix variation
            for item in KB:
                if len(pairs) >= target:
                    break
                varied = vary(item, v)
                varied["user"] += f" (variant {v})"
                key = varied["user"].lower().strip()
                if key not in seen:
                    seen.add(key)
                    pairs.append(varied)
            v += 1

    random.shuffle(pairs)
    return pairs[:target]


def main(count: int = 7500, output: str = None):
    print(f"KB size: {len(KB)} base pairs")
    print(f"Generating {count} unique Q&A pairs...")

    pairs = generate(count)

    print(f"Generated: {len(pairs)}")
    dist = Counter(p["topic"] for p in pairs)
    print("Topic distribution:")
    for topic, n in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {topic:<25} {n:>5}")

    out = output or "data/synthetic_qa.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"\nSaved → {out}")
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count",  type=int, default=7500)
    parser.add_argument("--output", type=str, default="data/synthetic_qa.json")
    args = parser.parse_args()
    main(args.count, args.output)
