import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

# Paths (all file-based, no DB)
RAW_DIR      = DATA / "raw"
INDEXES_DIR  = DATA / "indexes"
CACHE_DIR    = DATA / "cache"
LOGS_DIR     = DATA / "logs"

ARTICLES_FILE  = RAW_DIR / "articles.json"
FAISS_FILE     = INDEXES_DIR / "faiss.index"
FAISS_MAP_FILE = INDEXES_DIR / "faiss_map.pkl"
BM25_FILE      = INDEXES_DIR / "bm25.pkl"
CHUNKS_FILE    = INDEXES_DIR / "chunks.pkl"
FAISS_LC_DIR   = INDEXES_DIR / "faiss_lc"
BM25_LC_FILE   = INDEXES_DIR / "bm25_lc.pkl"
CACHE_FILE     = CACHE_DIR / "query_cache.json"
FEEDBACK_FILE  = LOGS_DIR / "feedback.jsonl"

WIKI_TOPICS = [
    # ── Core AI / ML ──────────────────────────────────────────────────────
    "Artificial intelligence", "Machine learning", "Deep learning",
    "Neural network", "Transformer (machine learning model)",
    "Large language model", "Generative pre-trained transformer",
    "Reinforcement learning", "Generative adversarial network", "Diffusion model",
    "Convolutional neural network", "Recurrent neural network",
    "Long short-term memory", "Backpropagation", "Attention (machine learning)",

    # ── Classical ML ──────────────────────────────────────────────────────
    "Random forest", "Decision tree", "Support vector machine",
    "Gradient boosting", "XGBoost", "K-nearest neighbors algorithm",
    "Logistic regression", "Linear regression", "Naive Bayes classifier",
    "K-means clustering", "Principal component analysis",
    "Overfitting", "Regularization (mathematics)", "Cross-validation (statistics)",
    "Bias–variance tradeoff",

    # ── Optimization ──────────────────────────────────────────────────────
    "Gradient descent", "Stochastic gradient descent",
    "Optimization (mathematics)", "Loss function", "Backpropagation",

    # ── NLP ───────────────────────────────────────────────────────────────
    "Natural language processing", "Word2vec", "BERT (language model)",
    "Tokenization (natural language processing)", "Named-entity recognition",
    "Sentiment analysis", "Machine translation", "Text summarization",

    # ── Computer Vision ───────────────────────────────────────────────────
    "Computer vision", "Object detection", "Image segmentation",
    "Feature extraction", "Generative adversarial network",

    # ── Math & Stats ──────────────────────────────────────────────────────
    "Mathematics", "Calculus", "Linear algebra", "Matrix (mathematics)",
    "Eigenvalues and eigenvectors", "Statistics", "Probability theory",
    "Bayes' theorem", "Normal distribution", "Central limit theorem",
    "Information theory", "Entropy (information theory)",

    # ── Physics ───────────────────────────────────────────────────────────
    "Physics", "Quantum mechanics", "Quantum computing", "Thermodynamics",
    "Statistical mechanics", "Special relativity",

    # ── Computer Science ──────────────────────────────────────────────────
    "Computer science", "Algorithm", "Computational complexity theory",
    "Graph theory", "Cryptography", "Data structure",
    "Sorting algorithm", "Dynamic programming",

    # ── RAG & Search ──────────────────────────────────────────────────────
    "Retrieval-augmented generation", "Knowledge graph",
    "Information retrieval", "Vector database", "Semantic search",

    # ── Other Sciences ────────────────────────────────────────────────────
    "Biology", "DNA", "Evolution", "Astronomy", "Cosmology",
    "Robotics", "Neuroscience",
]
MAX_ARTICLES = 200

# Chunking
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

# Models
EMBED_MODEL    = "BAAI/bge-small-en"
EMBED_DIM      = 384
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = "qwen2.5:14b"
JUDGE_MODEL    = "qwen2.5:14b"

# Retrieval
TOP_K_DENSE       = 10
TOP_K_SPARSE      = 10
TOP_K_RERANK      = 5
RRF_K             = 60
RERANK_THRESHOLD  = -3.0  # chunks below this score are considered irrelevant

# Cache
CACHE_TTL_HOURS = 24
