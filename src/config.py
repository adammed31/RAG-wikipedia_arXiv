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
    "Artificial intelligence", "Machine learning", "Deep learning",
    "Neural network", "Physics", "Quantum mechanics", "Thermodynamics",
    "Mathematics", "Calculus", "Linear algebra", "Statistics",
    "Chemistry", "Biology", "DNA", "Evolution", "Astronomy",
    "Computer science", "Algorithm", "Cryptography",
]
MAX_ARTICLES = 500

# Chunking
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

# Models
EMBED_MODEL    = "BAAI/bge-small-en"
EMBED_DIM      = 384
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_URL     = "http://localhost:11434"
OLLAMA_MODEL   = "llama3.1:8b"

# Retrieval
TOP_K_DENSE       = 20
TOP_K_SPARSE      = 20
TOP_K_RERANK      = 5
RRF_K             = 60
RERANK_THRESHOLD  = -3.0  # chunks below this score are considered irrelevant

# Cache
CACHE_TTL_HOURS = 24
