# RAG Wikipedia + arXiv — 100% Local, 100% Free

> Advanced RAG system over Wikipedia + arXiv (~21 000 chunks). No cloud. No API keys. No databases. Just files.
> Built with **LangChain** (FAISS vectorstore · BM25Retriever · EnsembleRetriever · ChatOllama · LCEL).

## ⚡ Quick start

```bash
git clone https://github.com/adammed31/RAG-wikipedia_arXiv.git
cd RAG-wikipedia_arXiv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
ollama pull mistral-small3.1
python scripts/build.py && streamlit run app/main.py
```

Open **http://localhost:8501** → ready to query 21 000+ chunks (Wikipedia + arXiv).

> First run downloads ~400 MB of HuggingFace models automatically (BAAI/bge-small-en embeddings + ms-marco-MiniLM reranker).

---

## 🔧 Build options

```bash
python scripts/build.py              # full dataset (Wikipedia + arXiv PDFs, ~20-30 min)
python scripts/build.py --demo       # 30 articles + 20 papers, ~3 min
python scripts/build.py --no-pdf     # arXiv abstracts only (faster)
```

---

## Architecture

```
Question
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  Query Router (LangChain)                           │
│  simple → ensemble  │  conceptual → HyDE            │
│  complex → decompose + ensemble  │  compare → fusion│
└─────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────┐   ┌──────────────────────┐
│  LangChain FAISS         │   │  LangChain           │
│  (dense vectors)         │   │  BM25Retriever       │
│  BAAI/bge-small-en       │   │  (keyword search)    │
└──────────────┬───────────┘   └──────────┬───────────┘
               │  EnsembleRetriever (RRF) │
               └──────────────┬───────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Cross-encoder  │
                    │  ms-marco-MiniLM│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  ChatOllama     │
                    │  + LCEL chain   │
                    │  (local, Q4)    │
                    └────────┬────────┘
                             │
                             ▼
                    Streaming answer + sources
```

### Storage — 100% files, zero databases

| What | Where |
|------|-------|
| Wikipedia articles | `data/raw/articles.json` |
| arXiv papers | `data/raw/arxiv_papers.json.gz` |
| LangChain FAISS index | `data/indexes/faiss_lc/` |
| BM25 retriever | `data/indexes/bm25_lc.pkl` |
| Chunks | `data/indexes/chunks.pkl` |
| Query cache | `data/cache/query_cache.json` |
| Query logs | `data/logs/queries.jsonl` |
| Feedback | `data/logs/feedback.jsonl` |

---

## Installation

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed and running.

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
ollama pull mistral-small3.1
python scripts/build.py && streamlit run app/main.py
```

---

## Usage

Go to **http://localhost:8501** and ask any scientific question:

- *"What is quantum entanglement?"*
- *"Compare supervised vs unsupervised learning"*
- *"How does the attention mechanism work in Transformers?"*

The sidebar lets you:
- **Source filter** — search Wikipedia only, arXiv only, or both
- **Model selector** — switch between installed Ollama models
- **Compare mode** — run the same query on two models simultaneously
- **Query cache** — instant replay of previous queries
- **Debug mode** — show query type and sub-queries

---

## LangChain stack

| Component | Where used |
|-----------|-----------|
| `RecursiveCharacterTextSplitter` | `src/ingest.py` — chunk Wikipedia + arXiv |
| `ChatOllama + ChatPromptTemplate + LCEL` | `src/generate.py` — streaming answer |
| `ChatOllama` | `src/retrieve.py` — HyDE, decomposition, RAG Fusion |
| `FAISS` vectorstore + `BGEEmbeddings` | `src/index.py` — dense index |
| `BM25Retriever + EnsembleRetriever` | `src/retrieve.py` — hybrid retrieval |

---

## RAG Techniques

| Technique | Trigger | What it does |
|-----------|---------|--------------|
| **Hybrid search** | Always | EnsembleRetriever: FAISS (dense) + BM25 (sparse) fused with RRF |
| **Reranking** | Always | Cross-encoder rescores top-20 → keeps top-5 |
| **Query Routing** | Always | Classifies query → picks best strategy |
| **HyDE** | Conceptual queries | Embeds a hypothetical answer, not the question |
| **Query Decomposition** | Complex queries | Breaks into sub-queries, merges results |
| **RAG Fusion** | Comparative queries | 3 query variants → RRF merge |

---

## Project structure

```
RAG-wikipedia_arXiv/
├── src/
│   ├── config.py          # All settings in one place
│   ├── ingest.py          # Wikipedia download + chunking (RecursiveCharacterTextSplitter)
│   ├── ingest_arxiv.py    # arXiv download + PDF extraction
│   ├── index.py           # LangChain FAISS + BM25Retriever build/load
│   ├── retrieve.py        # EnsembleRetriever + all RAG techniques
│   ├── generate.py        # ChatOllama + LCEL streaming
│   ├── cache.py           # File-based query cache
│   └── logger.py          # JSONL query + feedback logs
├── app/
│   └── main.py            # Streamlit chat interface
├── scripts/
│   ├── build.py           # One-shot index builder (Wikipedia + arXiv)
│   └── build_micro.py     # Add specific Wikipedia articles to existing index
├── tests/
│   └── test_retrieval.py
├── data/                  # All runtime files (gitignored)
└── requirements.txt
```

---

## Configuration

All settings live in `src/config.py`. Key knobs:

```python
MAX_ARTICLES  = 500      # Wikipedia articles
CHUNK_SIZE    = 512      # Characters per chunk
TOP_K_DENSE   = 20       # FAISS candidates
TOP_K_RERANK  = 5        # Final chunks sent to LLM
RERANK_THRESHOLD = -3.0  # Discard irrelevant chunks (no hallucination on out-of-scope)
OLLAMA_MODEL  = "llama3.1:8b"  # Swap for qwen2.5:14b, mistral-small3.1, etc.
```

---

## Running tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## Troubleshooting

**Ollama not reachable**
```bash
curl http://localhost:11434/api/tags
# If not running: ollama serve
```

**Index not found**
```bash
python scripts/build.py
```

**Out of memory during indexing**
Reduce `MAX_ARTICLES` in `src/config.py` (try 100 for a quick test).

**Slow first query**
Models load on first use (~5s). Subsequent queries are fast.
