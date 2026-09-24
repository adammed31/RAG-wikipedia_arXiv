# RAG Wikipedia + arXiv

Ask questions about science. Get answers from Wikipedia and arXiv papers. Runs entirely on your machine.

| Metric | Score |
|---|---|
| Recall@3 | 0.85 |
| Recall@5 | 0.95 |
| Faithfulness | 0.73 |

## Quick start — Docker (recommended)

```bash
git clone https://github.com/adammed31/RAG-wikipedia_arXiv.git
cd RAG-wikipedia_arXiv
docker compose up --build
```

In a second terminal:

```bash
# Pull the LLM (once)
docker compose exec ollama ollama pull llama3.1:8b

# Build the index (once, ~15 min)
docker compose exec app python scripts/build.py --no-pdf
```

Open http://localhost:8501.

## Quick start — Local

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com)

```bash
git clone https://github.com/adammed31/RAG-wikipedia_arXiv.git
cd RAG-wikipedia_arXiv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull qwen2.5:14b        # optional — used as faithfulness judge
ollama pull mistral-small3.1   # optional
python scripts/build.py --no-pdf
streamlit run app/main.py
```

Open http://localhost:8501.

**Build options:**

```bash
python scripts/build.py            # full dataset (with arXiv PDFs)
python scripts/build.py --no-pdf   # arXiv abstracts only (faster)
python scripts/build.py --demo     # small demo (~3 min)
```

## How it works

Each query is routed to one of four retrieval strategies, then reranked before generation:

| Query type | Strategy |
|---|---|
| Simple | Hybrid FAISS + BM25 → RRF |
| Conceptual | HyDE — embeds a hypothetical answer, fused with BM25 via RRF |
| Complex | Query Decomposition — parallel sub-queries merged by RRF |
| Comparative | RAG Fusion — multiple query variants merged by RRF |

All strategies use **Reciprocal Rank Fusion** (Rackauckas 2023) and a **CrossEncoder reranker** (ms-marco-MiniLM).

## Evaluation

Run the full evaluation suite (Recall@K + Faithfulness) on 10 held-out queries:

```bash
python scripts/eval_recall.py --k 1 3 5 --source arxiv --faithfulness
```

Faithfulness is evaluated by `qwen2.5:14b` as LLM-as-judge.

## Features

- **21k chunks** — 200 Wikipedia articles + arXiv foundational papers
- **Hybrid retrieval** — FAISS (dense) + BM25 (sparse) + CrossEncoder reranking
- **Advanced routing** — HyDE, Query Decomposition, RAG Fusion, Query Routing
- **Metrics** — Recall@K, Faithfulness score, latency stats in the sidebar
- **Multi-model** — llama3.1:8b, qwen2.5:14b, mistral-small3.1 with side-by-side comparison
- **Streamlit UI** — streaming, source filter, query cache, debug mode

## Project structure

```
RAG-wikipedia_arXiv/
├── src/
│   ├── ingest.py        # Wikipedia ingestion (MediaWiki API)
│   ├── ingest_arxiv.py  # arXiv ingestion (PDF + abstracts)
│   ├── index.py         # FAISS + BM25 index building
│   ├── retrieve.py      # Retrieval pipeline + routing + RRF
│   ├── generate.py      # LLM generation (streaming)
│   ├── metrics.py       # Recall@K, Faithfulness, latency stats
│   └── config.py        # Configuration
├── app/
│   └── main.py          # Streamlit interface
├── scripts/
│   ├── build.py         # Index build script
│   └── eval_recall.py   # Evaluation script
├── data/                # Runtime files (gitignored)
├── Dockerfile
└── docker-compose.yml
```
