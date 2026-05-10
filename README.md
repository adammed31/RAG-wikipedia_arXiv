# RAG Wikipedia + arXiv

Ask questions about science. Get answers from Wikipedia and arXiv papers. Runs entirely on your machine.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed

## Install

```bash
git clone https://github.com/adammed31/RAG-wikipedia_arXiv.git
cd RAG-wikipedia_arXiv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
ollama pull mistral-small3.1
python scripts/build.py
streamlit run app/main.py
```

Open http://localhost:8501 and start asking questions.

Quick options:

```bash
python scripts/build.py            # full dataset
python scripts/build.py --demo      # small demo dataset
python scripts/build.py --no-pdf     # arXiv abstracts only
```

## How it works

Your question goes through hybrid search (keyword + semantic), gets reranked, then answered by a local LLM.

Different question types trigger different strategies:
- Simple → straight retrieval
- Conceptual → HyDE (embeds a hypothetical answer)
- Complex → breaks into sub-questions
- Comparative → multiple query variants


## Usage

Open http://localhost:8501 and ask scientific questions like:

- How does the attention mechanism work in transformers?
- Compare supervised vs unsupervised learning
- What is reinforcement learning?

Use the sidebar to filter sources, switch models, or compare answers side-by-side.

## Features

- 21,000 chunks from 500 Wikipedia articles + arXiv papers
- Hybrid retrieval (FAISS + BM25)
- Local generation (Ollama with llama3.1, qwen2.5, or mistral-small3.1)
- Streamlit UI with query cache and debug mode

## Project structure

RAG-wikipedia_arXiv/
├── src/          # Core logic (ingest, index, retrieve, generate)
├── app/          # Streamlit interface
├── scripts/      # Build tools
├── tests/        # Tests
└── data/         # Runtime files (gitignored)

See [ARCHITECTURE.md](ARCHITECTURE.md) for technical details.
