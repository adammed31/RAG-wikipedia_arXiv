"""
Fetch specific Wikipedia articles by exact title and add them to the existing index.
Does NOT re-download everything — appends to chunks.pkl and rebuilds indexes.

Usage:
    python scripts/build_micro.py
    python scripts/build_micro.py --force   # re-fetch even if already in index
"""
import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

import wikipedia
from src.config import CHUNKS_FILE, INDEXES_DIR, RAW_DIR
from src.ingest import _clean, _split
from src.index import build_indexes

MICRO_TITLES = [
    # Feature importance & selection
    "Feature (machine learning)",
    "Feature engineering",
    "Feature selection",
    "Feature extraction",
    "Dimensionality reduction",
    "Random forest",                    # covers Gini importance
    "Decision tree",                    # covers Gini impurity formula
    "Information gain in decision trees",
    "Mutual information",
    # Statistics for ML
    "Variance",
    "Correlation",
    "Mean squared error",
    "Coefficient of determination",     # R-squared
    "Precision and recall",
    "F-score",
    # Optimization
    "Gradient descent",
    "Stochastic gradient descent",
    "Backpropagation",
    # Core ML
    "Bias–variance tradeoff",
    "Overfitting",
    "Cross-validation (statistics)",
    "Regularization (mathematics)",
]


def _fetch_one(title: str, retries: int = 3) -> dict | None:
    for attempt in range(retries):
        try:
            time.sleep(3 + attempt * 2)   # 3s, 5s, 7s
            page = wikipedia.page(title, auto_suggest=False)
            text = _clean(page.content)
            if len(text) < 200:
                return None
            return {
                "id": str(page.pageid),
                "title": page.title,
                "text": text,
                "url": page.url,
                "source": "wikipedia",
            }
        except Exception as e:
            log.warning(f"  Attempt {attempt+1}/{retries} failed for '{title}': {e}")
    return None


def _make_chunks(article: dict) -> list[dict]:
    chunks = []
    for i, text in enumerate(_split(article["text"])):
        if len(text.strip()) < 50:
            continue
        chunks.append({
            "id": f"{article['id']}_{i}",
            "article_id": article["id"],
            "title": article["title"],
            "url": article["url"],
            "source": article.get("source", "wikipedia"),
            "text": text,
        })
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fetch even if title already in index")
    args = parser.parse_args()

    wikipedia.set_lang("en")
    wikipedia.set_user_agent("RAG-Wikipedia/1.0 (educational project; adam.medb@gmail.com)")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    existing_chunks: list[dict] = []
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE, "rb") as f:
            existing_chunks = pickle.load(f)
    existing_titles = {c["title"] for c in existing_chunks}
    log.info(f"Existing index: {len(existing_chunks)} chunks, {len(existing_titles)} unique titles")

    new_chunks: list[dict] = []
    for title in MICRO_TITLES:
        if not args.force and title in existing_titles:
            log.info(f"  ✓ Already indexed: {title}")
            continue
        log.info(f"  Fetching: {title}")
        article = _fetch_one(title)
        if article:
            chunks = _make_chunks(article)
            new_chunks.extend(chunks)
            log.info(f"    → {len(chunks)} chunks")
        time.sleep(1)   # rate limiting

    if not new_chunks:
        log.info("Nothing new to add.")
        sys.exit(0)

    all_chunks = existing_chunks + new_chunks
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    n_wiki  = sum(1 for c in all_chunks if c.get("source") == "wikipedia")
    n_arxiv = sum(1 for c in all_chunks if c.get("source") == "arxiv")
    log.info(f"Total: {len(all_chunks)} chunks ({n_wiki} Wikipedia · {n_arxiv} arXiv)")
    log.info(f"Added: {len(new_chunks)} new chunks from micro build")

    print("\n=== Rebuilding LangChain FAISS + BM25 ===")
    build_indexes(all_chunks, force=True)
    print(f"\n✓ Done! {len(all_chunks):,} chunks indexed.")
    print("  Run: streamlit run app/main.py")
