"""
Build script: Wikipedia + arXiv → chunks → LangChain FAISS + BM25 indexes.
Idempotent: skips steps already done.

Usage:
    python scripts/build.py                  # normal (skips cached steps)
    python scripts/build.py --force          # rebuild everything
    python scripts/build.py --skip-arxiv     # Wikipedia only
    python scripts/build.py --skip-wikipedia # arXiv only
"""
import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from src.ingest import fetch_wikipedia, make_chunks, CHUNK_SIZE, CHUNK_OVERLAP
from src.ingest import ARTICLES_FILE, CHUNKS_FILE
from src.ingest_arxiv import fetch_arxiv, ARXIV_FILE
from src.index import build_indexes
from src.config import RAW_DIR, INDEXES_DIR


def _split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    from src.ingest import _split as split_fn
    return split_fn(text, size, overlap)


def make_arxiv_chunks(papers: list[dict]) -> list[dict]:
    chunks = []
    for paper in papers:
        for i, text in enumerate(_split(paper["text"])):
            if len(text.strip()) < 50:
                continue
            chunks.append({
                "id": f"arxiv_{paper['id']}_{i}",
                "article_id": paper["id"],
                "title": paper["title"],
                "url": paper["url"],
                "source": "arxiv",
                "authors": paper.get("authors", []),
                "published": paper.get("published", ""),
                "categories": paper.get("categories", []),
                "text": text,
            })
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Rebuild everything")
    parser.add_argument("--skip-arxiv", action="store_true", help="Skip arXiv download")
    parser.add_argument("--skip-wikipedia", action="store_true", help="Skip Wikipedia download")
    parser.add_argument("--no-pdf", action="store_true", help="Use abstract only (no PDF download)")
    parser.add_argument("--demo", action="store_true",
                        help="Quick demo mode: 30 Wikipedia articles + 20 foundational arXiv papers (~3 min)")
    args = parser.parse_args()

    # Demo mode overrides
    if args.demo:
        import src.config as _cfg
        _cfg.MAX_ARTICLES = 30
        from src.ingest_arxiv import FOUNDATIONAL_IDS as _fids
        # Patch: only fetch first 20 foundational papers, skip recent
        import src.ingest_arxiv as _ax
        _ax.FOUNDATIONAL_IDS = _fids[:20]
        _ax.CATEGORIES = []   # no recent papers in demo
        print("\n⚡ DEMO MODE — 30 articles + 20 foundational papers (~3 min)")
        args.force = True

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    wiki_chunks:  list[dict] = []
    arxiv_chunks: list[dict] = []

    if not args.skip_wikipedia:
        print("\n=== Step 1/3 — Wikipedia ===")
        if ARTICLES_FILE.exists() and not args.force:
            print("  ↳ Using cached articles")
            with open(ARTICLES_FILE) as f:
                wiki_articles = json.load(f)
        else:
            wiki_articles = fetch_wikipedia()
            with open(ARTICLES_FILE, "w", encoding="utf-8") as f:
                json.dump(wiki_articles, f, ensure_ascii=False, indent=2)
        wiki_chunks = make_chunks(wiki_articles)
        print(f"  ✓ {len(wiki_articles)} articles → {len(wiki_chunks)} chunks")
    else:
        print("\n=== Step 1/3 — Wikipedia (loading from disk) ===")
        if ARTICLES_FILE.exists():
            with open(ARTICLES_FILE) as f:
                wiki_articles = json.load(f)
            wiki_chunks = make_chunks(wiki_articles)
            print(f"  ↳ {len(wiki_chunks)} chunks restored from cache")
        else:
            print("  ↳ No cached Wikipedia articles found, skipping")

    if not args.skip_arxiv:
        print("\n=== Step 2/3 — arXiv ===")
        arxiv_papers = fetch_arxiv(force=args.force, use_pdf=not args.no_pdf)
        arxiv_chunks = make_arxiv_chunks(arxiv_papers)
        print(f"  ✓ {len(arxiv_papers)} papers → {len(arxiv_chunks)} chunks")
    else:
        print("\n=== Step 2/3 — arXiv (loading from disk) ===")
        if ARXIV_FILE.exists():
            from src.ingest_arxiv import _load as load_arxiv
            arxiv_papers = load_arxiv()
            arxiv_chunks = make_arxiv_chunks(arxiv_papers)
            print(f"  ↳ {len(arxiv_chunks)} chunks restored from cache")
        else:
            print("  ↳ No cached arXiv papers found, skipping")

    all_chunks = wiki_chunks + arxiv_chunks

    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(all_chunks, f)

    n_wiki = sum(1 for c in all_chunks if c.get("source") == "wikipedia")
    n_arxiv = sum(1 for c in all_chunks if c.get("source") == "arxiv")
    print(f"\n  Total: {len(all_chunks):,} chunks ({n_wiki:,} Wikipedia · {n_arxiv:,} arXiv)")

    print("\n=== Step 3/3 — Building LangChain FAISS + BM25 indexes ===")
    build_indexes(all_chunks, force=args.force)

    print(f"\n✓ All done! {len(all_chunks):,} chunks indexed.")
    print("  Run: streamlit run app/main.py")
