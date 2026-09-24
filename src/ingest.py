"""
Download Wikipedia articles and split into chunks.
arXiv is handled separately by src/ingest_arxiv.py.
Output: data/raw/articles.json
"""
import logging
import re
import time
import unicodedata

import requests
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    ARTICLES_FILE, CHUNKS_FILE, RAW_DIR,
    WIKI_TOPICS, MAX_ARTICLES, CHUNK_SIZE, CHUNK_OVERLAP,
)

log = logging.getLogger(__name__)

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    return _text_splitter.split_text(text)


def _clean(text: str) -> str:
    text = re.sub(r"==+[^=]+=+", "\n", text)
    text = re.sub(r"\{\{[^}]*\}\}", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_WP_API = "https://en.wikipedia.org/w/api.php"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "RAG-educational-project/1.0 (https://github.com; educational use)"})


def _wp_search(topic: str, limit: int = 15) -> list[str]:
    resp = _SESSION.get(_WP_API, params={
        "action": "query", "list": "search",
        "srsearch": topic, "srlimit": limit, "format": "json",
    }, timeout=10)
    resp.raise_for_status()
    return [r["title"] for r in resp.json()["query"]["search"]]


def _wp_fetch(title: str) -> dict | None:
    resp = _SESSION.get(_WP_API, params={
        "action": "query", "titles": title,
        "prop": "extracts|info", "explaintext": 1,
        "inprop": "url", "format": "json",
    }, timeout=15)
    resp.raise_for_status()
    pages = resp.json()["query"]["pages"]
    page = next(iter(pages.values()))
    if "missing" in page or not page.get("extract"):
        return None
    return {
        "id": str(page["pageid"]),
        "title": page["title"],
        "text": _clean(page["extract"]),
        "url": f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}",
        "source": "wikipedia",
    }


def fetch_wikipedia(topics: list[str] = WIKI_TOPICS, max_total: int = MAX_ARTICLES) -> list[dict]:
    seen_titles: set[str] = set()
    articles = []

    for topic in tqdm(topics, desc="Wikipedia topics"):
        if len(articles) >= max_total:
            break
        time.sleep(1)
        try:
            titles = _wp_search(topic, limit=15)
        except Exception as e:
            log.warning(f"Search failed for '{topic}': {e}")
            time.sleep(5)
            continue

        for title in titles:
            if len(articles) >= max_total:
                break
            if title in seen_titles:
                continue
            seen_titles.add(title)
            try:
                time.sleep(0.5)
                article = _wp_fetch(title)
                if article and len(article["text"]) >= 300:
                    articles.append(article)
            except Exception:
                continue

    log.info(f"Wikipedia: fetched {len(articles)} articles")
    return articles


def make_chunks(articles: list[dict]) -> list[dict]:
    chunks = []
    for art in articles:
        for i, text in enumerate(_split(art["text"])):
            if len(text.strip()) < 50:
                continue
            chunks.append({
                "id": f"{art['id']}_{i}",
                "article_id": art["id"],
                "title": art["title"],
                "url": art["url"],
                "source": art.get("source", "wikipedia"),
                "text": text,
            })
    log.info(f"Created {len(chunks)} chunks from {len(articles)} documents")
    return chunks
