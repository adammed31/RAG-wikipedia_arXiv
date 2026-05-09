"""
Download Wikipedia articles and split into chunks.
arXiv is handled separately by src/ingest_arxiv.py.
Output: data/raw/articles.json
"""
import logging
import re
import time
import unicodedata

import wikipedia
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


def fetch_wikipedia(topics: list[str] = WIKI_TOPICS, max_total: int = MAX_ARTICLES) -> list[dict]:
    wikipedia.set_lang("en")
    seen_titles: set[str] = set()
    articles = []

    for topic in tqdm(topics, desc="Wikipedia topics"):
        if len(articles) >= max_total:
            break
        time.sleep(2)
        try:
            search_results = wikipedia.search(topic, results=30)
        except Exception as e:
            log.warning(f"Search failed for '{topic}': {e}")
            continue

        for title in search_results:
            if len(articles) >= max_total:
                break
            if title in seen_titles:
                continue
            seen_titles.add(title)
            try:
                page = wikipedia.page(title, auto_suggest=False)
                text = _clean(page.content)
                if len(text) < 300:
                    continue
                articles.append({
                    "id": str(page.pageid),
                    "title": page.title,
                    "text": text,
                    "url": page.url,
                    "source": "wikipedia",
                })
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
