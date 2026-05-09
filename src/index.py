"""
Build LangChain FAISS vectorstore (dense) + BM25Retriever (sparse) from chunks.pkl.
Uses LangChain: FAISS vectorstore + BM25Retriever.
Outputs: data/indexes/faiss_lc/  (LangChain FAISS)
         data/indexes/bm25_lc.pkl (BM25Retriever)
"""
import logging
import pickle

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    BM25_LC_FILE, CHUNKS_FILE, EMBED_MODEL, FAISS_LC_DIR,
    INDEXES_DIR, TOP_K_SPARSE,
)

log = logging.getLogger(__name__)

BGE_PREFIX = "Represent this sentence for searching relevant passages: "

_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        log.info(f"Loading embedding model: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


class BGEEmbeddings(Embeddings):
    """LangChain Embeddings adapter for BGE model with query prefix."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _get_embedder().encode(
            texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        return _get_embedder().encode(
            BGE_PREFIX + text, normalize_embeddings=True
        ).tolist()


def _chunks_to_documents(chunks: list[dict]) -> list[Document]:
    return [
        Document(
            page_content=c["text"],
            metadata={k: v for k, v in c.items() if k != "text"},
        )
        for c in tqdm(chunks, desc="Preparing documents")
    ]


def build_indexes(chunks: list[dict], force: bool = False) -> None:
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)

    documents = _chunks_to_documents(chunks)
    embeddings = BGEEmbeddings()

    faiss_index_file = FAISS_LC_DIR / "index.faiss"
    if not faiss_index_file.exists() or force:
        log.info(f"Building LangChain FAISS vectorstore ({len(documents)} docs)…")
        lc_faiss = LangChainFAISS.from_documents(documents, embeddings)
        FAISS_LC_DIR.mkdir(parents=True, exist_ok=True)
        lc_faiss.save_local(str(FAISS_LC_DIR))
        log.info(f"LangChain FAISS saved → {FAISS_LC_DIR}")
    else:
        log.info("LangChain FAISS already exists, skipping.")

    if not BM25_LC_FILE.exists() or force:
        log.info(f"Building BM25 retriever ({len(documents)} docs)…")
        bm25_retriever = BM25Retriever.from_documents(documents, k=TOP_K_SPARSE)
        with open(BM25_LC_FILE, "wb") as f:
            pickle.dump(bm25_retriever, f)
        log.info(f"BM25 retriever saved → {BM25_LC_FILE}")
    else:
        log.info("BM25 retriever already exists, skipping.")


def load_indexes() -> tuple[LangChainFAISS, BM25Retriever]:
    log.info(f"Loading LangChain FAISS from {FAISS_LC_DIR}")
    embeddings = BGEEmbeddings()
    lc_faiss = LangChainFAISS.load_local(
        str(FAISS_LC_DIR), embeddings, allow_dangerous_deserialization=True
    )
    with open(BM25_LC_FILE, "rb") as f:
        bm25_retriever = pickle.load(f)
    n_docs = lc_faiss.index.ntotal
    log.info(f"LangChain FAISS: {n_docs} vectors | BM25: {len(bm25_retriever.docs)} docs")
    return lc_faiss, bm25_retriever


def load_chunks() -> dict[str, dict]:
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return {c["id"]: c for c in chunks}
