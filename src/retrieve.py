"""
Retrieval pipeline using LangChain: EnsembleRetriever (FAISS + BM25) + CrossEncoder reranking.
Advanced routing: SIMPLE / COMPLEX (decompose) / CONCEPTUAL (HyDE) / COMPARE (RAG Fusion).
"""
import json
import logging
import re
from enum import Enum

from langchain.retrievers import EnsembleRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS as LangChainFAISS
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import (
    EMBED_MODEL, OLLAMA_MODEL, OLLAMA_URL,
    RERANK_MODEL, RERANK_THRESHOLD, TOP_K_DENSE, TOP_K_RERANK, TOP_K_SPARSE,
)

log = logging.getLogger(__name__)
BGE_PREFIX = "Represent this sentence for searching relevant passages: "

_embedder: SentenceTransformer | None = None
_reranker: CrossEncoder | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        log.info(f"Loading embedder: {EMBED_MODEL}")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        log.info(f"Loading reranker: {RERANK_MODEL}")
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


class QueryType(str, Enum):
    SIMPLE     = "simple"
    COMPLEX    = "complex"
    CONCEPTUAL = "conceptual"
    COMPARE    = "compare"


def route(query: str) -> QueryType:
    q = query.lower().strip()
    if len(q.split()) <= 3:
        return QueryType.SIMPLE
    if re.search(r"\b(vs|versus|compare|difference between)\b", q):
        return QueryType.COMPARE
    if re.search(r"^(what is|define|explain|describe|how does)\b", q):
        return QueryType.CONCEPTUAL
    if re.search(r"\b(and|also|both|additionally|furthermore)\b", q) and len(q.split()) > 12:
        return QueryType.COMPLEX
    return QueryType.SIMPLE


async def _ollama_generate(prompt: str, system: str = "", model: str = OLLAMA_MODEL) -> str:
    llm = ChatOllama(model=model, base_url=OLLAMA_URL, temperature=0.1, num_ctx=4096)
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    response = await llm.ainvoke(messages)
    return response.content


async def hyde_vec(query: str, model: str = OLLAMA_MODEL) -> list[float]:
    """Generate a hypothetical document and return its embedding (no query prefix)."""
    system = "Write a short encyclopedic passage that directly answers the question."
    try:
        hypothetical = await _ollama_generate(query, system=system, model=model)
        if hypothetical.strip():
            return get_embedder().encode(hypothetical, normalize_embeddings=True).tolist()
    except Exception as e:
        log.warning(f"HyDE failed: {e}")
    return get_embedder().encode(BGE_PREFIX + query, normalize_embeddings=True).tolist()


async def decompose(query: str, model: str = OLLAMA_MODEL) -> list[str]:
    system = "Return ONLY a JSON array of 2-4 simpler sub-questions that together answer the original. No explanation."
    prompt = f'Decompose: "{query}"'
    try:
        raw = await _ollama_generate(prompt, system=system, model=model)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            subs = json.loads(match.group())
            if isinstance(subs, list) and subs:
                return [s for s in subs if isinstance(s, str)][:4]
    except Exception as e:
        log.warning(f"Decomposition failed: {e}")
    return [query]


async def rag_fusion_queries(query: str, model: str = OLLAMA_MODEL) -> list[str]:
    system = "Return ONLY a JSON array of 3 different search queries for the same information. No explanation."
    prompt = f'Generate alternatives for: "{query}"'
    try:
        raw = await _ollama_generate(prompt, system=system, model=model)
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            variants = json.loads(match.group())
            if isinstance(variants, list):
                return [query] + [v for v in variants if isinstance(v, str)][:3]
    except Exception as e:
        log.warning(f"RAG Fusion variant generation failed: {e}")
    return [query]


async def retrieve(
    query: str,
    lc_faiss: LangChainFAISS,
    bm25_retriever: BM25Retriever,
    chunk_store: dict[str, dict],
    source_filter: str = "both",
    top_k: int = TOP_K_DENSE,
    model: str = OLLAMA_MODEL,
) -> tuple[list[tuple[dict, float]], str, list[str]]:
    qtype = route(query)
    log.info(f"Query type: {qtype} — '{query[:60]}'")
    sub_queries = [query]

    # Fetch more candidates when filtering by source so post-filter has enough
    fetch_k = top_k * 3 if source_filter != "both" else top_k

    faiss_retriever = lc_faiss.as_retriever(search_kwargs={"k": fetch_k})
    bm25_retriever.k = fetch_k
    ensemble = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    def _docs_to_chunks(docs) -> list[tuple[dict, float]]:
        seen: set[str] = set()
        result = []
        for doc in docs:
            cid = doc.metadata.get("id")
            if not cid or cid in seen or cid not in chunk_store:
                continue
            seen.add(cid)
            chunk = chunk_store[cid]
            if source_filter != "both" and chunk.get("source") != source_filter:
                continue
            result.append((chunk, 1.0))
        return result[:top_k]

    def _merge_docs(all_docs) -> list:
        seen: set[str] = set()
        unique = []
        for doc in all_docs:
            cid = doc.metadata.get("id")
            if cid not in seen:
                seen.add(cid)
                unique.append(doc)
        return unique

    if qtype == QueryType.COMPLEX:
        sub_queries = await decompose(query, model=model)
        all_docs = []
        for sq in sub_queries:
            all_docs.extend(await ensemble.ainvoke(sq))
        candidates = _docs_to_chunks(_merge_docs(all_docs))

    elif qtype == QueryType.COMPARE:
        variants = await rag_fusion_queries(query, model=model)
        sub_queries = variants
        all_docs = []
        for v in variants:
            all_docs.extend(await ensemble.ainvoke(v))
        candidates = _docs_to_chunks(_merge_docs(all_docs))

    elif qtype == QueryType.CONCEPTUAL:
        # HyDE: embed a hypothetical answer, search by the resulting vector
        vec = await hyde_vec(query, model=model)
        docs = lc_faiss.similarity_search_by_vector(vec, k=fetch_k)
        candidates = _docs_to_chunks(docs)

    else:
        docs = await ensemble.ainvoke(query)
        candidates = _docs_to_chunks(docs)

    if len(candidates) > TOP_K_RERANK:
        pairs = [(query, c["text"]) for c, _ in candidates]
        scores = get_reranker().predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        chunk_list = [
            (c, float(s)) for (c, _), s in ranked[:TOP_K_RERANK]
            if float(s) >= RERANK_THRESHOLD
        ]
    else:
        chunk_list = candidates[:TOP_K_RERANK]

    return chunk_list, qtype.value, sub_queries
