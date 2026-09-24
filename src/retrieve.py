"""
Retrieval pipeline using LangChain: FAISS (dense) + BM25 (sparse) + CrossEncoder reranking.
Advanced routing: SIMPLE / COMPLEX (decompose) / CONCEPTUAL (HyDE) / COMPARE (RAG Fusion).
All strategies use Reciprocal Rank Fusion (Rackauckas 2023) for merging ranked lists.
Per-step latency is returned for observability.
"""
import asyncio
import json
import logging
import re
import time
from collections import defaultdict
from enum import Enum

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
    if re.search(r"\b(what is|define|explain|describe|how does)\b", q):
        return QueryType.CONCEPTUAL
    if re.search(r"\b(and|also|both|additionally|furthermore)\b", q) and len(q.split()) > 12:
        return QueryType.COMPLEX
    return QueryType.SIMPLE


async def _ollama_generate(prompt: str, system: str = "", model: str = OLLAMA_MODEL) -> str:
    llm = ChatOllama(model=model, base_url=OLLAMA_URL, temperature=0.1, num_ctx=2048)
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))
    response = await llm.ainvoke(messages)
    return response.content


async def hyde_vec(query: str, model: str = OLLAMA_MODEL) -> list[float]:
    """Generate a hypothetical document and return its embedding (HyDE — Gao et al. 2022)."""
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
) -> tuple[list[tuple[dict, float]], str, list[str], dict[str, float]]:
    """
    Returns (chunks, query_type, sub_queries, timing).
    timing keys: strategy_ms, faiss_ms, bm25_ms, retrieval_ms, rerank_ms.
    """
    timing: dict[str, float] = {
        "strategy_ms": 0.0,   # LLM call: HyDE / decompose / rag_fusion
        "faiss_ms":    0.0,   # FAISS similarity search
        "bm25_ms":     0.0,   # BM25 keyword search
        "retrieval_ms": 0.0,  # total retrieval for COMPLEX/COMPARE (parallel sub-queries)
        "rerank_ms":   0.0,   # CrossEncoder reranking
    }

    qtype = route(query)
    log.info(f"Query type: {qtype} — '{query[:60]}'")
    sub_queries = [query]

    fetch_k = top_k * 3 if source_filter != "both" else top_k
    faiss_retriever = lc_faiss.as_retriever(search_kwargs={"k": fetch_k})
    bm25_retriever.k = fetch_k

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

    def _rrf(rankings: list[list], k: int = 60) -> list:
        """Reciprocal Rank Fusion — Rackauckas 2023 / Cormack et al. 2009."""
        scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, object] = {}
        for ranking in rankings:
            for rank, doc in enumerate(ranking):
                cid = doc.metadata.get("id")
                if not cid:
                    continue
                scores[cid] += 1.0 / (k + rank + 1)
                if cid not in doc_map:
                    doc_map[cid] = doc
        return [doc_map[cid] for cid in sorted(scores, key=scores.__getitem__, reverse=True)]

    if qtype == QueryType.COMPLEX:
        t = time.perf_counter()
        sub_queries = await decompose(query, model=model)
        timing["strategy_ms"] = (time.perf_counter() - t) * 1000

        async def _retrieve_sub(q: str) -> list:
            f = await faiss_retriever.ainvoke(q)
            b = await bm25_retriever.ainvoke(q)
            return _rrf([f, b])

        t = time.perf_counter()
        results = await asyncio.gather(*[_retrieve_sub(sq) for sq in sub_queries])
        candidates = _docs_to_chunks(_rrf(list(results)))
        timing["retrieval_ms"] = (time.perf_counter() - t) * 1000

    elif qtype == QueryType.COMPARE:
        t = time.perf_counter()
        variants = await rag_fusion_queries(query, model=model)
        sub_queries = variants
        timing["strategy_ms"] = (time.perf_counter() - t) * 1000

        async def _retrieve_variant(v: str) -> list:
            f = await faiss_retriever.ainvoke(v)
            b = await bm25_retriever.ainvoke(v)
            return _rrf([f, b])

        t = time.perf_counter()
        results = await asyncio.gather(*[_retrieve_variant(v) for v in variants])
        candidates = _docs_to_chunks(_rrf(list(results)))
        timing["retrieval_ms"] = (time.perf_counter() - t) * 1000

    elif qtype == QueryType.CONCEPTUAL:
        # HyDE: FAISS on hypothetical doc + BM25 on original query, merged by RRF
        t = time.perf_counter()
        vec = await hyde_vec(query, model=model)
        timing["strategy_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        faiss_docs = lc_faiss.similarity_search_by_vector(vec, k=fetch_k)
        timing["faiss_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        bm25_docs = await bm25_retriever.ainvoke(query)
        timing["bm25_ms"] = (time.perf_counter() - t) * 1000

        candidates = _docs_to_chunks(_rrf([faiss_docs, bm25_docs]))

    else:  # SIMPLE
        t = time.perf_counter()
        faiss_docs = await faiss_retriever.ainvoke(query)
        timing["faiss_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        bm25_docs = await bm25_retriever.ainvoke(query)
        timing["bm25_ms"] = (time.perf_counter() - t) * 1000

        candidates = _docs_to_chunks(_rrf([faiss_docs, bm25_docs]))

    if len(candidates) > TOP_K_RERANK:
        t = time.perf_counter()
        pairs = [(query, c["text"]) for c, _ in candidates]
        reranker = get_reranker()
        scores = await asyncio.to_thread(reranker.predict, pairs)
        timing["rerank_ms"] = (time.perf_counter() - t) * 1000
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        chunk_list = [
            (c, float(s)) for (c, _), s in ranked[:TOP_K_RERANK]
            if float(s) >= RERANK_THRESHOLD
        ]
    else:
        chunk_list = candidates[:TOP_K_RERANK]

    return chunk_list, qtype.value, sub_queries, timing
