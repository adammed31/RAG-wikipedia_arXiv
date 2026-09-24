"""
Build prompt from retrieved chunks and stream answer via Ollama.
Uses LangChain: ChatOllama + ChatPromptTemplate + LCEL (| chain) + StrOutputParser.
"""
import re
from typing import AsyncIterator

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """\
You are a scientific AI assistant. You must answer ONLY using the context passages provided. \
Never use your training knowledge — if the context does not contain the information, say so explicitly.

Structure your answer as follows:

**Definition:** 1-2 sentence answer taken directly from the context.

**Explanation:** Technical explanation using ONLY facts stated in the context. Cover ALL methods or variants mentioned in the context.

**Example (if in context):** Include an example ONLY if it appears explicitly in the context. Do NOT invent examples.

**Formula (if in context):** Write formulas in $$LaTeX$$ ONLY if they appear in the context. Do NOT invent formulas.

**Sources:** Cite every passage used as [Article Title].

Hard rules:
- ONLY use information explicitly present in the context. Zero outside knowledge.
- If context is insufficient, write: "The context does not contain enough information to answer this fully."
- ALWAYS use $$...$$ for equations — never plain text math.
- Never cite or reference sources not present in the context."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{user_prompt}"),
])


def build_prompt(query: str, chunks: list[tuple[dict, float]], max_chars: int = 3500) -> tuple[str, list[dict]]:
    context_parts: list[str] = []
    sources: list[dict] = []
    used = 0

    for chunk, score in chunks:
        passage = f"[{chunk['title']}]\n{chunk['text']}"
        if used + len(passage) > max_chars:
            break
        context_parts.append(passage)
        sources.append({
            "title":      chunk["title"],
            "url":        chunk["url"],
            "excerpt":    chunk["text"][:250],
            "score":      round(score, 4),
            "source":     chunk.get("source", "wikipedia"),
            "authors":    chunk.get("authors", []),
            "published":  chunk.get("published", ""),
            "categories": chunk.get("categories", []),
        })
        used += len(passage)

    context = "\n\n---\n\n".join(context_parts)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    return prompt, sources


async def stream_answer(
    query: str, chunks: list[tuple[dict, float]], model: str | None = None
) -> tuple[AsyncIterator[str], list[dict]]:
    from src.config import OLLAMA_MODEL, OLLAMA_URL
    user_prompt, sources = build_prompt(query, chunks)
    llm = ChatOllama(
        model=model or OLLAMA_MODEL,
        base_url=OLLAMA_URL,
        temperature=0.1,
        num_ctx=2048,
    )
    chain = _prompt | llm | StrOutputParser()

    async def _stream() -> AsyncIterator[str]:
        async for token in chain.astream({"user_prompt": user_prompt}):
            yield token

    return _stream(), sources


def extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[([^\]]+)\]", text)))
