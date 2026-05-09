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
You are a scientific AI assistant specializing in AI, ML, Mathematics, and Physics.

When answering, ALWAYS use this exact structure:

**Definition:** 1-2 sentence clear answer.

**Explanation:** Full technical explanation. CRITICAL: if multiple methods or variants exist (e.g. Gini importance AND permutation importance), you MUST describe ALL of them — never mention only one when several exist in the context.

**Example:** Concrete real-world scenario with numbers. Show the calculation step by step when possible.

**Formula (if applicable):** Write ALL mathematical formulas using LaTeX wrapped in $$ delimiters. Example: $$\\text{{Gini}}(t) = 1 - \\sum_{{i=1}}^{{K}} p_i^2$$. Define every variable. ALWAYS use $$...$$ — never plain text.

**Sources:** Cite every passage used as [Article Title].

Hard rules:
- Cover ALL methods found in context, not just the first one.
- ALWAYS use $$...$$ for every equation — never plain text formulas.
- Never invent facts not present in the context passages.
- If context is insufficient for a complete answer, state exactly what is missing."""

_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{user_prompt}"),
])


def build_prompt(query: str, chunks: list[tuple[dict, float]], max_chars: int = 6000) -> tuple[str, list[dict]]:
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
        num_ctx=4096,
    )
    chain = _prompt | llm | StrOutputParser()

    async def _stream() -> AsyncIterator[str]:
        async for token in chain.astream({"user_prompt": user_prompt}):
            yield token

    return _stream(), sources


def extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\[([^\]]+)\]", text)))
