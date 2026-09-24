"""Evaluation metrics: Recall@K, faithfulness (LLM-as-judge), latency stats."""
import json
import re
import statistics
from pathlib import Path

from src.config import JUDGE_MODEL, LOGS_DIR, OLLAMA_MODEL, OLLAMA_URL


def recall_at_k(retrieved_titles: list[str], relevant_titles: list[str]) -> float:
    """Fraction of relevant titles found in the retrieved set (case-insensitive)."""
    if not relevant_titles:
        return 0.0
    retrieved_lower = {t.lower() for t in retrieved_titles}
    hits = sum(1 for t in relevant_titles if t.lower() in retrieved_lower)
    return hits / len(relevant_titles)


async def faithfulness_score(
    context_chunks: list[tuple[dict, float]],
    answer: str,
    model: str = JUDGE_MODEL,
) -> float:
    """
    LLM-as-judge: fraction of answer claims supported by the retrieved context.
    Returns a score in [0.0, 1.0].
    """
    from langchain_community.chat_models import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage

    context = "\n\n".join(c["text"] for c, _ in context_chunks[:5])
    system = (
        "You are a strict faithfulness evaluator. "
        "Your task: for each factual claim in the answer, check if it is explicitly supported by the context. "
        "A claim is supported ONLY if the context contains that exact information. "
        "If a claim uses outside knowledge not present in the context, it is NOT supported. "
        "Compute: supported_claims / total_claims. "
        "Reply with ONLY a decimal between 0.0 and 1.0. No explanation, no text — just the number."
    )
    prompt = (
        f"Context passages:\n{context}\n\n"
        f"Answer to evaluate:\n{answer}\n\n"
        f"For each claim in the answer, verify it appears in the context. "
        f"Score (supported/total):"
    )

    try:
        llm = ChatOllama(model=model, base_url=OLLAMA_URL, temperature=0.0, num_ctx=2048)
        resp = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        match = re.search(r"\b([01](?:\.\d*)?|\d*\.\d+)\b", resp.content.strip())
        if match:
            return max(0.0, min(1.0, float(match.group())))
    except Exception:
        pass
    return 0.0


def latency_stats() -> dict:
    """Compute avg/p50/p95 latency (ms) from queries.jsonl, excluding cached responses."""
    path = LOGS_DIR / "queries.jsonl"
    if not path.exists():
        return {}

    retrieve_ms: list[int] = []
    generate_ms: list[int] = []
    total_ms: list[int] = []
    ttft_ms: list[float] = []
    faithfulness_scores: list[float] = []
    steps: dict[str, list[float]] = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("cached"):
                    continue
                if "retrieve_ms" in r:
                    retrieve_ms.append(r["retrieve_ms"])
                if "generate_ms" in r:
                    generate_ms.append(r["generate_ms"])
                if "latency_ms" in r:
                    total_ms.append(r["latency_ms"])
                if "ttft_ms" in r and r["ttft_ms"]:
                    ttft_ms.append(r["ttft_ms"])
                if "faithfulness" in r and r["faithfulness"]:
                    faithfulness_scores.append(r["faithfulness"])
                for k, v in r.items():
                    if k.startswith("step_"):
                        steps.setdefault(k, []).append(v)
            except json.JSONDecodeError:
                continue

    def _stats(vals: list) -> dict | None:
        if not vals:
            return None
        s = sorted(vals)
        n = len(s)
        return {
            "avg": round(statistics.mean(s)),
            "p50": round(s[n // 2]),
            "p95": round(s[min(int(n * 0.95), n - 1)]),
            "n": n,
        }

    return {
        "retrieve": _stats(retrieve_ms),
        "generate": _stats(generate_ms),
        "total": _stats(total_ms),
        "ttft": _stats(ttft_ms),
        "faithfulness_avg": round(statistics.mean(faithfulness_scores), 2) if faithfulness_scores else None,
        "steps": {k: _stats(v) for k, v in steps.items()},
    }
