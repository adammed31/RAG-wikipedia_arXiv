"""
Append-only JSONL loggers for queries and feedback.
No database — just files.
"""
import json
import time
from pathlib import Path

from src.config import FEEDBACK_FILE, LOGS_DIR


def _append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_query(
    query: str,
    query_type: str,
    sub_queries: list[str],
    sources: list[dict],
    answer: str,
    latency_ms: int,
    cached: bool,
) -> None:
    _append(
        LOGS_DIR / "queries.jsonl",
        {
            "ts": time.time(),
            "query": query,
            "query_type": query_type,
            "sub_queries": sub_queries,
            "n_sources": len(sources),
            "answer_len": len(answer),
            "latency_ms": latency_ms,
            "cached": cached,
        },
    )


def log_feedback(query: str, answer: str, thumbs: str, comment: str = "") -> None:
    _append(
        FEEDBACK_FILE,
        {
            "ts": time.time(),
            "query": query,
            "answer": answer[:500],
            "thumbs": thumbs,
            "comment": comment,
        },
    )
