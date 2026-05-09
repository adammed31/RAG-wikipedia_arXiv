"""
Simple file-based query cache.
Stores results in data/cache/query_cache.json.
TTL enforced on read.
"""
import hashlib
import json
import time
from pathlib import Path

from src.config import CACHE_FILE, CACHE_TTL_HOURS


def _key(query: str) -> str:
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def _load() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save(data: dict) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get(query: str) -> dict | None:
    data = _load()
    entry = data.get(_key(query))
    if entry is None:
        return None
    if time.time() - entry["ts"] > CACHE_TTL_HOURS * 3600:
        return None  # expired
    return entry["result"]


def set(query: str, result: dict) -> None:
    data = _load()
    data[_key(query)] = {"ts": time.time(), "result": result}
    # keep max 500 entries
    if len(data) > 500:
        oldest = sorted(data.items(), key=lambda x: x[1]["ts"])
        data = dict(oldest[-500:])
    _save(data)


def clear() -> None:
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
