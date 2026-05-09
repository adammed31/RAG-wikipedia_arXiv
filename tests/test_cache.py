import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.cache as cache_store


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("src.cache.CACHE_FILE", tmp_path / "test_cache.json")
    yield


def test_set_and_get():
    cache_store.set("hello", {"answer": "world", "sources": []})
    result = cache_store.get("hello")
    assert result is not None
    assert result["answer"] == "world"


def test_miss_returns_none():
    assert cache_store.get("nonexistent query xyz") is None


def test_case_insensitive_key():
    cache_store.set("Hello World", {"answer": "a", "sources": []})
    assert cache_store.get("hello world") is not None


def test_ttl_expiry(monkeypatch):
    monkeypatch.setattr("src.cache.CACHE_TTL_HOURS", 0)
    cache_store.set("expiring", {"answer": "x", "sources": []})
    # TTL=0 means immediately expired
    assert cache_store.get("expiring") is None


def test_clear():
    cache_store.set("q", {"answer": "a", "sources": []})
    cache_store.clear()
    assert cache_store.get("q") is None
