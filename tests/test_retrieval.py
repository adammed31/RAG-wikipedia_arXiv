"""Unit tests — no Ollama, no internet required."""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieve import QueryType, route


# ── Router ────────────────────────────────────────────────────────────────────

def test_route_short_query():
    assert route("DNA") == QueryType.SIMPLE


def test_route_comparison():
    assert route("Compare Newton vs Einstein theories") == QueryType.COMPARE


def test_route_conceptual():
    assert route("What is quantum mechanics?") == QueryType.CONCEPTUAL


def test_route_complex():
    q = "What are the differences between supervised and unsupervised learning, and how do they apply to neural networks?"
    assert route(q) == QueryType.COMPLEX


# ── Decomposition (mocked) ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_decompose_parses_json():
    from src.retrieve import decompose
    mock_response = '["What is supervised learning?", "What is unsupervised learning?"]'
    with patch("src.retrieve._ollama_generate", new_callable=AsyncMock) as mock:
        mock.return_value = mock_response
        subs = await decompose("Explain supervised and unsupervised learning")
    assert len(subs) == 2
    assert all(isinstance(s, str) for s in subs)


@pytest.mark.asyncio
async def test_decompose_fallback_on_bad_json():
    from src.retrieve import decompose
    with patch("src.retrieve._ollama_generate", new_callable=AsyncMock) as mock:
        mock.return_value = "not valid json at all"
        subs = await decompose("Some question")
    assert subs == ["Some question"]


# ── RAG Fusion (mocked) ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rag_fusion_includes_original():
    from src.retrieve import rag_fusion_queries
    with patch("src.retrieve._ollama_generate", new_callable=AsyncMock) as mock:
        mock.return_value = '["query variant 1", "query variant 2"]'
        variants = await rag_fusion_queries("original query")
    assert variants[0] == "original query"
    assert len(variants) >= 2
