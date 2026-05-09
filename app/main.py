"""Streamlit RAG interface — Wikipedia + arXiv, streaming, source filter, model comparison, feedback.
Run: streamlit run app/main.py
"""

import asyncio
import re
import sys
import time
from pathlib import Path

import streamlit as st

# src/ is one level up
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx
import src.cache as cache_store
from src.config import FAISS_LC_DIR, BM25_LC_FILE, CHUNKS_FILE, OLLAMA_URL
from src.generate import extract_citations, stream_answer
from src.index import load_chunks, load_indexes
from src.logger import log_feedback, log_query
from src.retrieve import retrieve

ALL_MODELS = ["llama3.1:8b", "qwen2.5:14b", "mistral-small3.1"]


def _fix_latex(text: str) -> str:
    """
    Convert LaTeX delimiters to Streamlit-compatible format.

    Transforms \\[...\\] → $$...$$ (display math)
    and \\(...\\) → $...$ (inline math)
    """
    text = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$$\n{m.group(1).strip()}\n$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', lambda m: f'${m.group(1).strip()}$', text)
    return text


def get_installed_models() -> set[str]:
    """
    Fetch locally installed Ollama models.

    Queries the Ollama API and filters against ALL_MODELS
    to return only installed models from our supported list.
    """
    try:
        resp = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=2.0)
        tags = {m["name"] for m in resp.json().get("models", [])}
        return {m for m in ALL_MODELS if any(m.split(":")[0] in tag for tag in tags)}
    except Exception:
        return set()


st.set_page_config(
    page_title="RAG Wikipedia + arXiv",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading indexes…")
def load_all():
    """
    Load all indexes and chunks into memory.

    Uses Streamlit cache to prevent reloading on every interaction.
    Returns (None, None, None) if index files don't exist yet.
    """
    if not ((FAISS_LC_DIR / "index.faiss").exists() and BM25_LC_FILE.exists() and CHUNKS_FILE.exists()):
        return None, None, None
    lc_faiss, bm25_retriever = load_indexes()
    chunk_store = load_chunks()
    return lc_faiss, bm25_retriever, chunk_store


lc_faiss, bm25_retriever, chunk_store = load_all()
indexes_ready = lc_faiss is not None


def _source_counts() -> tuple[int, int]:
    if not chunk_store:
        return 0, 0
    n_wiki  = sum(1 for c in chunk_store.values() if c.get("source") == "wikipedia")
    n_arxiv = sum(1 for c in chunk_store.values() if c.get("source") == "arxiv")
    return n_wiki, n_arxiv


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("🔬 RAG Wikipedia + arXiv")
    st.caption("100% local · No API keys · No cloud")

    st.divider()
    st.subheader("🗂️ Source filter")
    source_filter = st.radio(
        "Search in:",
        options=["both", "wikipedia", "arxiv"],
        format_func=lambda x: {"both": "🌐 All sources", "wikipedia": "📖 Wikipedia only", "arxiv": "📄 arXiv only"}[x],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("🤖 Model")
    installed = get_installed_models()

    def _label(m):
        return m if m in installed else f"{m} ⬇️"

    compare_mode = st.toggle("Compare two models", value=False)
    if compare_mode:
        model_a = st.selectbox("Model A", options=ALL_MODELS, index=0, key="model_a", format_func=_label)
        model_b = st.selectbox("Model B", options=ALL_MODELS, index=1, key="model_b", format_func=_label)
        selected_model = model_a
        if model_a not in installed or model_b not in installed:
            missing = [m for m in [model_a, model_b] if m not in installed]
            for m in missing:
                st.warning(f"`{m}` not installed — run `ollama pull {m}`")
    else:
        selected_model = st.selectbox("LLM", options=ALL_MODELS, index=0,
                                      label_visibility="collapsed", format_func=_label)
        model_a = model_b = selected_model
        if selected_model not in installed:
            st.warning(f"`{selected_model}` not installed — run `ollama pull {selected_model}`")

    st.divider()
    st.subheader("⚙️ Options")
    use_cache  = st.toggle("Use query cache", value=True)
    show_debug = st.toggle("Show query type / sub-queries", value=False)

    st.divider()
    st.subheader("📊 Index status")
    if indexes_ready:
        n_wiki, n_arxiv = _source_counts()
        st.success(f"✅ {len(chunk_store):,} chunks loaded")
        col1, col2 = st.columns(2)
        col1.metric("📖 Wikipedia", f"{n_wiki:,}")
        col2.metric("📄 arXiv", f"{n_arxiv:,}")
    else:
        st.error("❌ Indexes not built yet")
        st.code("python scripts/build.py", language="bash")

    if st.button("🗑️ Clear cache", use_container_width=True):
        cache_store.clear()
        st.toast("Cache cleared!")

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("BAAI/bge-small-en · ms-marco-MiniLM · LLaMA 3.1")


def _render_source(src: dict) -> None:
    is_arxiv = src.get("source") == "arxiv"
    badge    = "📄 arXiv" if is_arxiv else "📖 Wikipedia"
    title    = src["title"]
    url      = src["url"]
    score    = src["score"]
    excerpt  = src["excerpt"]

    header = f"{badge} · **[{title}]({url})** — score `{score}`"
    st.markdown(header)

    if is_arxiv:
        authors   = src.get("authors", [])
        published = src.get("published", "")
        cats      = src.get("categories", [])
        meta_parts = []
        if authors:
            meta_parts.append("👤 " + ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else ""))
        if published:
            meta_parts.append(f"📅 {published}")
        if cats:
            meta_parts.append("🏷️ " + " · ".join(cats[:3]))
        if meta_parts:
            st.caption(" · ".join(meta_parts))

    st.markdown(f"> {excerpt}")
    st.divider()


st.title("Ask a scientific question")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} sources", expanded=False):
                for src in msg["sources"]:
                    _render_source(src)

query = st.chat_input(
    "E.g. What is a transformer? How does RLHF work?",
    disabled=not indexes_ready,
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        t0          = time.time()
        sources     = []
        query_type  = "simple"
        sub_queries = [query]

        with st.spinner("Searching…"):
            chunks, query_type, sub_queries = asyncio.run(
                retrieve(query, lc_faiss, bm25_retriever,
                         chunk_store, source_filter=source_filter, model=model_a)
            )

        if not chunks:
            st.warning("No relevant sources found in the index for this question. Try a different query or build a larger index.")
            st.session_state.messages.append({"role": "assistant", "content": "No relevant sources found.", "sources": []})
            st.stop()

        if show_debug:
            st.info(
                f"**Query type:** `{query_type}` · "
                f"**Source filter:** `{source_filter}` · "
                f"**Sub-queries:** {', '.join(f'`{q}`' for q in sub_queries)}"
            )

        def _run_stream(placeholder, model):
            token_stream, srcs = asyncio.run(stream_answer(query, chunks, model=model))
            async def _collect():
                buf = []
                async for token in token_stream:
                    buf.append(token)
                    text = _fix_latex("".join(buf))
                    cursor = "" if text.count("$$") % 2 != 0 else "▌"
                    placeholder.markdown(text + cursor)
                final = _fix_latex("".join(buf))
                placeholder.markdown(final)
                return final
            return asyncio.run(_collect()), srcs

        if compare_mode:
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"🤖 `{model_a}`")
                ph1 = st.empty()
            with col2:
                st.caption(f"🤖 `{model_b}`")
                ph2 = st.empty()
            answer_a, sources   = _run_stream(ph1, model_a)
            answer_b, _         = _run_stream(ph2, model_b)
            full_answer = f"**{model_a}:**\n{answer_a}\n\n---\n\n**{model_b}:**\n{answer_b}"
        else:
            full_answer  = ""
            from_cache   = False
            cache_key    = f"{query}||{source_filter}||{selected_model}"
            if use_cache:
                cached = cache_store.get(cache_key)
                if cached:
                    from_cache  = True
                    full_answer = cached["answer"]
                    sources     = cached["sources"]
                    st.markdown(full_answer)
                    st.caption("⚡ From cache")

            if not from_cache:
                ph = st.empty()
                try:
                    full_answer, sources = _run_stream(ph, selected_model)
                except Exception as e:
                    full_answer = f"❌ Error: {e}"
                    ph.error(full_answer)

                if use_cache and full_answer and not full_answer.startswith("❌"):
                    cache_store.set(cache_key, {
                        "answer": full_answer,
                        "sources": sources,
                        "query_type": query_type,
                        "sub_queries": sub_queries,
                    })

        latency_ms = int((time.time() - t0) * 1000)

        if sources:
            n_wiki_src  = sum(1 for s in sources if s.get("source") == "wikipedia")
            n_arxiv_src = sum(1 for s in sources if s.get("source") == "arxiv")
            label = f"📚 {len(sources)} sources"
            if n_wiki_src and n_arxiv_src:
                label += f" ({n_wiki_src} Wikipedia · {n_arxiv_src} arXiv)"
            label += f" · {latency_ms}ms"
            with st.expander(label, expanded=False):
                for src in sources:
                    _render_source(src)

        col1, col2, _ = st.columns([1, 1, 8])
        with col1:
            if st.button("👍", key=f"up_{len(st.session_state.messages)}"):
                log_feedback(query, full_answer, "up")
                st.toast("Thanks!")
        with col2:
            if st.button("👎", key=f"dn_{len(st.session_state.messages)}"):
                log_feedback(query, full_answer, "down")
                st.toast("Thanks, we'll improve!")

    log_query(query, query_type, sub_queries, sources, full_answer, latency_ms, False)
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
    })
