"""
Download arXiv papers: foundational classics (full PDF) + recent papers by category (abstract only).
Output: data/raw/arxiv_papers.json.gz
"""
import gzip
import json
import logging
import re
import tempfile
import time
from pathlib import Path

import arxiv
from tqdm import tqdm

from src.config import RAW_DIR

log = logging.getLogger(__name__)

ARXIV_FILE = RAW_DIR / "arxiv_papers.json.gz"

# ── Foundational papers — fetched by exact ID ─────────────────────────────────
# These are seminal papers that must always be in the index.

FOUNDATIONAL_IDS = [
    # ── Transformers & Attention ──────────────────────────────────────────
    "1706.03762",   # Attention Is All You Need (Vaswani et al. 2017)
    "1409.0473",    # Neural Machine Translation + Bahdanau Attention
    "1508.04025",   # Effective Approaches to Attention (Luong et al.)

    # ── Language Models ───────────────────────────────────────────────────
    "1810.04805",   # BERT (Devlin et al. 2018)
    "1907.11692",   # RoBERTa
    "1906.08237",   # XLNet
    "1910.10683",   # T5 — Exploring the Limits of Transfer Learning
    "2005.14165",   # GPT-3 — Language Models are Few-Shot Learners
    "2302.13971",   # LLaMA (Touvron et al. 2023)
    "2307.09288",   # LLaMA 2
    "2203.02155",   # InstructGPT / RLHF (Ouyang et al.)
    "2212.08073",   # Constitutional AI (Anthropic)
    "2305.18290",   # DPO — Direct Preference Optimization
    "2203.15556",   # Chinchilla — Training Compute-Optimal LLMs
    "1910.13461",   # BART

    # ── Embeddings & Retrieval ────────────────────────────────────────────
    "1301.3781",    # Word2Vec (Mikolov et al.)
    "1802.05365",   # ELMo — Deep Contextualized Word Representations
    "1908.10084",   # Sentence-BERT
    "2004.12832",   # ColBERT — Efficient Passage Retrieval
    "2005.11401",   # RAG — Retrieval-Augmented Generation (Lewis et al.)
    "2309.07597",   # BGE — BAAI General Embeddings
    "2112.09118",   # HyDE — Hypothetical Document Embeddings
    "2210.11610",   # BEIR — Heterogeneous Retrieval Benchmark

    # ── Reasoning & Prompting ─────────────────────────────────────────────
    "2201.11903",   # Chain-of-Thought Prompting (Wei et al.)
    "2205.01068",   # Least-to-Most Prompting
    "2203.11171",   # Self-Consistency (Wang et al.)
    "2210.06726",   # ReAct — Reasoning + Acting
    "2305.10601",   # Tree of Thoughts

    # ── Deep Learning Foundations ─────────────────────────────────────────
    "1512.03385",   # ResNet — Deep Residual Learning
    "1409.1556",    # VGGNet
    "1502.03167",   # Batch Normalization
    "1207.0580",    # Dropout (Hinton et al.)
    "1412.6980",    # Adam optimizer
    "1409.3215",    # Seq2Seq — Learning to Align and Translate
    "1406.2661",    # GAN — Generative Adversarial Networks
    "1312.6114",    # VAE — Auto-Encoding Variational Bayes
    "1505.04597",   # U-Net
    "1506.02640",   # YOLO v1

    # ── Generative & Diffusion Models ─────────────────────────────────────
    "2006.11239",   # DDPM — Denoising Diffusion Probabilistic Models
    "2112.10752",   # Stable Diffusion / Latent Diffusion
    "2102.12092",   # DALL-E
    "2103.00020",   # CLIP (Radford et al.)
    "2401.04088",   # Mixtral of Experts

    # ── Reinforcement Learning ────────────────────────────────────────────
    "1312.5602",    # DQN — Playing Atari with Deep RL
    "1707.06347",   # PPO — Proximal Policy Optimization
    "1509.02971",   # DDPG — Continuous Control with Deep RL

    # ── Graph & Structured Models ─────────────────────────────────────────
    "1609.02907",   # GCN — Semi-Supervised Classification with Graph CNNs
    "1706.02216",   # GraphSAGE
    "1710.10903",   # GAT — Graph Attention Networks

    # ── Efficient Transformers & Fine-tuning ──────────────────────────────
    "2106.09685",   # LoRA — Low-Rank Adaptation
    "2312.00752",   # Mamba — Linear-Time Sequence Modeling
    "2009.14794",   # Flash Attention foundations
    "1901.02860",   # Transformer-XL

    # ── Multimodal ────────────────────────────────────────────────────────
    "2204.14198",   # Flamingo — Visual Language Model
    "2301.13688",   # InstructBLIP
]

# ── Recent papers by category (fills remaining budget) ───────────────────────

CATEGORIES = [
    ("cs.AI",         60),
    ("cs.LG",         60),
    ("cs.CL",         60),
    ("cs.CV",         40),
    ("cs.IR",         30),
    ("math.ST",       30),
    ("quant-ph",      30),
    ("physics.data-an", 20),
]
TOTAL_BUDGET = 600   # foundational + recent


def _extract_pdf_text(result: arxiv.Result) -> str | None:
    """Download PDF and extract clean text. Returns None on failure."""
    try:
        import fitz  # pymupdf
    except ImportError:
        log.warning("pymupdf not installed — run: pip install pymupdf==1.24.5")
        return None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = result.download_pdf(dirpath=tmp)
            doc = fitz.open(path)
            pages = []
            for page in doc:
                pages.append(page.get_text())
            doc.close()
            raw = "\n".join(pages)
            # Remove references section (everything after "References\n")
            raw = re.split(r"\nReferences\s*\n", raw, maxsplit=1)[0]
            raw = re.sub(r"\n{3,}", "\n\n", raw).strip()
            return raw if len(raw) > 500 else None
    except Exception as e:
        log.warning(f"PDF extraction failed: {e}")
        return None


def _to_dict(result: arxiv.Result, full_text: str | None = None) -> dict:
    pid = result.entry_id.split("/abs/")[-1]
    abstract = result.summary.strip()
    title = result.title.strip()
    return {
        "id": pid,
        "title": title,
        "abstract": abstract,
        "text": full_text if full_text else f"{title}\n\n{abstract}",
        "authors": [a.name for a in result.authors[:5]],
        "published": str(result.published.date()),
        "url": f"https://arxiv.org/abs/{pid}",
        "categories": [c if isinstance(c, str) else c.id for c in result.categories],
        "source": "arxiv",
        "has_full_text": full_text is not None,
    }


def _fetch_foundational(client: arxiv.Client, use_pdf: bool = True) -> list[dict]:
    """Fetch foundational papers by exact arXiv ID, with optional full PDF extraction."""
    papers = []
    log.info(f"Fetching {len(FOUNDATIONAL_IDS)} foundational papers (PDF={use_pdf})…")

    batch_size = 100
    for i in range(0, len(FOUNDATIONAL_IDS), batch_size):
        batch = FOUNDATIONAL_IDS[i:i + batch_size]
        search = arxiv.Search(id_list=batch)
        try:
            results = list(client.results(search))
            for result in tqdm(results, desc=f"Foundational batch {i//batch_size + 1}"):
                full_text = _extract_pdf_text(result) if use_pdf else None
                papers.append(_to_dict(result, full_text))
                if use_pdf:
                    time.sleep(1)  # be polite to arXiv
        except Exception as e:
            log.warning(f"Error fetching foundational batch: {e}")
        time.sleep(2)

    n_full = sum(1 for p in papers if p.get("has_full_text"))
    log.info(f"Fetched {len(papers)}/{len(FOUNDATIONAL_IDS)} foundational papers ({n_full} with full PDF)")
    return papers


def _fetch_recent(client: arxiv.Client, seen_ids: set[str], budget: int) -> list[dict]:
    """Fill remaining budget with recent papers by category."""
    papers = []

    for category, quota in tqdm(CATEGORIES, desc="Recent papers by category"):
        if len(papers) >= budget:
            break
        remaining = min(quota, budget - len(papers))
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=remaining,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        try:
            for result in client.results(search):
                pid = result.entry_id.split("/abs/")[-1]
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                papers.append(_to_dict(result))
        except Exception as e:
            log.warning(f"Error fetching {category}: {e}")
        time.sleep(1)

    return papers


def fetch_arxiv(force: bool = False, use_pdf: bool = True) -> list[dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ARXIV_FILE.exists() and not force:
        log.info(f"arXiv cache exists at {ARXIV_FILE}, loading…")
        return _load()

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)

    # 1. Foundational papers — full PDF if available
    foundational = _fetch_foundational(client, use_pdf=use_pdf)
    seen_ids = {p["id"] for p in foundational}

    # 2. Recent papers to fill remaining budget
    recent_budget = max(0, TOTAL_BUDGET - len(foundational))
    log.info(f"Fetching up to {recent_budget} recent papers…")
    recent = _fetch_recent(client, seen_ids, recent_budget)

    papers = foundational + recent
    log.info(f"Total: {len(papers)} papers ({len(foundational)} foundational + {len(recent)} recent)")
    _save(papers)
    return papers


def _save(papers: list[dict]) -> None:
    with gzip.open(ARXIV_FILE, "wt", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(papers)} arXiv papers → {ARXIV_FILE}")


def _load() -> list[dict]:
    with gzip.open(ARXIV_FILE, "rt", encoding="utf-8") as f:
        papers = json.load(f)
    log.info(f"Loaded {len(papers)} arXiv papers from {ARXIV_FILE}")
    return papers


def stats(papers: list[dict]) -> dict:
    from collections import Counter
    cats: list[str] = []
    for p in papers:
        cats.extend(p.get("categories", []))
    return {
        "total": len(papers),
        "foundational": sum(1 for p in papers if p["id"].split("v")[0] in
                           [fid.split("v")[0] for fid in FOUNDATIONAL_IDS]),
        "top_categories": Counter(cats).most_common(10),
        "date_range": f"{min(p['published'] for p in papers)} → {max(p['published'] for p in papers)}",
    }
