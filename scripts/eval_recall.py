"""
Offline Recall@K + Faithfulness evaluation against data/eval/eval_set.json.

Usage:
    python scripts/eval_recall.py [--k 1 3 5] [--source both|wikipedia|arxiv]
    python scripts/eval_recall.py --faithfulness          # also compute faithfulness
    python scripts/eval_recall.py --list-titles           # inspect index titles
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.generate import stream_answer
from src.index import load_chunks, load_indexes
from src.metrics import faithfulness_score, recall_at_k
from src.retrieve import retrieve

EVAL_FILE = ROOT / "data" / "eval" / "eval_set.json"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--source", default="both", choices=["both", "wikipedia", "arxiv"])
    p.add_argument("--faithfulness", action="store_true", help="Also compute faithfulness score")
    p.add_argument("--list-titles", action="store_true")
    return p.parse_args()


def _unique_titles(chunk_store: dict) -> list[str]:
    seen, titles = set(), []
    for c in chunk_store.values():
        t = c.get("title", "")
        if t and t not in seen:
            seen.add(t)
            titles.append(t)
    return sorted(titles)


async def _generate_answer(query: str, chunks: list) -> str:
    stream, _ = await stream_answer(query, chunks)
    buf = []
    async for token in stream:
        buf.append(token)
    return "".join(buf)


async def _eval(eval_set, lc_faiss, bm25, chunk_store, k_values, source, with_faithfulness):
    max_k = max(k_values)
    rows = []

    for item in eval_set:
        query    = item["query"]
        relevant = item["relevant_titles"]

        chunks, qtype, _, _ = await retrieve(
            query, lc_faiss, bm25, chunk_store,
            source_filter=source, top_k=max_k,
        )

        retrieved_titles = [c["title"] for c, _ in chunks]
        scores = {k: recall_at_k(retrieved_titles[:k], relevant) for k in k_values}

        f_score = 0.0
        if with_faithfulness and chunks:
            answer  = await _generate_answer(query, chunks)
            f_score = await faithfulness_score(chunks, answer)

        rows.append({
            "query": query, "qtype": qtype,
            "relevant": relevant, "retrieved": retrieved_titles[:max_k],
            "scores": scores, "faithfulness": f_score,
        })

        tag = " ".join(f"R@{k}={s:.2f}" for k, s in scores.items())
        f_tag = f"  Faith={f_score:.2f}" if with_faithfulness else ""
        print(f"  [{tag}{f_tag}]  {query[:60]}")

    return rows


def _print_report(rows, k_values, with_faithfulness):
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    n = len(rows)
    summary = {}
    for k in k_values:
        avg = sum(r["scores"][k] for r in rows) / n
        summary[f"recall_{k}"] = round(avg, 3)
        print(f"  Recall@{k}:     {avg:.3f}")
    if with_faithfulness:
        avg_f = sum(r["faithfulness"] for r in rows) / n
        summary["faithfulness"] = round(avg_f, 3)
        print(f"  Faithfulness:  {avg_f:.3f}")
    print(f"  ({n} queries)")

    results_file = EVAL_FILE.parent / "results.json"
    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved → {results_file}")


def main():
    args = _parse_args()

    print("Loading indexes…")
    lc_faiss, bm25 = load_indexes()
    chunk_store = load_chunks()
    print(f"Index ready: {len(chunk_store):,} chunks\n")

    if args.list_titles:
        for t in _unique_titles(chunk_store):
            print(f"  {t}")
        return

    with open(EVAL_FILE, encoding="utf-8") as f:
        eval_set = json.load(f)

    mode = "Recall@K + Faithfulness" if args.faithfulness else "Recall@K"
    print(f"Evaluating {len(eval_set)} queries — {mode}\n")
    rows = asyncio.run(_eval(eval_set, lc_faiss, bm25, chunk_store, args.k, args.source, args.faithfulness))
    _print_report(rows, args.k, args.faithfulness)


if __name__ == "__main__":
    main()
