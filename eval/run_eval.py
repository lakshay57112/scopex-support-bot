"""Retrieval evaluation: does the right help article show up in the top results?

Usage:
    python -m eval.run_eval              # hybrid (keywords + embeddings)
    python -m eval.run_eval --keyword    # keyword search only

Metrics:
    Hit@1 / Hit@3 - share of questions where the right article is ranked 1st / in the top 3
    MRR           - mean reciprocal rank (1.0 = always ranked first)
"""

import argparse
import json
from pathlib import Path

from scopex_bot.knowledge import load_articles
from scopex_bot.retriever import HybridRetriever

QUESTIONS = Path(__file__).parent / "questions.json"


def evaluate(retriever: HybridRetriever, k: int = 5, verbose: bool = True) -> dict:
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    hit1 = hit3 = rr_total = 0.0
    misses = []

    for item in questions:
        ranked = [h.article.id for h in retriever.search(item["q"], k=k)]
        rank = ranked.index(item["expected"]) + 1 if item["expected"] in ranked else None
        hit1 += rank == 1
        hit3 += rank is not None and rank <= 3
        rr_total += 1 / rank if rank else 0
        if rank != 1:
            misses.append((item["q"], item["expected"], rank, ranked[:3]))

    n = len(questions)
    results = {"questions": n, "hit@1": hit1 / n, "hit@3": hit3 / n, "mrr": rr_total / n}

    if verbose:
        print(f"Mode: {retriever.mode}  |  {n} questions")
        print(f"Hit@1: {results['hit@1']:.0%}   Hit@3: {results['hit@3']:.0%}   MRR: {results['mrr']:.3f}")
        if misses:
            print("\nNot ranked first:")
            for q, expected, rank, top3 in misses:
                print(f"  - {q!r}\n      expected {expected} (rank {rank or '>' + str(k)}), got {top3}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", action="store_true", help="disable embeddings")
    args = parser.parse_args()
    evaluate(HybridRetriever(load_articles(), use_embeddings=not args.keyword))
