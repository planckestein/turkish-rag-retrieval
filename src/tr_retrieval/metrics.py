"""Offline retrieval metrics with binary relevance."""

import math
from statistics import mean


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, d in enumerate(retrieved[:k], start=1) if d in relevant)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_run(run: dict[str, list[str]], qrels: dict[str, set[str]], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    """Average metrics over all queries in `qrels`. Missing queries count as zero."""
    out: dict[str, float] = {}
    qids = list(qrels)
    for k in ks:
        out[f"recall@{k}"] = mean(recall_at_k(run.get(q, []), qrels[q], k) for q in qids)
    out["mrr"] = mean(reciprocal_rank(run.get(q, []), qrels[q]) for q in qids)
    out[f"ndcg@{max(ks)}"] = mean(ndcg_at_k(run.get(q, []), qrels[q], max(ks)) for q in qids)
    return out
