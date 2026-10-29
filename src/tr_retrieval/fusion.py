"""Hybrid retrieval with Reciprocal Rank Fusion (Cormack et al., 2009).

RRF only uses ranks, so BM25 scores and cosine similarities do not need to be
calibrated against each other. That makes it a safe default for hybrid search.
"""

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[int]], weights: list[float] | None = None, k: int = 60
) -> list[tuple[int, float]]:
    weights = weights or [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights and rankings must have the same length")

    fused: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += weight / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever:
    def __init__(self, retrievers: list, weights: list[float] | None = None, candidates: int = 50, rrf_k: int = 60):
        self.retrievers = retrievers
        self.weights = weights
        self.candidates = candidates
        self.rrf_k = rrf_k

    def fit(self, documents: list[str]) -> "HybridRetriever":
        for retriever in self.retrievers:
            retriever.fit(documents)
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        rankings = [[doc_id for doc_id, _ in r.search(query, self.candidates)] for r in self.retrievers]
        return reciprocal_rank_fusion(rankings, self.weights, self.rrf_k)[:k]
