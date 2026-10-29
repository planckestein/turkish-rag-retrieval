"""Okapi BM25 over an inverted index."""

import math
from collections import Counter, defaultdict
from typing import Callable

import numpy as np

from .text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75, tokenizer: Callable[[str], list[str]] = tokenize):
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer

    def fit(self, documents: list[str]) -> "BM25":
        self.n_docs = len(documents)
        self.doc_len = np.zeros(self.n_docs, dtype=np.float32)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_id, doc in enumerate(documents):
            tokens = self.tokenizer(doc)
            self.doc_len[doc_id] = len(tokens)
            for term, tf in Counter(tokens).items():
                self.postings[term].append((doc_id, tf))

        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        # BM25+ style idf that never goes negative for very common terms
        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }
        return self

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term in set(self.tokenizer(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings[term]:
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm[doc_id])
        return scores

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores = self.scores(query)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
