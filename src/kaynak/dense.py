"""Vector retrieval with pluggable encoders.

`TfidfEncoder` runs anywhere and is used in tests. `SentenceTransformerEncoder`
wraps a real embedding model (default: multilingual E5) for production use.
"""

from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .text import normalize


class Encoder(Protocol):
    def fit(self, documents: list[str]) -> "Encoder": ...
    def encode_documents(self, texts: list[str]): ...
    def encode_queries(self, texts: list[str]): ...


class TfidfEncoder:
    """Character n-gram TF-IDF. Robust to suffixes and typos, no model download."""

    def __init__(self, ngram_range: tuple[int, int] = (3, 5)):
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, preprocessor=normalize, sublinear_tf=True
        )

    def fit(self, documents: list[str]) -> "TfidfEncoder":
        self.vectorizer.fit(documents)
        return self

    def encode_documents(self, texts: list[str]):
        return self.vectorizer.transform(texts)

    def encode_queries(self, texts: list[str]):
        return self.vectorizer.transform(texts)


class SentenceTransformerEncoder:
    """Embedding model encoder. E5 models expect 'query: ' / 'passage: ' prefixes."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        batch_size: int = 32,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.batch_size = batch_size

    def fit(self, documents: list[str]) -> "SentenceTransformerEncoder":
        return self

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True
        )

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.query_prefix + t for t in texts])


class DenseRetriever:
    """Cosine similarity search. Vectors are L2-normalized, so a dot product is enough."""

    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def fit(self, documents: list[str]) -> "DenseRetriever":
        self.encoder.fit(documents)
        self.doc_vectors = self.encoder.encode_documents(documents)
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.encoder.encode_queries([query])
        sims = self.doc_vectors @ q.T
        if hasattr(sims, "toarray"):  # sparse result from TF-IDF
            sims = sims.toarray()
        return np.asarray(sims).ravel()

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores = self.scores(query)
        top = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]
