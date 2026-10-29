"""Compare retrievers on a corpus + labeled queries.

    python -m tr_retrieval.evaluate data/corpus.jsonl data/queries.jsonl
    python -m tr_retrieval.evaluate data/corpus.jsonl data/queries.jsonl --dense-model intfloat/multilingual-e5-base
"""

import argparse
import json
from pathlib import Path

from .bm25 import BM25
from .dense import DenseRetriever, TfidfEncoder
from .fusion import HybridRetriever
from .metrics import evaluate_run
from .text import tokenize


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_retrievers(dense_model: str | None = None) -> dict:
    retrievers = {
        "bm25 (no stemming)": BM25(tokenizer=lambda t: tokenize(t, stem_prefix=None)),
        "bm25 (F5 stemming)": BM25(),
        "char-tfidf": DenseRetriever(TfidfEncoder()),
        "hybrid: bm25 + char-tfidf": HybridRetriever([BM25(), DenseRetriever(TfidfEncoder())]),
    }
    if dense_model:
        from .dense import SentenceTransformerEncoder

        encoder = SentenceTransformerEncoder(dense_model)
        retrievers[f"dense ({dense_model})"] = DenseRetriever(encoder)
        retrievers["hybrid: bm25 + dense"] = HybridRetriever([BM25(), DenseRetriever(encoder)])
    return retrievers


def run_benchmark(corpus: list[dict], queries: list[dict], retrievers: dict, k: int = 10) -> dict[str, dict]:
    doc_ids = [d["id"] for d in corpus]
    texts = [d.get("title", "") + "\n" + d["text"] for d in corpus]
    qrels = {q["id"]: set(q["relevant"]) for q in queries}

    results = {}
    for name, retriever in retrievers.items():
        retriever.fit(texts)
        run = {q["id"]: [doc_ids[i] for i, _ in retriever.search(q["text"], k)] for q in queries}
        results[name] = evaluate_run(run, qrels)
    return results


def format_table(results: dict[str, dict]) -> str:
    metrics = list(next(iter(results.values())))
    lines = ["| retriever | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for name, scores in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{scores[m]:.3f}" for m in metrics) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus")
    parser.add_argument("queries")
    parser.add_argument("--dense-model", default=None, help="sentence-transformers model name")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    results = run_benchmark(load_jsonl(args.corpus), load_jsonl(args.queries), build_retrievers(args.dense_model), args.k)
    print(format_table(results))


if __name__ == "__main__":
    main()
