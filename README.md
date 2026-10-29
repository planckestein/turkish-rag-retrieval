# turkish-rag-retrieval

**Hybrid retrieval and offline evaluation for Turkish RAG systems.**

Most RAG stacks are tuned for English. Turkish breaks a few of their assumptions:

- **Agglutinative morphology.** One stem appears with many suffixes (`iade`, `iadesi`, `iademin`, `iadeler`). Exact term matching misses most of them.
- **Casing.** `str.lower()` maps `I → i` and `İ → i̇` (i + combining dot), which quietly breaks matching.
- **Score calibration.** BM25 scores and embedding similarities live on different scales, so naive score mixing for hybrid search is fragile.

`turkish-rag-retrieval` is a small, dependency-light toolkit that handles these problems and, more importantly, **measures** the effect of each retrieval choice before it reaches production.

## What's inside

| Module | What it does |
|---|---|
| `tr_retrieval.text` | Turkish-aware lowercasing, tokenization, stopwords, F5 prefix stemming |
| `tr_retrieval.bm25` | Okapi BM25 over an inverted index |
| `tr_retrieval.dense` | Vector retrieval with pluggable encoders: char n-gram TF-IDF (no download) or any `sentence-transformers` model (default `intfloat/multilingual-e5-base`, with E5 query/passage prefixes) |
| `tr_retrieval.fusion` | Weighted Reciprocal Rank Fusion and a `HybridRetriever` |
| `tr_retrieval.chunking` | Sentence-aware chunking with sentence overlap |
| `tr_retrieval.metrics` | Recall@k, MRR, nDCG@k |
| `tr_retrieval.evaluate` | CLI that benchmarks retrievers on a corpus and labeled queries |

## Quick start

```bash
pip install -e ".[dev]"
pytest
python -m tr_retrieval.evaluate data/corpus.jsonl data/queries.jsonl
```

With a real embedding model:

```bash
pip install -e ".[dense]"
python -m tr_retrieval.evaluate data/corpus.jsonl data/queries.jsonl --dense-model intfloat/multilingual-e5-base
```

Library use:

```python
from tr_retrieval import BM25, DenseRetriever, HybridRetriever, TfidfEncoder

docs = ["İade talebi Siparişlerim sayfasından oluşturulur.", "Kargo takip numarası e-posta ile gönderilir."]
retriever = HybridRetriever([BM25(), DenseRetriever(TfidfEncoder())]).fit(docs)
retriever.search("iade talebini nereden açarım", k=1)
```

## Results on the sample set

`data/` contains a small customer-support knowledge base (15 documents, 14 queries) written to resemble real support traffic: users rarely use the same word forms as the documentation.

| retriever | recall@1 | recall@3 | recall@5 | mrr | ndcg@5 |
|---|---|---|---|---|---|
| bm25 (no stemming) | 0.500 | 0.571 | 0.571 | 0.536 | 0.545 |
| bm25 (F5 stemming) | 0.821 | 0.929 | 0.929 | 0.902 | 0.902 |
| char-tfidf | 0.750 | 0.929 | 1.000 | 0.871 | 0.903 |
| hybrid: bm25 + char-tfidf | 0.821 | 0.929 | 0.929 | 0.903 | 0.902 |

Takeaways:

- **Prefix stemming alone raises MRR from 0.54 to 0.90.** For Turkish, morphology handling matters more than the choice of ranking function.
- Character n-grams reach full recall@5 because they are naturally robust to suffixes.
- The set is small on purpose (it runs in CI in about a second), so treat these numbers as a sanity check, not a benchmark. Plug in your own `corpus.jsonl` / `queries.jsonl` for real decisions.

## Data format

```jsonl
// corpus.jsonl
{"id": "d01", "title": "İade süreci", "text": "..."}
// queries.jsonl
{"id": "q01", "text": "aldığım ürünü geri göndermek istiyorum", "relevant": ["d01"]}
```

## Design notes

- **Why F5 stemming?** Truncating words to their first five characters is a simple, dictionary-free baseline that performs close to full morphological analyzers for Turkish IR (Can et al., 2008, *Information Retrieval on Turkish Texts*, JASIST). It is fast and has no external dependencies.
- **Why RRF?** It only uses ranks, so it needs no score normalization between lexical and dense retrievers (Cormack, Clarke & Büttcher, SIGIR 2009). Weights allow you to favor one retriever.
- **Why offline evaluation first?** Retrieval quality caps answer quality in RAG. A labeled query set makes retrieval changes measurable and reviewable before running online A/B tests.

## Roadmap

- [ ] Cross-encoder reranking stage
- [ ] Morphological analyzer backend (e.g. Zemberek) as an alternative to F5
- [ ] LLM-as-judge for answer faithfulness on top of retrieval metrics
- [ ] Larger public Turkish evaluation set

## License

MIT
