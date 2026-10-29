from pathlib import Path

import pytest

from kaynak.bm25 import BM25
from kaynak.dense import DenseRetriever, TfidfEncoder
from kaynak.evaluate import build_retrievers, load_jsonl, run_benchmark
from kaynak.fusion import HybridRetriever, reciprocal_rank_fusion
from kaynak.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

DATA = Path(__file__).resolve().parents[1] / "data"
DOCS = [
    "İade talebi Siparişlerim sayfasından oluşturulur.",
    "Kargo takip numarası e-posta ile gönderilir.",
    "Şifre sıfırlama bağlantısı 30 dakika geçerlidir.",
]


def test_bm25_matches_inflected_forms():
    bm25 = BM25().fit(DOCS)
    # "talebini" and "talebi" share the prefix "taleb"
    assert bm25.search("iade talebini nereden açarım", k=1)[0][0] == 0


def test_bm25_unknown_query_returns_nothing():
    assert BM25().fit(DOCS).search("xyzzy") == []


def test_dense_tfidf_retriever():
    retriever = DenseRetriever(TfidfEncoder()).fit(DOCS)
    assert retriever.search("kargomu takip etmek", k=1)[0][0] == 1


def test_rrf_prefers_documents_ranked_well_by_both():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert {fused[0][0], fused[1][0]} == {1, 2}
    assert fused[-1][0] in {3, 4}


def test_rrf_weights_length_mismatch():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1], [2]], weights=[1.0])


def test_hybrid_retriever():
    hybrid = HybridRetriever([BM25(), DenseRetriever(TfidfEncoder())]).fit(DOCS)
    assert hybrid.search("şifremi sıfırlamak", k=1)[0][0] == 2


def test_metrics():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 2) == 0.5
    assert reciprocal_rank(["a", "b"], {"b"}) == 0.5
    assert ndcg_at_k(["a"], {"a"}, 5) == 1.0
    assert ndcg_at_k(["x", "y"], {"a"}, 5) == 0.0


def test_benchmark_on_sample_data():
    results = run_benchmark(load_jsonl(DATA / "corpus.jsonl"), load_jsonl(DATA / "queries.jsonl"), build_retrievers())
    hybrid = results["hybrid: bm25 + char-tfidf"]
    assert hybrid["recall@5"] >= 0.7
    # stemming should help on Turkish morphology
    assert results["bm25 (F5 stemming)"]["mrr"] >= results["bm25 (no stemming)"]["mrr"]
