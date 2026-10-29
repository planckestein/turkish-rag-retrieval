import pytest

from kaynak.chunking import chunk_text, split_sentences

TEXT = "Birinci cümle burada. İkinci cümle biraz daha uzun! Üçüncü cümle mi? Dördüncü cümle bitti."


def test_split_sentences():
    assert split_sentences(TEXT) == [
        "Birinci cümle burada.",
        "İkinci cümle biraz daha uzun!",
        "Üçüncü cümle mi?",
        "Dördüncü cümle bitti.",
    ]


def test_chunks_respect_limit_and_overlap():
    chunks = chunk_text(TEXT, max_chars=60, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)
    # consecutive chunks share a sentence
    for a, b in zip(chunks, chunks[1:]):
        assert split_sentences(a)[-1] == split_sentences(b)[0]


def test_no_overlap():
    chunks = chunk_text(TEXT, max_chars=50, overlap_sentences=0)
    assert " ".join(chunks) == TEXT


def test_long_sentence_is_its_own_chunk():
    long = "a" * 100 + "."
    assert chunk_text(long, max_chars=20) == [long]


def test_negative_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text(TEXT, overlap_sentences=-1)
