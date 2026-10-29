from tr_retrieval.text import normalize, tokenize, turkish_lower


def test_turkish_lower_handles_dotted_and_dotless_i():
    assert turkish_lower("IŞIK İADE") == "ışık iade"


def test_normalize_collapses_whitespace():
    assert normalize("  Kargo   NEREDE \n") == "kargo nerede"


def test_tokenize_removes_stopwords_and_stems():
    assert tokenize("İadeler ve iadesi için") == ["iadel", "iades"]
    assert tokenize("siparişlerim siparişim")[0][:5] == tokenize("siparişim")[0]


def test_tokenize_without_stemming():
    assert tokenize("Siparişlerim", stem_prefix=None) == ["siparişlerim"]
