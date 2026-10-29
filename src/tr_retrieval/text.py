"""Turkish-aware text processing for retrieval.

Turkish is agglutinative: one stem shows up with many suffixes
("iade", "iadesi", "iadeler", "iademi"). Exact-match retrieval misses these, so
the lexical index uses prefix truncation (F5 stemming), which is a simple and
well-studied baseline for Turkish IR (Can et al., 2008) and needs no dictionary.
"""

import re
import unicodedata

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")

STOPWORDS = frozenset(
    """
    acaba ama ancak artık bana bazı belki ben beni benim bir biri birkaç biz bize
    bu buna bunu bunun çok da daha de defa diye en gibi hem hep her hiç için ile
    ise kadar ki kim mi mı mu mü nasıl ne neden nerede niye o olan olarak ona onu
    sen siz şey şu ve veya ya yani yine
    """.split()
)


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish rules ('I' -> 'ı', 'İ' -> 'i').

    str.lower() turns 'I' into 'i' and 'İ' into 'i' + combining dot, which
    silently breaks term matching for Turkish.
    """
    return text.replace("I", "ı").replace("İ", "i").lower()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return _SPACE_RE.sub(" ", turkish_lower(text)).strip()


def tokenize(text: str, *, stem_prefix: int | None = 5, remove_stopwords: bool = True) -> list[str]:
    """Split into word tokens, drop stopwords and truncate to a fixed prefix."""
    tokens = _TOKEN_RE.findall(normalize(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem_prefix:
        tokens = [t[:stem_prefix] for t in tokens]
    return tokens
