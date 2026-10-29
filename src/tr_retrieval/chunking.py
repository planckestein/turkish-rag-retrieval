"""Sentence-aware chunking for long documents before indexing."""

import re

_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ0-9\"'(])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def chunk_text(text: str, max_chars: int = 600, overlap_sentences: int = 1) -> list[str]:
    """Pack whole sentences into chunks of at most `max_chars`.

    The last `overlap_sentences` of each chunk are repeated at the start of the
    next one, so an answer that spans a boundary is still retrievable.
    A single sentence longer than `max_chars` becomes its own chunk.
    """
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        candidate = " ".join(current + [sentence])
        if current and len(candidate) > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            # drop the overlap if it alone would push us over the limit
            if len(" ".join(current + [sentence])) > max_chars:
                current = []
        current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks
