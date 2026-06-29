"""Sparse vector encoder for hybrid BM25-style retrieval.

Builds a simple TF-IDF sparse vector representation for each chunk using a
shared vocabulary. The vocabulary and IDF weights are saved alongside the
processed chunks so query-time encoding uses the same term -> index mapping.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from qdrant_client.http import models


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SPARSE_VOCAB_FILENAME = "sparse_vocab.json"


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(texts: list[str]) -> dict[str, int]:
    """Build a term -> index vocabulary from a list of texts."""
    vocab: dict[str, int] = {}
    for text in texts:
        for token in set(tokenize(text)):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def compute_idf(texts: list[str], vocab: dict[str, int]) -> dict[int, float]:
    """Compute IDF weight for each vocabulary index."""
    df: dict[int, int] = {idx: 0 for idx in range(len(vocab))}
    for text in texts:
        seen: set[int] = set()
        for token in tokenize(text):
            idx = vocab.get(token)
            if idx is not None and idx not in seen:
                df[idx] += 1
                seen.add(idx)

    n = len(texts)
    return {
        idx: math.log(n / count) if count > 0 else 0.0
        for idx, count in df.items()
    }


def compute_sparse_vectors(
    texts: list[str],
    vocab: dict[str, int],
    idf: dict[int, float],
) -> list[models.SparseVector]:
    """Compute a TF-IDF sparse vector for each text."""
    vectors: list[models.SparseVector] = []
    for text in texts:
        tokens = tokenize(text)
        if not tokens:
            vectors.append(models.SparseVector(indices=[], values=[]))
            continue

        counts: dict[int, int] = {}
        for token in tokens:
            idx = vocab.get(token)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1

        indices = sorted(counts.keys())
        values = [float(counts[idx] * idf[idx]) for idx in indices]
        vectors.append(models.SparseVector(indices=indices, values=values))

    return vectors


def save_vocab(
    vocab: dict[str, int],
    idf: dict[int, float],
    path: Path | str,
) -> None:
    """Persist vocabulary and IDF weights to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    idf_list = [idf.get(i, 0.0) for i in range(len(vocab))]
    with path.open("w", encoding="utf-8") as f:
        json.dump({"vocab": vocab, "idf": idf_list}, f)


def load_vocab(path: Path | str) -> tuple[dict[str, int], dict[int, float]] | None:
    """Load vocabulary and IDF weights from JSON.

    Returns None if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    vocab: dict[str, int] = data["vocab"]
    idf_list: list[float] = data["idf"]
    idf = {i: float(value) for i, value in enumerate(idf_list)}
    return vocab, idf


def build_query_sparse_vector(
    query: str,
    vocab: dict[str, int],
) -> models.SparseVector:
    """Encode a query string as a sparse vector using the shared vocabulary."""
    tokens = tokenize(query)
    counts: dict[int, int] = {}
    for token in tokens:
        idx = vocab.get(token)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1

    indices = sorted(counts.keys())
    values = [float(counts[idx]) for idx in indices]
    return models.SparseVector(indices=indices, values=values)


def get_vocab_path(processed_dir: Path | str) -> Path:
    """Return the default path for the sparse vocabulary file."""
    return Path(processed_dir) / SPARSE_VOCAB_FILENAME


if __name__ == "__main__":
    sample_texts = [
        "A case report of diabetes treatment with metformin.",
        "Clinical trial results for cancer chemotherapy drugs.",
        "Gene therapy and mRNA therapy in cardiovascular risk.",
    ]
    vocab = build_vocabulary(sample_texts)
    idf = compute_idf(sample_texts, vocab)
    vectors = compute_sparse_vectors(sample_texts, vocab, idf)
    print(f"Vocab size: {len(vocab)}")
    for i, vec in enumerate(vectors):
        print(f"Doc {i}: indices={vec.indices}, values={[round(v, 4) for v in vec.values]}")

    query_vec = build_query_sparse_vector("cancer chemotherapy", vocab)
    print(f"Query: indices={query_vec.indices}, values={query_vec.values}")
