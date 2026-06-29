"""Hybrid dense + sparse retrieval with Reciprocal Rank Fusion.

Queries Qdrant using both dense cosine search and native BM25 sparse search,
then fuses the two ranked lists using standard RRF (k=60).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.ingestion.embedder import TEIEmbedder
from src.ingestion.sparse_encoder import build_query_sparse_vector, load_vocab, get_vocab_path


load_dotenv()

logger = logging.getLogger(__name__)

COLLECTION_NAME = "medical_documents"
DENSE_LIMIT = 30
SPARSE_LIMIT = 30
SPARSE_VECTOR_NAME = "sparse"
RRF_K = 60


def _collection_has_sparse_vectors(client: QdrantClient) -> bool:
    """Return True if the collection defines any sparse vector config."""
    try:
        info = client.get_collection(COLLECTION_NAME)
        config = info.config.params.sparse_vectors
        return config is not None and len(config) > 0
    except Exception as exc:
        logger.warning(f"Could not determine sparse vector config: {exc}")
        return False


def _get_sparse_vocab() -> tuple[dict[str, int], dict[int, float]] | None:
    """Load the sparse vocabulary if it exists."""
    processed_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    return load_vocab(get_vocab_path(processed_dir))


def _get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client from environment variables."""
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    return QdrantClient(host=host, port=port, grpc_port=grpc_port, prefer_grpc=False)


def _build_filter(section_filter: str | None) -> models.Filter | None:
    """Build a Qdrant payload filter on the `section` field if requested."""
    if not section_filter:
        return None
    return models.Filter(
        must=[
            models.FieldCondition(
                key="section",
                match=models.MatchValue(value=section_filter.upper()),
            )
        ]
    )


def _dense_search(
    client: QdrantClient,
    query_vector: list[float],
    limit: int,
    section_filter: str | None,
) -> list[models.ScoredPoint]:
    """Run dense vector search in Qdrant."""
    try:
        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=_build_filter(section_filter),
            limit=limit,
            with_payload=True,
        )
    except Exception as exc:
        logger.error(f"Dense search failed: {exc}")
        return []


def _sparse_search(
    client: QdrantClient,
    query: str,
    vocab: dict[str, int],
    limit: int,
    section_filter: str | None,
) -> list[models.ScoredPoint]:
    """Run sparse TF-IDF vector search in Qdrant using the shared vocabulary."""
    try:
        sparse_vector = build_query_sparse_vector(query, vocab)
        if not sparse_vector.indices:
            logger.info("[hybrid_search] query has no known sparse terms; skipping sparse search")
            return []

        return client.search(
            collection_name=COLLECTION_NAME,
            query_vector=models.NamedSparseVector(
                name=SPARSE_VECTOR_NAME,
                vector=sparse_vector,
            ),
            query_filter=_build_filter(section_filter),
            limit=limit,
            with_payload=True,
        )
    except Exception as exc:
        logger.warning(f"Sparse search failed; falling back to dense-only: {exc}")
        return []


def _point_to_dict(point: models.ScoredPoint, source: str) -> dict[str, Any]:
    """Convert a Qdrant scored point into a plain result dictionary."""
    payload = point.payload or {}
    return {
        "id": point.id,
        "chunk_id": payload.get("chunk_id"),
        "doc_id": payload.get("doc_id"),
        "title": payload.get("title"),
        "section": payload.get("section"),
        "text": payload.get("text"),
        "source": source,
    }


def reciprocal_rank_fusion(
    dense_results: list[models.ScoredPoint],
    sparse_results: list[models.ScoredPoint],
    k: int = RRF_K,
) -> list[tuple[str, float, dict[str, Any]]]:
    """Fuse two ranked lists using the standard RRF formula.

    score = sum(1 / (k + rank_i)) for each list where the item appears.
    """
    scores: dict[str, float] = {}
    metadata: dict[str, dict[str, Any]] = {}

    for rank, point in enumerate(dense_results, start=1):
        key = str(point.id)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in metadata:
            metadata[key] = _point_to_dict(point, "dense")

    for rank, point in enumerate(sparse_results, start=1):
        key = str(point.id)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
        if key not in metadata:
            metadata[key] = _point_to_dict(point, "sparse")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(key, score, metadata[key]) for key, score in ranked]


def hybrid_search(
    query: str,
    limit: int = 20,
    section_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Run hybrid dense + sparse retrieval and return top fused candidates.

    Parameters
    ----------
    query: str
        Search query text.
    limit: int
        Number of fused results to return (default 20).
    section_filter: str | None
        Optional section payload filter (e.g., "RESULTS", "CONCLUSION").

    Returns
    -------
    list[dict]
        Fused candidate list enriched with RRF scores and metadata.
    """
    if not query or not query.strip():
        raise ValueError("Query must be non-empty.")

    embedder = TEIEmbedder()
    client = _get_qdrant_client()

    try:
        logger.info(f"[hybrid_search] embedding query: {query!r}")
        query_vector = embedder.embed_query(query)

        logger.info("[hybrid_search] running dense search...")
        dense_results = _dense_search(
            client, query_vector, DENSE_LIMIT, section_filter
        )

        vocab = _get_sparse_vocab()
        if _collection_has_sparse_vectors(client) and vocab is not None:
            logger.info("[hybrid_search] running sparse BM25 search...")
            sparse_results = _sparse_search(
                client, query, vocab[0], SPARSE_LIMIT, section_filter
            )
        else:
            logger.info("[hybrid_search] collection has no sparse vectors or vocab; dense-only")
            sparse_results = []

        logger.info(
            f"[hybrid_search] dense={len(dense_results)}, sparse={len(sparse_results)}"
        )

        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        results = []
        for key, rrf_score, meta in fused[:limit]:
            results.append(
                {
                    "id": key,
                    "rrf_score": rrf_score,
                    **meta,
                }
            )

        return results
    finally:
        embedder.close()
        client.close()
