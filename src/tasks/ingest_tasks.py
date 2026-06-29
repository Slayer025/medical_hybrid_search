"""Celery ingestion task: chunk, embed, and upsert PubMed documents into Qdrant."""

from __future__ import annotations

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import TEIEmbedder
from src.ingestion.sparse_encoder import (
    build_vocabulary,
    compute_idf,
    compute_sparse_vectors,
    get_vocab_path,
    save_vocab,
)
from src.tasks.celery_app import app


logger = logging.getLogger(__name__)

COLLECTION_NAME = "medical_documents"


def _get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client from environment variables."""
    load_dotenv()
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    return QdrantClient(host=host, port=port, grpc_port=grpc_port, prefer_grpc=False)


def _qdrant_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUID5 for a chunk from its chunk_id."""
    namespace = uuid.NAMESPACE_URL
    return str(uuid.uuid5(namespace, chunk_id))


def _load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load processed chunk records from a JSONL file."""
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _upsert_batch(
    client: QdrantClient,
    embedder: TEIEmbedder,
    batch_chunks: list[dict[str, Any]],
    batch_sparse_vectors: list[models.SparseVector],
) -> int:
    """Embed and upsert a single batch of chunks into Qdrant.

    Each point contains both a dense embedding (default/unnamed vector) and a
    named sparse vector for keyword retrieval.
    """
    texts = [chunk["text"] for chunk in batch_chunks]
    embeddings = embedder.embed_texts(texts)

    if len(embeddings) != len(batch_chunks):
        raise ValueError(
            f"Embedding count mismatch: {len(embeddings)} vs {len(batch_chunks)}"
        )
    if len(batch_sparse_vectors) != len(batch_chunks):
        raise ValueError(
            f"Sparse vector count mismatch: {len(batch_sparse_vectors)} vs {len(batch_chunks)}"
        )

    points = []
    for chunk, vector, sparse_vector in zip(
        batch_chunks, embeddings, batch_sparse_vectors
    ):
        point_id = _qdrant_uuid(chunk["chunk_id"])
        payload = {
            "doc_id": chunk["doc_id"],
            "title": chunk.get("title", ""),
            "section": chunk["section"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
        }
        points.append(
            models.PointStruct(
                id=point_id,
                vector={"": vector, "sparse": sparse_vector},
                payload=payload,
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return len(points)


@app.task(bind=True, max_retries=3, default_retry_delay=10)
def ingest_documents_pipeline(self, file_path: str) -> dict[str, Any]:
    """Celery task that chunks, embeds, and upserts a raw PubMed JSONL file.

    Parameters
    ----------
    file_path: str
        Path to the raw PubMed JSONL file.

    Returns
    -------
    dict
        Summary containing file_path, processed_path, chunk_count, and upsert_count.
    """
    load_dotenv()

    raw_path = Path(file_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    processed_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"{raw_path.stem}_chunks.jsonl"

    logger.info(f"[ingest_documents_pipeline] chunking {raw_path} -> {processed_path}")
    chunk_count = chunk_documents(str(raw_path), str(processed_path))

    if chunk_count == 0:
        logger.warning("No chunks produced; skipping embedding and upsert.")
        return {
            "file_path": file_path,
            "processed_path": str(processed_path),
            "chunk_count": 0,
            "upsert_count": 0,
        }

    logger.info(f"[ingest_documents_pipeline] loading {chunk_count} chunks")
    chunks = _load_chunks(processed_path)

    logger.info("[ingest_documents_pipeline] building sparse vector vocabulary")
    texts = [chunk["text"] for chunk in chunks]
    vocab = build_vocabulary(texts)
    idf = compute_idf(texts, vocab)
    sparse_vectors = compute_sparse_vectors(texts, vocab, idf)
    vocab_path = get_vocab_path(processed_dir)
    save_vocab(vocab, idf, vocab_path)
    logger.info(
        f"[ingest_documents_pipeline] sparse vocab size: {len(vocab)} saved to {vocab_path}"
    )

    embedder = TEIEmbedder()
    client = _get_qdrant_client()

    upsert_count = 0
    batch_size = embedder.batch_size

    try:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_sparse = sparse_vectors[i : i + batch_size]
            logger.info(
                f"[ingest_documents_pipeline] embedding batch {i // batch_size + 1} "
                f"({len(batch)} chunks)"
            )
            upserted = _upsert_batch(client, embedder, batch, batch_sparse)
            upsert_count += upserted
            logger.info(
                f"[ingest_documents_pipeline] upserted {upsert_count}/{len(chunks)} chunks"
            )
    except Exception as exc:
        logger.exception("[ingest_documents_pipeline] ingestion failed")
        try:
            self.retry(exc=exc)
        except Exception as retry_exc:
            logger.error(f"[ingest_documents_pipeline] retries exhausted: {retry_exc}")
            raise
    finally:
        embedder.close()
        client.close()

    logger.info(
        f"[ingest_documents_pipeline] complete: {chunk_count} chunks, "
        f"{upsert_count} upserted"
    )

    return {
        "file_path": file_path,
        "processed_path": str(processed_path),
        "chunk_count": chunk_count,
        "upsert_count": upsert_count,
    }
