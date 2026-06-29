"""FastAPI skeleton for the Medical Hybrid Search POC.

Endpoints:
- GET  /health          Health check for API and downstream services.
- POST /ingest          Trigger async ingestion of a raw PubMed JSONL file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import redis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_results
from src.tasks.ingest_tasks import ingest_documents_pipeline


load_dotenv()

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis_client: redis.Redis | None = None


def _get_redis_client() -> redis.Redis | None:
    """Return a Redis client, or None if Redis is unreachable."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        _redis_client.ping()
        return _redis_client
    except Exception as exc:
        logger.warning(f"Redis unavailable, search caching disabled: {exc}")
        _redis_client = None
        return None


def _cache_key(query: str, top_k: int, section_filter: str | None) -> str:
    """Build a deterministic cache key for a search request."""
    key_payload = f"{query.strip().lower()}:{top_k}:{section_filter or ''}"
    return f"search_cache:{hashlib.sha256(key_payload.encode()).hexdigest()}"

app = FastAPI(
    title="Medical Hybrid Search API",
    version="0.1.0",
    description="Hybrid dense + sparse vector search over PubMed abstracts.")


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""

    filename: str = Field(
        ...,
        description="Name of the raw PubMed JSONL file located in DATA_RAW_DIR.",
        examples=["pubmed_subset_1000.jsonl"],
    )


class IngestResponse(BaseModel):
    """Response body for the /ingest endpoint."""

    task_id: str
    status: str
    file_path: str


class SearchRequest(BaseModel):
    """Request body for the /search endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language search query.",
        examples=["Does metformin reduce cardiovascular risk in type 2 diabetes?"],
    )
    section_filter: str | None = Field(
        default=None,
        description="Optional section payload filter (e.g., RESULTS, CONCLUSION).",
        examples=["RESULTS"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top reranked results to return (1-50).",
        examples=[5],
    )


class SearchResponse(BaseModel):
    """Response body for the /search endpoint."""

    query: str
    section_filter: str | None
    results: list[dict[str, Any]]


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Return API health status."""
    return {
        "status": "ok",
        "qdrant_host": os.getenv("QDRANT_HOST", "localhost"),
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Queue the ingestion pipeline for the given raw PubMed JSONL file."""
    raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
    file_path = raw_dir / request.filename

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {file_path}",
        )

    task = ingest_documents_pipeline.delay(str(file_path))

    return IngestResponse(
        task_id=task.id,
        status="queued",
        file_path=str(file_path),
    )


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Run hybrid dense + sparse retrieval, rerank, and return the top results.

    Results are cached in Redis for 1 hour using a key derived from the query,
    top_k, and optional section_filter.
    """
    redis_client = _get_redis_client()
    cache_key = _cache_key(request.query, request.top_k, request.section_filter)

    # Try to serve a cached response.
    if redis_client is not None:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info(f"cache hit: {cache_key}")
                payload = json.loads(cached)
                return SearchResponse(
                    query=payload["query"],
                    section_filter=payload["section_filter"],
                    results=payload["results"],
                )
        except Exception as exc:
            logger.warning(f"Redis cache read failed for {cache_key}: {exc}")

    try:
        candidates = hybrid_search(
            query=request.query,
            limit=20,
            section_filter=request.section_filter,
        )
        final_results = rerank_results(
            query=request.query,
            candidates=candidates,
            top_k=request.top_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {exc}",
        ) from exc

    # Convert numpy types (e.g., numpy.float32 scores) to native Python types so
    # FastAPI/Pydantic can serialize the response to JSON.
    serialized_results = json.loads(
        json.dumps(
            {
                "query": request.query,
                "section_filter": request.section_filter,
                "results": final_results,
            },
            default=lambda obj: float(obj) if isinstance(obj, np.floating) else obj,
        )
    )

    # Store the serialized response in Redis.
    if redis_client is not None:
        try:
            redis_client.setex(cache_key, 3600, json.dumps(serialized_results))
        except Exception as exc:
            logger.warning(f"Redis cache write failed for {cache_key}: {exc}")

    # Convert numpy types to Python floats
    for r in final_results:
        if hasattr(r, "score"):
            r.score = float(r.score)
        if hasattr(r, "final_score"):
            r.final_score = float(r.final_score)

    return SearchResponse(
        query=serialized_results["query"],
        section_filter=serialized_results["section_filter"],
        results=serialized_results["results"],
    )
