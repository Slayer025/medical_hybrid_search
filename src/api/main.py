"""FastAPI skeleton for the Medical Hybrid Search POC.

Endpoints:
- GET    /health          Health check for API and downstream services.
- POST   /ingest          Trigger async ingestion of a raw PubMed JSONL file.
- POST   /search          Hybrid dense + sparse search with reranking and Redis caching.
- POST   /cache/warm      Pre-populate the Redis cache with common medical queries.
- GET    /cache/stats     Return operational cache statistics.
- DELETE /cache/clear     Clear all cached search results.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.cache_warming import (
    cache_key,
    cache_stats,
    clear_cache,
    get_redis_client,
    warm_cache,
)
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_results
from src.tasks.ingest_tasks import ingest_documents_pipeline


load_dotenv()

logger = logging.getLogger(__name__)


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


class WarmCacheRequest(BaseModel):
    """Optional body for /cache/warm."""

    queries: list[str] | None = Field(
        default=None,
        description="List of queries to warm. Uses default medical queries if omitted.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results per warmed query.",
    )


class WarmCacheResponse(BaseModel):
    """Response body for /cache/warm."""

    queries_warmed: int
    already_cached: int
    newly_cached: int
    errors: list[str]


@app.on_event("startup")
def startup_warm_cache() -> None:
    """Optionally warm the search cache in the background on startup."""
    if os.getenv("CACHE_WARM_ON_STARTUP", "false").lower() in ("1", "true", "yes"):
        logger.info("[startup] background cache warming enabled")

        def _warm() -> None:
            try:
                result = warm_cache()
                logger.info(f"[startup] cache warming complete: {result}")
            except Exception as exc:
                logger.warning(f"[startup] background cache warming failed: {exc}")

        threading.Thread(target=_warm, daemon=True).start()
    else:
        logger.info("[startup] background cache warming disabled")


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
    redis_client = get_redis_client()
    key = cache_key(request.query, request.top_k, request.section_filter)

    # Try to serve a cached response.
    if redis_client is not None:
        try:
            cached = redis_client.get(key)
            if cached:
                logger.info(f"cache hit: {key}")
                payload = json.loads(cached)
                return SearchResponse(
                    query=payload["query"],
                    section_filter=payload["section_filter"],
                    results=payload["results"],
                )
        except Exception as exc:
            logger.warning(f"Redis cache read failed for {key}: {exc}")

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
            redis_client.setex(key, 3600, json.dumps(serialized_results))
        except Exception as exc:
            logger.warning(f"Redis cache write failed for {key}: {exc}")

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


@app.post("/cache/warm", response_model=WarmCacheResponse)
def cache_warm(request: WarmCacheRequest) -> WarmCacheResponse:
    """Pre-populate the Redis cache with the supplied (or default) queries."""
    result = warm_cache(queries=request.queries, top_k=request.top_k)
    return WarmCacheResponse(**result)


@app.get("/cache/stats")
def get_cache_stats() -> dict[str, Any]:
    """Return operational statistics for the search cache."""
    return cache_stats()


@app.delete("/cache/clear")
def delete_cache() -> dict[str, Any]:
    """Clear all cached search results."""
    return clear_cache()
