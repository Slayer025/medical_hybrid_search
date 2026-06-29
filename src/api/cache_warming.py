"""Cache warming and admin helpers for the /search endpoint.

Provides reusable functions for:
- Building deterministic Redis cache keys
- Connecting to Redis
- Pre-populating the cache with common medical queries
- Reading cache statistics
- Clearing cached search results
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import numpy as np
import redis
from dotenv import load_dotenv
from fastapi import HTTPException

from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_results


load_dotenv()

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CACHE_TTL_SECONDS = 3600
CACHE_KEY_PREFIX = "search_cache:"
_redis_client: redis.Redis | None = None

DEFAULT_WARM_QUERIES = [
    "cancer chemotherapy",
    "diabetes treatment",
    "Alzheimer's biomarkers",
    "cardiovascular risk",
    "gene therapy",
    "COVID vaccine",
    "mRNA therapy",
    "clinical trial results",
    "drug side effects",
    "machine learning medical",
    "hypertension management",
    "obesity treatment",
    "asthma therapy",
    "depression medication",
    "arthritis treatment",
]


def get_redis_client() -> redis.Redis | None:
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


def cache_key(query: str, top_k: int, section_filter: str | None) -> str:
    """Build a deterministic cache key for a search request."""
    key_payload = f"{query.strip().lower()}:{top_k}:{section_filter or ''}"
    return f"{CACHE_KEY_PREFIX}{hashlib.sha256(key_payload.encode()).hexdigest()}"


def _run_search(query: str, top_k: int, section_filter: str | None = None) -> dict[str, Any]:
    """Execute the full search pipeline and return a JSON-serializable payload."""
    candidates = hybrid_search(
        query=query,
        limit=20,
        section_filter=section_filter,
    )
    final_results = rerank_results(
        query=query,
        candidates=candidates,
        top_k=top_k,
    )

    # Ensure all numpy float types are converted to native Python floats
    # so the payload is JSON-serializable before storing in Redis.
    for r in final_results:
        if hasattr(r, "score"):
            r.score = float(r.score)
        if hasattr(r, "final_score"):
            r.final_score = float(r.final_score)
        # Recursively convert any nested numpy scalars in result dicts.
        for key, value in list(r.items()):
            if isinstance(value, np.floating):
                r[key] = float(value)
            elif isinstance(value, np.integer):
                r[key] = int(value)

    return {
        "query": query,
        "section_filter": section_filter,
        "results": final_results,
    }


def warm_cache(
    queries: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """Pre-populate Redis cache with common queries.

    Returns
    -------
    dict with keys:
        queries_warmed, already_cached, newly_cached, errors
    """
    redis_client = get_redis_client()
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis unavailable; cannot warm cache")

    queries = queries if queries is not None else DEFAULT_WARM_QUERIES
    already_cached = 0
    newly_cached = 0
    errors: list[str] = []

    for query in queries:
        key = cache_key(query, top_k, None)
        try:
            if redis_client.exists(key):
                already_cached += 1
                continue

            logger.info(f"[warm_cache] warming query: {query!r}")
            payload = _run_search(query, top_k)
            redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(payload))
            newly_cached += 1
        except Exception as exc:
            msg = f"Failed to warm query {query!r}: {exc}"
            logger.warning(msg)
            errors.append(msg)

    return {
        "queries_warmed": len(queries),
        "already_cached": already_cached,
        "newly_cached": newly_cached,
        "errors": errors,
    }


def cache_stats() -> dict[str, Any]:
    """Return operational statistics for the search cache."""
    redis_client = get_redis_client()
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    try:
        total_keys = redis_client.dbsize()
        cache_keys = redis_client.keys(f"{CACHE_KEY_PREFIX}*")

        memory_info = redis_client.info("memory")
        clients_info = redis_client.info("clients")
        server_info = redis_client.info("server")
        stats_info = redis_client.info("stats")

        ttls = [redis_client.ttl(k) for k in cache_keys if redis_client.ttl(k) > 0]
        avg_ttl = sum(ttls) / len(ttls) if ttls else 0

        hits = stats_info.get("keyspace_hits", 0)
        misses = stats_info.get("keyspace_misses", 0)
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0

        return {
            "total_keys": total_keys,
            "cache_keys": len(cache_keys),
            "memory_used": memory_info.get("used_memory_human", "unknown"),
            "connected_clients": clients_info.get("connected_clients", 0),
            "uptime_seconds": server_info.get("uptime_in_seconds", 0),
            "hit_rate": round(hit_rate, 4),
            "hits": hits,
            "misses": misses,
            "avg_ttl_remaining": round(avg_ttl, 1),
        }
    except Exception as exc:
        logger.warning(f"Failed to read cache stats: {exc}")
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {exc}") from exc


def clear_cache() -> dict[str, Any]:
    """Delete all search cache keys and return a summary."""
    redis_client = get_redis_client()
    if redis_client is None:
        raise HTTPException(status_code=503, detail="Redis unavailable")

    try:
        keys = redis_client.keys(f"{CACHE_KEY_PREFIX}*")
        if keys:
            redis_client.delete(*keys)
        return {
            "cleared_keys": len(keys),
            "message": "Cache cleared successfully",
        }
    except Exception as exc:
        logger.warning(f"Failed to clear cache: {exc}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {exc}") from exc
