"""FastAPI skeleton for the Medical Hybrid Search POC.

Endpoints:
- GET  /health          Health check for API and downstream services.
- POST /ingest          Trigger async ingestion of a raw PubMed JSONL file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank_results
from src.tasks.ingest_tasks import ingest_documents_pipeline


load_dotenv()

app = FastAPI(
    title="Medical Hybrid Search API",
    version="0.1.0",
    description="Hybrid dense + sparse vector search over PubMed abstracts.")


class IngestRequest(BaseModel):
    """Request body for the /ingest endpoint."""

    filename: str = Field(
        ...,
        description="Name of the raw PubMed JSONL file located in DATA_RAW_DIR.",
        examples=["pubmed_subset_100.jsonl"],
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
    """Run hybrid dense + sparse retrieval, rerank, and return the top results."""
    try:
        candidates = hybrid_search(
            query=request.query,
            limit=20,
            section_filter=request.section_filter,
        )
        final_results = rerank_results(
            query=request.query,
            candidates=candidates,
            top_k=5,
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

    # Convert numpy types to Python floats
    for r in final_results:
        if hasattr(r, "score"):
            r.score = float(r.score)
        if hasattr(r, "final_score"):
            r.final_score = float(r.final_score)
    
    # Convert numpy types to Python floats

    
    for r in final_results:

    
        if hasattr(r, 'score'):

    
            r.score = float(r.score)

    
        if hasattr(r, 'final_score'):

    
            r.final_score = float(r.final_score)

    
    

    
    return SearchResponse(
        query=serialized_results["query"],
        section_filter=serialized_results["section_filter"],
        results=serialized_results["results"],
    )
