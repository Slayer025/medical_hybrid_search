# Medical Hybrid Search POC — Specification Document

> **Version:** 1.0  
> **Date:** 2026-06-29  
> **Status:** Proof of Concept — End-to-End Pipeline Working  
> **Author:** DC  

---

## 1. Project Overview

The **Medical Hybrid Search POC** is a production-style demonstration of an intelligent search engine over medical literature. It ingests PubMed abstracts and exposes a fast, relevance-ranked search interface that combines semantic understanding with traditional keyword matching.

The system is designed as a **reference architecture** for modern AI retrieval pipelines, showcasing how dense vector search, sparse lexical retrieval, and cross-encoder reranking can be composed into a single, low-latency search experience.

---

## 2. Goals & Objectives

### Primary Goals
- Build an **end-to-end AI search system** that indexes and searches real biomedical text.
- Implement **hybrid retrieval** by combining dense semantic vectors with sparse keyword matching (BM25).
- Demonstrate a **scalable, production-oriented architecture** using containerized services and async workers.
- Process and search **real medical abstracts** sourced from PubMed via the HuggingFace scientific_papers dataset.

### Key Objectives
| Objective | Description |
|-----------|-------------|
| End-to-end pipeline | Ingest → chunk → embed → index → search → rerank → display |
| Hybrid search | Dense + sparse retrieval with reciprocal rank fusion |
| Sub-second latency | Return ranked results in under one second |
| Section-aware chunking | Preserve document structure (BACKGROUND, METHODS, RESULTS, CONCLUSION) |
| Modular architecture | FastAPI, Celery, Qdrant, Streamlit, Docker compose cleanly |
| Real data | Index and query actual PubMed abstracts |

---

## 3. System Architecture

The project follows a **service-oriented, container-ready architecture** with clear separation between ingestion, storage, search, and UI layers.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   FastAPI API   │────▶│  Qdrant Vector  │
│  (Search +      │     │  (Search +      │     │     Store       │
│   Ingest Jobs)  │     │   Ingest Tasks) │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  Celery Worker  │
                        │  + Redis Broker │
                        └─────────────────┘
```

### Core Services

| Component | Technology | Responsibility |
|-----------|------------|----------------|
| REST API | FastAPI | Exposes `/search`, `/ingest`, and health endpoints |
| Async Workers | Celery + Redis | Runs background ingestion and embedding jobs |
| Vector Store | Qdrant | Stores 384-dimensional embeddings + metadata |
| User Interface | Streamlit | Provides a web-based search and filter UI |
| Embeddings | sentence-transformers | Produces dense semantic vectors |
| Reranking | FlashRank | Cross-encoder reranking for top results |
| Containerization | Docker + Docker Compose | Orchestrates services locally |

---

## 4. Key Features

### 4.1 Hybrid Search with Reciprocal Rank Fusion
- Combines results from **dense vector search** (cosine similarity) and **sparse BM25 search**.
- Uses **Reciprocal Rank Fusion (RRF)** to merge ranked lists without requiring calibrated scores.
- Tunable fusion parameters to balance semantic vs. keyword relevance.

### 4.2 Cross-Encoder Reranking for Precision
- Top candidates from the fused result set are passed through a **cross-encoder reranker** (FlashRank with `ms-marco-MiniLM-L-12-v2`).
- Improves precision by re-scoring passages in query-context pairs.

### 4.3 Section-Aware Document Chunking
- PubMed abstracts are decomposed into structured sections:
  - **BACKGROUND**
  - **METHODS**
  - **RESULTS**
  - **CONCLUSION**
- Preserves domain structure so users can filter by section and so embeddings focus on semantically coherent passages.

### 4.4 Async Data Ingestion Pipeline
- PubMed abstracts are downloaded from HuggingFace (`scientific_papers` dataset).
- Chunking and embedding are executed as **Celery background tasks**.
- The pipeline stores both vectors and rich metadata (title, section, source article ID, raw text).

### 4.5 Real-Time Search
- Query embedding, dense retrieval, fusion, and reranking complete in **~6 seconds** for the current 1000-document corpus (see Performance section below).
- Results are returned with relevance scores, section labels, and source metadata.

---

## 5. Tech Stack

| Layer | Technology | Version / Model |
|-------|------------|-----------------|
| Language | Python | 3.12 |
| Web Framework | FastAPI | Latest stable |
| Task Queue | Celery + Redis | Latest stable |
| Vector Database | Qdrant | Latest stable |
| Embedding Model | sentence-transformers | `all-MiniLM-L6-v2` (384-dim) |
| Reranker | FlashRank | `ms-marco-MiniLM-L-12-v2` |
| UI Framework | Streamlit | Latest stable |
| Containers | Docker + Docker Compose | Latest stable |
| Dataset Source | HuggingFace Datasets | `scientific_papers` / PubMed subset |

---

## 6. Data Pipeline

The ingestion pipeline moves raw medical abstracts into a searchable vector index through the following stages:

### 6.1 Data Source
- **Dataset:** HuggingFace `scientific_papers` dataset (PubMed split)
- **Content:** Biomedical article abstracts and metadata
- **Current scale:** 1000 documents for the POC
- **Scalable target:** 10,000+ documents

### 6.2 Processing Stages

```
Download PubMed abstracts
        │
        ▼
Parse structure (title + abstract text)
        │
        ▼
Chunk abstract into sections
(BACKGROUND, METHODS, RESULTS, CONCLUSION)
        │
        ▼
Generate 384-dimensional embeddings
(all-MiniLM-L6-v2)
        │
        ▼
Store vectors + metadata in Qdrant
```

### 6.3 Storage Schema
Each indexed point in Qdrant contains:
- `id`: unique chunk identifier
- `vector`: 384-dim dense embedding
- `payload`: metadata including title, section, source article ID, raw text, and ingestion timestamp

---

## 7. Search Pipeline

A single search request flows through the following steps:

1. **Embed Query**
   - The user query is encoded with `sentence-transformers/all-MiniLM-L6-v2` into a 384-dimensional vector.

2. **Dense Search**
   - Qdrant performs approximate nearest neighbor search using **cosine similarity**.
   - Returns top-30 candidate chunks.

3. **Sparse Search (BM25)** *(enabled)*
   - Keyword-based retrieval uses Qdrant named sparse vectors (`sparse`) built from TF-IDF term-weighted vectors over a shared corpus vocabulary.
   - Returns a separate top-k ranking that is fused with the dense results.

4. **Reciprocal Rank Fusion (RRF)**
   - Dense and sparse rankings are fused using the standard RRF formula:
     $$
     \text{RRF score}(d) = \sum_{r \in R} \frac{1}{k + r(d)}
     $$
     where $k$ is a constant (typically 60) and $r(d)$ is the rank of document $d$ in each result list.

5. **Cross-Encoder Reranking**
   - Top fused candidates are reranked with FlashRank (`ms-marco-MiniLM-L-12-v2`) for improved precision.

6. **Return Results**
   - The final top-5 results are returned to the user with metadata, section labels, and confidence scores.

```
Query ──▶ Embed ──▶ Dense Search (top 30)
                  ──▶ Sparse Search (BM25)
                              │
                              ▼
                    Reciprocal Rank Fusion
                              │
                              ▼
                    Cross-Encoder Rerank
                              │
                              ▼
                         Return Top 5
```

---

## 8. Current Status

| Milestone | Status | Notes |
|-----------|--------|-------|
| End-to-end pipeline working | ✅ | Ingestion through UI fully operational |
| 1000 PubMed documents ingested | ✅ | Stored as section-aware chunks in Qdrant |
| Search queries returning relevant results | ✅ | Dense + BM25 sparse-vector retrieval active, fused with RRF and reranked |
| True hybrid search with BM25 sparse vectors | ✅ | Qdrant collection configured with dense (384-dim) and named `sparse` vectors; all 1002 chunks indexed with both |
| UI functional with section filtering | ✅ | Streamlit interface supports query + section filter |
| Code pushed to GitHub | ✅ | Repository initialized and committed |
| Docker Compose orchestration | ✅ | Redis, Qdrant, API, worker, and UI services defined |

### Known Limitations
- Corpus size is intentionally small (1000 docs) for rapid iteration and validation.
- No authentication, logging, or cloud deployment yet.

---

## 9. Performance Baseline

Measured after scaling the corpus from 100 to 1000 documents on the local CPU-only stack (WSL2, Docker Desktop, Celery concurrency=2).

| Metric | Value |
|--------|-------|
| Documents downloaded | 1000 |
| Qdrant points (chunks) | 1002 |
| Ingestion time | ~462 seconds (~7.7 minutes) |
| Embedding batch size | 32 |
| Search cache TTL | 3600 seconds (1 hour) |
| Search cache key | SHA-256 of `query:top_k:section_filter` |
| Cache warming | Default set of 15 common medical queries, on-demand via `POST /cache/warm` |
| Cache admin endpoints | `GET /cache/stats`, `DELETE /cache/clear`, `POST /cache/warm` |

### Search Latency & Top Scores

| Query | Latency | Top Rerank Score |
|-------|---------|------------------|
| cancer chemotherapy | 6.86 s | 0.9651 |
| diabetes treatment | 7.48 s | 0.9518 |
| Alzheimer's biomarkers | 7.29 s | 0.9994 |
| cardiovascular risk | 8.38 s | 0.9937 |
| gene therapy | 6.19 s | 0.8027 |

**Notes:**
- All five queries returned 5 results.
- Four of five queries produced top rerank scores above 0.95. The "gene therapy" query returned a still-strong top score of 0.80, reflecting a smaller number of directly relevant abstracts in the sampled 1000-document subset.
- Cache-miss latency is dominated by the CPU cross-encoder reranker and TEI embedding call (~6–8 s).
- Cache-hit latency is **<100 ms** (observed ~5–15 ms for repeated identical queries) because results are served directly from Redis without re-running embedding, retrieval, or reranking.
- Redis caches search responses for 1 hour using a key derived from the normalized query, `top_k`, and optional `section_filter`.
- A `POST /cache/warm` endpoint pre-populates the cache with a default set of 15 common medical queries (or a user-supplied list), so the first user search for those queries is also sub-100 ms.
- Operational cache endpoints (`GET /cache/stats`, `DELETE /cache/clear`) allow runtime cache inspection and eviction.

---

## 10. Next Steps / Future Enhancements

### 9.1 Cache Warming & Administration

A dedicated cache management layer keeps first-time queries fast:

- **`POST /cache/warm`** — pre-populates the cache with the default warm query set or a custom list. Skips already-cached keys. Returns `queries_warmed`, `already_cached`, `newly_cached`, and any `errors`.
- **`GET /cache/stats`** — returns `total_keys`, `cache_keys`, `memory_used`, `connected_clients`, `uptime_seconds`, `hit_rate`, `hits`, `misses`, and `avg_ttl_remaining`.
- **`DELETE /cache/clear`** — removes all `search_cache:*` keys and returns the number of cleared keys.
- **Startup warming** — optional; enabled by setting `CACHE_WARM_ON_STARTUP=true`. Runs in a background thread so application boot is not blocked.

---

## 10. Next Steps / Future Enhancements

### Near Term
1. **Scale to 10,000+ documents**
   - Extend ingestion pipeline and validate latency remains sub-second.
2. **Add monitoring and logging**
   - Track query latency, relevance scores, and ingestion throughput.

### Medium Term
3. **Add user authentication**
   - Protect ingest endpoints and enable per-user search history.
4. **Implement document similarity search**
   - Allow "find similar articles" from any result.
5. **Deploy to cloud (AWS/GCP)**
   - Containerize production build and deploy with managed Redis/Qdrant.

### Long Term
6. **Evaluate larger embedding models** for domain-specific medical performance.
7. **Add query expansion / MeSH term enrichment** to improve recall.
8. **Build A/B testing framework** for ranking strategy comparison.

---

## 11. Success Criteria

The POC is considered successful when the following criteria are met:

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Search latency | < 1 second | Average query round-trip time |
| Relevance scores | > 0.6 for top relevant queries | FlashRank / cross-encoder score |
| Real data processing | 1000+ PubMed abstracts indexed | Qdrant collection count |
| Production-ready architecture | Containerized, modular services | Docker Compose + service separation |
| Functional UI | Section filter + ranked results | Manual end-to-end test |
| Code quality | Documented, version-controlled | GitHub repository + README + SPEC |

---

## Appendix A — Glossary

| Term | Definition |
|------|------------|
| Dense retrieval | Vector similarity search over learned embeddings |
| Sparse retrieval | Keyword or inverted-index search (e.g., BM25) |
| RRF | Reciprocal Rank Fusion — a method for combining ranked lists |
| Cross-encoder | A Transformer model that scores query-document pairs jointly |
| Section-aware chunking | Splitting documents by structural sections rather than arbitrary length |
| Celery | Distributed task queue for Python |
| Qdrant | Open-source vector database |

---

## Appendix B — Document History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-06-28 | Initial specification for Medical Hybrid Search POC |

---

*End of Specification*
