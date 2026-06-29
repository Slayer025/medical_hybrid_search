# Medical Hybrid Search POC

A production-grade medical literature search engine that combines semantic understanding with keyword matching to find relevant PubMed research papers.

## 🚀 Quick Access

The main entry point for using the application is the **Streamlit web UI**.

**Open the UI:** [http://localhost:8501](http://localhost:8501)

**Start the UI:**
```bash
streamlit run src/ui/app.py
```

> Make sure the FastAPI backend and Celery worker are running first (see [Running the System](#running-the-system) below).

## 📄 Documentation

For a comprehensive technical overview, see the **[Project Specification](docs/SPEC.md)**, which covers:

- System architecture and component responsibilities
- Full technology stack and model choices
- Feature descriptions and design rationale
- End-to-end data and search pipeline details
- Current status, success criteria, and future roadmap

## Features

- **Hybrid Search**: Combines dense vector search (semantic) with sparse BM25 search (keyword)
- **Cross-Encoder Reranking**: Uses FlashRank for precision scoring
- **Async Processing**: Celery + Redis for background task handling
- **Vector Database**: Qdrant for fast similarity search
- **Section Filtering**: Search within BACKGROUND, METHODS, RESULTS, CONCLUSION sections
- **Streamlit UI**: Clean web interface for search

## Architecture

```
User (Streamlit UI) → FastAPI → Redis Queue → Celery Worker → Qdrant
                                                        ↓
                                          Sentence-Transformers Embeddings
```

## Tech Stack

- **Backend**: FastAPI, Celery, Redis
- **Vector DB**: Qdrant
- **Embeddings**: BAAI/bge-small-en-v1.5 via Text Embeddings Inference (TEI)
- **Reranking**: FlashRank (ms-marco-MiniLM-L-12-v2)
- **UI**: Streamlit
- **Data**: HuggingFace `scientific_papers` / `pubmed_qa` datasets

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (for Qdrant, Redis, and TEI)
- WSL2 (if on Windows)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd medical_hybrid_search

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start infrastructure
docker compose up -d

# Initialize Qdrant collection
python src/ingestion/qdrant_setup.py

# Download sample data
python src/ingestion/downloader.py --count 1000
```

## Running the System

Open 3 terminals:

### Terminal 1 - FastAPI Backend

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2 - Celery Worker

```bash
celery -A src.tasks.celery_app worker --loglevel=info --concurrency=2
```

### Terminal 3 - Ingest Data & Start UI

```bash
# Trigger ingestion
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"filename": "pubmed_subset_1000.jsonl"}'

# Start UI
streamlit run src/ui/app.py
```

Open http://localhost:8501 in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/health` | Health check |
| POST   | `/ingest` | Queue document ingestion |
| POST   | `/search` | Hybrid search with reranking |

### Search Example

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "metformin cardiovascular risk diabetes", "section_filter": "RESULTS"}'
```

## Project Structure

```
medical_hybrid_search/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── ingestion/        # Data download, chunking, embeddings, Qdrant setup
│   ├── retrieval/        # Hybrid search and reranking logic
│   ├── tasks/            # Celery async tasks
│   └── ui/               # Streamlit interface
├── data/
│   ├── raw/              # Downloaded PubMed abstracts
│   └── processed/        # Chunked documents
├── docker-compose.yml    # Qdrant, Redis, TEI services
├── requirements.txt
└── README.md
```

## Search Pipeline

1. **Embed Query**: Convert query to 384-dim dense vector via TEI
2. **Dense Search**: Find top 30 similar vectors (cosine similarity)
3. **Sparse Search**: BM25 keyword matching via Qdrant named sparse vectors (active)
4. **Reciprocal Rank Fusion**: Combine both result lists with `score = Σ 1 / (60 + rank)`
5. **Cross-Encoder Reranking**: Score top 20 candidates with FlashRank
6. **Return Top 5**: With scores, titles, and text snippets

## Scaling

- Increase document count: `python src/ingestion/downloader.py --count 10000`
- Add more Celery workers for parallel processing
- Qdrant scales to millions of vectors
