"""Initialize the Qdrant collection for hybrid medical document search.

Configures:
- Dense vectors: 384-dimensional, Cosine distance (unnamed/default).
- Sparse vectors: named "sparse" vector for TF-IDF/BM25-style retrieval.

Run with --force to drop and recreate an existing collection.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse


COLLECTION_NAME = "medical_documents"
DENSE_VECTOR_SIZE = 384
DENSE_DISTANCE = models.Distance.COSINE
SPARSE_VECTOR_NAME = "sparse"


def get_qdrant_client() -> QdrantClient:
    """Create a Qdrant client from environment variables."""
    load_dotenv()

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))

    return QdrantClient(host=host, port=port, grpc_port=grpc_port, prefer_grpc=False)


def collection_exists(client: QdrantClient, name: str) -> bool:
    """Check whether a collection already exists."""
    try:
        client.get_collection(name)
        return True
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            return False
        raise


def setup_collection(client: QdrantClient, force: bool = False) -> None:
    """Create the hybrid medical_documents collection.

    Parameters
    ----------
    client: QdrantClient
        Configured Qdrant client.
    force: bool
        If True, delete the collection if it already exists before recreating.
    """
    exists = collection_exists(client, COLLECTION_NAME)

    if exists:
        if force:
            print(f"Dropping existing collection '{COLLECTION_NAME}'...")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(
                f"Collection '{COLLECTION_NAME}' already exists. "
                "Use --force to recreate it."
            )
            return

    print(f"Creating hybrid collection '{COLLECTION_NAME}'...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=DENSE_VECTOR_SIZE,
            distance=DENSE_DISTANCE,
        ),
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                ),
            )
        },
    )

    # Payload indexes are optional but helpful for filtering and debugging.
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="doc_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="section",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="title",
        field_schema=models.PayloadSchemaType.TEXT,
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="chunk_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )

    print(f"Collection '{COLLECTION_NAME}' is ready for hybrid search.")


def main() -> int:
    """CLI entrypoint for Qdrant collection setup."""
    parser = argparse.ArgumentParser(
        description="Initialize the Qdrant hybrid collection for medical document search."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate the collection if it already exists.",
    )
    args = parser.parse_args()

    try:
        client = get_qdrant_client()
        setup_collection(client, force=args.force)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            client.close()
        except NameError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
