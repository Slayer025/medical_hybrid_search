"""Client for the HuggingFace Text Embeddings Inference (TEI) service.

Provides dense vector embeddings via a local or remote TEI HTTP endpoint.
Supports batch encoding with a configurable batch size and single-query encoding.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv


DEFAULT_BATCH_SIZE = 32
EMBED_TIMEOUT_SECONDS = 120


class TEIEmbedder:
    """Embedder that calls a TEI-compatible HTTP embedding endpoint.

    Parameters
    ----------
    url: str | None
        Base URL of the TEI service. Defaults to the TEI_URL env variable.
    batch_size: int
        Maximum number of texts to send per request.
    timeout: int
        Request timeout in seconds.
    """

    def __init__(
        self,
        url: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: int = EMBED_TIMEOUT_SECONDS,
    ) -> None:
        load_dotenv()
        self.url = (url or os.getenv("TEI_URL", "http://localhost:8001")).rstrip("/")
        self.embed_url = f"{self.url}/embed"
        self.batch_size = batch_size
        self.timeout = timeout

        self._session = requests.Session()

    def _post(self, texts: list[str]) -> list[list[float]]:
        """Send a batch of texts to TEI and return their dense vectors."""
        # TEI expects JSON {"inputs": [text1, text2, ...]}
        payload: dict[str, Any] = {"inputs": texts}
        response = self._session.post(
            self.embed_url,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        # TEI returns a list of float vectors when batching.
        if not isinstance(data, list):
            raise ValueError(f"Unexpected TEI response format: {type(data)}")

        return data

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts in batches of size `batch_size`.

        Parameters
        ----------
        texts: list[str]
            Input strings to embed.

        Returns
        -------
        list[list[float]]
            Dense vectors, one per input text.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._post(batch)

            if len(embeddings) != len(batch):
                raise ValueError(
                    f"TEI returned {len(embeddings)} embeddings for {len(batch)} inputs."
                )

            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Parameters
        ----------
        text: str
            Query text.

        Returns
        -------
        list[float]
            Query dense vector.
        """
        if not text or not text.strip():
            raise ValueError("Query text must be non-empty.")

        embeddings = self.embed_texts([text])
        return embeddings[0]

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self) -> "TEIEmbedder":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


if __name__ == "__main__":
    # Quick sanity check when run directly.
    sample_texts = [
        "The patient presented with acute chest pain.",
        "MRI revealed a small meningioma in the left temporal lobe.",
    ]

    with TEIEmbedder() as embedder:
        print("Embedding sample batch...")
        vectors = embedder.embed_texts(sample_texts)
        print(f"Batch shape: {len(vectors)} vectors, dim={len(vectors[0])}")

        print("Embedding single query...")
        query_vector = embedder.embed_query("What causes migraines?")
        print(f"Query dim={len(query_vector)}")
