"""Cross-encoder reranking with FlashRank.

Reranks the top candidates from hybrid dense+sparse retrieval using a small,
CPU-friendly cross-encoder. The default model is `ms-marco-MiniLM-L-12-v2`.
"""

from __future__ import annotations

from typing import Any

from flashrank import Ranker, RerankRequest


DEFAULT_MODEL = "ms-marco-MiniLM-L-12-v2"


class FlashrankReranker:
    """Thin wrapper around FlashRank for reranking retrieval candidates."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._ranker = Ranker(model_name=model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank candidates and return the top_k results.

        Parameters
        ----------
        query: str
            Original search query.
        candidates: list[dict]
            Candidate chunks, each containing at least `text` and `id`.
        top_k: int
            Number of top reranked results to return.

        Returns
        -------
        list[dict]
            Reranked results with cross-encoder scores and original metadata.
        """
        if not candidates:
            return []

        # FlashRank expects a list of dicts with "id" and "text" keys.
        flashrank_passages = [
            {
                "id": cand.get("id") or cand.get("chunk_id") or idx,
                "text": cand.get("text", ""),
            }
            for idx, cand in enumerate(candidates)
        ]

        rerank_request = RerankRequest(
            query=query,
            passages=flashrank_passages,
        )
        ranked = self._ranker.rerank(rerank_request)

        # FlashRank returns sorted passages with a "score" field.
        top_ranked = ranked[:top_k]

        # Map back to the original candidate metadata.
        id_to_candidate = {
            (cand.get("id") or cand.get("chunk_id") or idx): cand
            for idx, cand in enumerate(candidates)
        }

        results = []
        for item in top_ranked:
            cand_id = item.get("id")
            original = id_to_candidate.get(cand_id, {})
            results.append(
                {
                    "id": cand_id,
                    "chunk_id": original.get("chunk_id"),
                    "doc_id": original.get("doc_id"),
                    "title": original.get("title"),
                    "section": original.get("section"),
                    "text": item.get("text", original.get("text", "")),
                    "rerank_score": item.get("score"),
                    "rrf_score": original.get("rrf_score"),
                }
            )

        return results


def rerank_results(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 5,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Convenience function to rerank hybrid search candidates.

    Parameters
    ----------
    query: str
        Search query.
    candidates: list[dict]
        Candidates from hybrid_search.
    top_k: int
        Number of final results (default 5).
    model_name: str
        FlashRank model name.

    Returns
    -------
    list[dict]
        Top reranked candidates with scores and metadata.
    """
    reranker = FlashrankReranker(model_name=model_name)
    return reranker.rerank(query, candidates, top_k=top_k)
