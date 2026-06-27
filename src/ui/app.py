"""Streamlit UI for the Medical Hybrid Search POC.

Provides a minimal, retrieval-focused interface that calls the FastAPI /search
endpoint and displays reranked results with full abstract context.
"""

from __future__ import annotations

import os

import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SEARCH_ENDPOINT = f"{API_BASE_URL}/search"

SECTION_OPTIONS = ["All", "BACKGROUND", "METHODS", "RESULTS", "CONCLUSION"]
PREVIEW_LENGTH = 200


def _search(query: str, section_filter: str | None) -> dict | None:
    """Send a search request to the FastAPI backend."""
    payload = {
        "query": query,
        "section_filter": section_filter,
    }

    try:
        response = requests.post(SEARCH_ENDPOINT, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as exc:
        st.error(f"Unable to reach the FastAPI backend at {API_BASE_URL}. Is it running?")
        return None
    except requests.exceptions.Timeout:
        st.warning("The backend took too long to respond. Please try again.")
        return None
    except requests.exceptions.HTTPError as exc:
        st.error(f"Search request failed: {exc}")
        return None


def main() -> None:
    """Render the Streamlit search interface."""
    st.set_page_config(
        page_title="Medical Hybrid Search POC",
        page_icon="🔬",
        layout="wide",
    )

    st.title("Medical Hybrid Search POC")
    st.caption("Hybrid dense + sparse retrieval with BM25 and cross-encoder reranking")

    with st.sidebar:
        st.header("Filters")
        section_filter = st.selectbox(
            "Section Filter",
            options=SECTION_OPTIONS,
            index=0,
            help="Restrict results to a specific medical section, or search across all sections.",
        )

    query = st.text_input(
        "Search query",
        placeholder="e.g., Does metformin reduce cardiovascular risk in type 2 diabetes?",
    )

    search_clicked = st.button("Search", type="primary", use_container_width=True)

    if search_clicked or (query and query != st.session_state.get("last_query")):
        st.session_state["last_query"] = query

    if search_clicked:
        if not query or not query.strip():
            st.warning("Please enter a search query.")
            return

        with st.spinner("Searching..."):
            section = section_filter if section_filter != "All" else None
            data = _search(query.strip(), section)

        if data is None:
            return

        results = data.get("results", [])

        if not results:
            st.info("No results found. Try a different query or section filter.")
            return

        st.subheader(f"Top {len(results)} results")

        for rank, result in enumerate(results, start=1):
            title = result.get("title") or "Untitled"
            section_name = result.get("section") or "Unknown"
            score = result.get("rerank_score")
            text = result.get("text") or ""

            preview = text[:PREVIEW_LENGTH]
            if len(text) > PREVIEW_LENGTH:
                preview = preview + "…"

            with st.container(border=True):
                st.markdown(f"**#{rank}** — `{section_name}` — Score: `{score:.4f}`")
                st.markdown(f"**{title}**")
                st.markdown(preview)

                with st.expander("View Full Abstract Context"):
                    st.markdown(text)


if __name__ == "__main__":
    main()
