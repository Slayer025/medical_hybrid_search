"""Section-aware chunking for PubMed medical abstracts.

This module parses standard medical section headers (BACKGROUND, OBJECTIVE,
METHODS, RESULTS, CONCLUSION, etc.) and produces semantic chunks respecting
section boundaries, size thresholds, and sentence boundaries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


try:
    from nltk.tokenize import sent_tokenize

    _NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback if nltk not installed
    _NLTK_AVAILABLE = False


# Standard section headers and their normalized names.
SECTION_PATTERNS: dict[str, list[str]] = {
    "OBJECTIVE": ["objective", "objectives", "aim", "aims", "purpose"],
    "BACKGROUND": ["background", "introduction"],
    "METHODS": ["methods", "methodology", "materials and methods", "patients and methods"],
    "RESULTS": ["results", "findings"],
    "CONCLUSION": ["conclusion", "conclusions", "summary"],
}

# Regex for matching any known section header line.
_SECTION_REGEX = re.compile(
    r"^\s*(?:\d+[.):-]?\s*)?("
    + "|".join(re.escape(label) for labels in SECTION_PATTERNS.values() for label in labels)
    + r")\s*[:)\]]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Regex for fallback splitting when NLTK is unavailable.
_SENTENCE_REGEX = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])",
)

# Heuristic token count: ~1.3 tokens per word for English biomedical text.
TOKENS_PER_WORD = 1.3
MAX_TOKENS = 512
MIN_TOKENS = 50


def _approx_token_count(text: str) -> int:
    """Return an approximate token count for a text fragment."""
    return int(len(text.split()) * TOKENS_PER_WORD)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK when available, otherwise regex."""
    if _NLTK_AVAILABLE:
        try:
            return sent_tokenize(text)
        except Exception:  # pragma: no cover - defensive
            pass
    return [s.strip() for s in _SENTENCE_REGEX.split(text) if s.strip()]


def _normalize_section_name(header: str) -> str:
    """Map a raw header string to a canonical section name."""
    cleaned = header.strip().rstrip(":;.)").lower()
    for canonical, labels in SECTION_PATTERNS.items():
        if cleaned in labels:
            return canonical
    return cleaned.upper()


def _parse_sections(text: str) -> list[tuple[str, str]]:
    """Parse medical section headers and return (section_name, body) pairs."""
    if not text or not text.strip():
        return []

    matches = list(_SECTION_REGEX.finditer(text))
    if not matches:
        return [("FULL_TEXT", text.strip())]

    sections: list[tuple[str, str]] = []
    cursor = 0

    for i, match in enumerate(matches):
        start, end = match.span()
        # Any text before the first header is treated as an INTRO/FULL_TEXT prefix.
        if i == 0 and start > 0:
            prefix = text[:start].strip()
            if prefix:
                sections.append(("FULL_TEXT", prefix))

        header = match.group(1)
        section_name = _normalize_section_name(header)

        section_start = end
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[section_start:section_end].strip()

        if body:
            sections.append((section_name, body))
        cursor = section_end

    # Append any trailing text after the last section header.
    trailing = text[cursor:].strip()
    if trailing and not sections:
        sections.append(("FULL_TEXT", trailing))

    return sections


def _merge_small_sections(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge sections under MIN_TOKENS with adjacent sections."""
    if not sections:
        return []

    merged: list[tuple[str, str]] = []

    for section_name, body in sections:
        if _approx_token_count(body) < MIN_TOKENS and merged:
            prev_name, prev_body = merged[-1]
            merged[-1] = (
                f"{prev_name}_{section_name}",
                f"{prev_body}\n\n{body}",
            )
        else:
            merged.append((section_name, body))

    return merged


def _split_section(section_name: str, body: str) -> list[str]:
    """Return one or more chunks for a section respecting sentence boundaries."""
    if _approx_token_count(body) <= MAX_TOKENS:
        return [body]

    sentences = _split_sentences(body)
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _approx_token_count(sentence)

        if sentence_tokens > MAX_TOKENS:
            # Sentence itself is oversized; flush current chunk first.
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0
            chunks.append(sentence)
            continue

        if current_tokens + sentence_tokens > MAX_TOKENS and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_text(text: str, title: str, doc_id: str) -> list[dict]:
    """Split a single PubMed document into semantically coherent chunks.

    Parameters
    ----------
    text: str
        Document text to chunk (full text or abstract). May contain section headers.
    title: str
        Document title.
    doc_id: str
        Original PubMed ID.

    Returns
    -------
    list[dict]
        List of chunk dictionaries with metadata.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sections = _parse_sections(text)
    sections = _merge_small_sections(sections)

    chunks: list[dict] = []
    section_counters: dict[str, int] = {}

    for section_name, body in sections:
        section_chunks = _split_section(section_name, body)
        for chunk_text in section_chunks:
            section_counters[section_name] = section_counters.get(section_name, 0) + 1
            index = section_counters[section_name]
            chunk_id = f"{doc_id}_{section_name.lower()}_{index:02d}"

            chunks.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "section": section_name,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                }
            )

    return chunks


def chunk_documents(input_jsonl_path: str | Path, output_jsonl_path: str | Path) -> int:
    """Read raw PubMed JSONL documents, chunk them, and write chunks to JSONL.

    Parameters
    ----------
    input_jsonl_path: str | Path
        Path to the raw document JSONL file.
    output_jsonl_path: str | Path
        Path where chunk JSONL will be written.

    Returns
    -------
    int
        Number of chunks written.
    """
    input_path = Path(input_jsonl_path)
    output_path = Path(output_jsonl_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_docs = 0

    with output_path.open("w", encoding="utf-8") as out_fh:
        with input_path.open("r", encoding="utf-8") as in_fh:
            for line_number, line in enumerate(in_fh, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Skipping malformed JSON at line {line_number}: {exc}")
                    continue

                total_docs += 1
                doc_id = str(doc.get("pubmed_id") or f"doc_{line_number}")
                title = doc.get("title", "")
                text = doc.get("abstract") or doc.get("text") or doc.get("full_text", "")

                if not text.strip():
                    print(f"Skipping document {doc_id}: no text content.")
                    continue

                chunks = chunk_text(text, title, doc_id)
                for chunk in chunks:
                    out_fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total_chunks += 1

                if total_docs % 100 == 0:
                    print(f"  chunked {total_docs} documents -> {total_chunks} chunks")

    print(
        f"Chunking complete: {total_docs:,} documents produced {total_chunks:,} chunks.\n"
        f"Output written to {output_path}"
    )
    return total_chunks


def main() -> int:
    """CLI entrypoint for chunking a raw PubMed JSONL file."""
    import argparse

    parser = argparse.ArgumentParser(description="Section-aware chunking for PubMed abstracts.")
    parser.add_argument("--input", required=True, help="Path to raw PubMed JSONL file.")
    parser.add_argument("--output", required=True, help="Path to output chunk JSONL file.")
    args = parser.parse_args()

    try:
        chunk_documents(args.input, args.output)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
