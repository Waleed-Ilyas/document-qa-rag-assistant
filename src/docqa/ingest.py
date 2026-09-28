"""Download the PDFs, extract page text, clean it and split it into page-anchored chunks.

Run: python -m docqa.ingest      (downloads to data/raw/, writes artifacts/chunks.parquet)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def clean_text(text: str) -> str:
    """Normalise ligatures/quotes, undo end-of-line hyphenation, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate wrapped words
    text = re.sub(r"[ \t]+", " ", text)
    return text


def strip_running_lines(pages: list[str], min_share: float = 0.35) -> list[str]:
    """Remove headers/footers: short lines that repeat on a large share of a document's pages."""
    n = len(pages)
    counts: Counter[str] = Counter()
    for p in pages:
        seen = {ln.strip() for ln in p.splitlines() if 0 < len(ln.strip()) <= 90}
        counts.update(seen)
    running = {ln for ln, c in counts.items() if c >= max(3, min_share * n)}
    out = []
    for p in pages:
        kept = [ln for ln in p.splitlines() if ln.strip() not in running and not re.fullmatch(r"\s*[ivxlc\d]+\s*", ln)]
        out.append("\n".join(kept))
    return out


def split_sentences(text: str) -> list[str]:
    # paragraphs first (blank line or bullet), then sentences inside them
    blocks = re.split(r"\n\s*\n|\n(?=[••\-–]\s)", text)
    sents: list[str] = []
    for b in blocks:
        b = re.sub(r"\s*\n\s*", " ", b).strip()
        if not b:
            continue
        sents.extend(s.strip() for s in _SENT_SPLIT.split(b) if s.strip())
    return sents


def chunk_page(text: str, target: int = config.CHUNK_TARGET_CHARS, max_chars: int = config.CHUNK_MAX_CHARS,
               overlap: int = config.CHUNK_OVERLAP_SENTENCES) -> list[str]:
    """Greedy sentence packing to ~target chars with a small sentence overlap."""
    sents = split_sentences(text)
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    for s in sents:
        if size + len(s) > max_chars and cur:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            size = sum(len(x) + 1 for x in cur)
        cur.append(s)
        size += len(s) + 1
        if size >= target:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            size = sum(len(x) + 1 for x in cur)
    if cur and (not chunks or " ".join(cur) != chunks[-1]):
        tail = " ".join(cur)
        if chunks and len(tail) < config.MIN_CHUNK_CHARS:
            chunks[-1] = chunks[-1] + " " + tail if tail not in chunks[-1] else chunks[-1]
        else:
            chunks.append(tail)
    return [c for c in chunks if len(c) >= 60]


def extract_pages(pdf: Path) -> tuple[list[str], list[str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf))
    raw = [clean_text(p.extract_text() or "") for p in reader.pages]
    labels = [str(x) for x in reader.page_labels]
    return strip_running_lines(raw), labels


def build_chunks(raw_dir: Path = config.RAW) -> pd.DataFrame:
    rows = []
    for doc in config.DOCS:
        pages, labels = extract_pages(raw_dir / doc.filename)
        for i, (text, label) in enumerate(zip(pages, labels, strict=True), start=1):
            for j, chunk in enumerate(chunk_page(text)):
                rows.append({
                    "chunk_id": f"{doc.doc_id}:{i}:{j}", "doc_id": doc.doc_id, "page": i,
                    "page_label": label, "text": chunk, "n_chars": len(chunk),
                })
    return pd.DataFrame(rows)


def main() -> None:
    from .download import fetch_all

    fetch_all()
    df = build_chunks()
    config.ARTIFACTS.mkdir(exist_ok=True)
    df.to_parquet(config.ARTIFACTS / "chunks.parquet", index=False)
    per_doc = df.groupby("doc_id").agg(chunks=("chunk_id", "count"), pages=("page", "nunique"),
                                       chars=("n_chars", "sum"))
    print(per_doc.to_string())
    print(f"total: {len(df)} chunks, median {df.n_chars.median():.0f} chars")


if __name__ == "__main__":
    main()
