from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF

from .config import settings


def clean_text(text: str) -> str:
    """Normalize extracted PDF text while retaining sentence boundaries."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int | None = None,
                      overlap: int | None = None) -> list[str]:
    """Split text into overlapping character windows with word-boundary adjustment."""
    size = chunk_size or settings.chunk_size
    stride_overlap = settings.chunk_overlap if overlap is None else overlap
    if size < 200:
        raise ValueError("chunk_size must be at least 200 characters")
    if stride_overlap < 0 or stride_overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = clean_text(text)
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer a nearby whitespace boundary to avoid cutting words.
            boundary = text.rfind(" ", start + int(size * 0.75), end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - stride_overlap, start + 1)
    return chunks


def extract_pdf_chunks(pdf_bytes: bytes, filename: str) -> list[dict]:
    """Extract page-aware chunks from a PDF upload."""
    if not pdf_bytes:
        raise ValueError(f"{filename} is empty")
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError(f"{filename} does not appear to be a valid PDF")

    chunks: list[dict] = []
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Could not open {filename}: {exc}") from exc

    try:
        for page_index, page in enumerate(document):
            page_text = clean_text(page.get_text("text"))
            if not page_text:
                continue
            for chunk_index, chunk_text in enumerate(split_into_chunks(page_text)):
                identity = hashlib.sha256(
                    f"{filename}|{page_index + 1}|{chunk_index}|{chunk_text}".encode("utf-8")
                ).hexdigest()[:16]
                chunks.append({
                    "id": identity,
                    "text": chunk_text,
                    "source": Path(filename).name,
                    "page": page_index + 1,
                    "chunk": chunk_index + 1,
                })
    finally:
        document.close()

    if not chunks:
        raise ValueError(
            f"No selectable text was found in {filename}. It may be a scanned PDF; "
            "OCR is not included in this MVP. Try a text-searchable PDF."
        )
    return chunks


def validate_uploads(files: Iterable, max_file_mb: int = 25) -> list[str]:
    errors = []
    for uploaded in files:
        name = getattr(uploaded, "name", "uploaded file")
        data = uploaded.getvalue()
        if len(data) > max_file_mb * 1024 * 1024:
            errors.append(f"{name}: exceeds the {max_file_mb} MB per-file limit")
        elif not data.startswith(b"%PDF"):
            errors.append(f"{name}: only PDF files are supported")
    return errors
