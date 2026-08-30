from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document


DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.I)
DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
NON_WORD = re.compile(r"[^a-z0-9\u4e00-\u9fff]+", re.I)


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    value = DOI_PREFIX.sub("", doi.strip()).strip().lower()
    return value.rstrip(".,; ") or None


def extract_doi(value: str | None) -> str | None:
    """Extract a DOI from a DOI string, URL, citation or HTML metadata value."""
    if not value:
        return None
    match = DOI_PATTERN.search(value)
    return normalize_doi(match.group(0)) if match else None


def normalize_openalex_id(openalex_id: str | None) -> str | None:
    if not openalex_id:
        return None
    return openalex_id.rstrip("/").rsplit("/", 1)[-1].upper()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    return NON_WORD.sub("", value)


def first_author_key(authors: list[dict[str, Any]] | list[str] | None) -> str:
    if not authors:
        return "unknown"
    first = authors[0]
    if isinstance(first, dict):
        first = str(first.get("name") or first.get("display_name") or first.get("family") or "unknown")
    return normalize_text(str(first)) or "unknown"


def title_author_fingerprint(title: str, authors: list[dict[str, Any]] | list[str] | None) -> str:
    payload = f"{normalize_text(title)}|{first_author_key(authors)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_duplicate(
    db: Session,
    knowledge_base_id: str,
    *,
    doi: str | None = None,
    openalex_id: str | None = None,
    fingerprint: str | None = None,
    file_sha256: str | None = None,
) -> tuple[Document | None, str | None]:
    checks = [
        ("doi", Document.doi_normalized, normalize_doi(doi)),
        ("openalex_id", Document.openalex_id, normalize_openalex_id(openalex_id)),
        ("title_author_fingerprint", Document.title_author_fingerprint, fingerprint),
        ("file_sha256", Document.file_sha256, file_sha256),
    ]
    for reason, column, value in checks:
        if not value:
            continue
        document = db.scalar(
            select(Document).where(Document.knowledge_base_id == knowledge_base_id, column == value)
        )
        if document:
            return document, reason
    return None, None


def known_identifiers(db: Session, knowledge_base_id: str) -> set[str]:
    rows = db.execute(
        select(Document.doi_normalized, Document.openalex_id, Document.title_author_fingerprint).where(
            Document.knowledge_base_id == knowledge_base_id
        )
    )
    values: set[str] = set()
    for doi, openalex, fingerprint in rows:
        values.update(item for item in (doi, openalex, fingerprint) if item)
    return values
