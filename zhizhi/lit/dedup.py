"""文献与知识图谱去重维护。

论文身份按 DOI、规范化题名、PDF 内容哈希三层判断。合并只改数据库记录，
不会删除用户的 PDF 文件；全文、已完成和 pinned 记录优先保留。
"""
from __future__ import annotations

import hashlib
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any

from ..core import db

LOCK = threading.RLock()


def normalize_doi(doi: str) -> str:
    raw = unicodedata.normalize("NFKC", str(doi or "")).strip().lower()
    raw = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", raw)
    raw = re.sub(r"^doi\s*:\s*", "", raw)
    return raw.strip().rstrip(".,;)")


def normalize_title(title: str) -> str:
    raw = unicodedata.normalize("NFKC", str(title or "")).casefold()
    return "".join(ch for ch in raw if ch.isalnum())


def pdf_hash(path: str | Path) -> str:
    p = Path(path) if path else Path()
    if not path or not p.is_file():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(doi: str = "", title: str = "", path: str | Path = "") -> dict[str, str]:
    return {
        "doi_key": normalize_doi(doi),
        "title_key": normalize_title(title),
        "content_hash": pdf_hash(path),
    }


def refresh_identities(force_hash: bool = False) -> dict:
    rows = db.q("SELECT id,doi,title,path,doi_key,title_key,content_hash FROM papers")
    updated = hashed = 0
    c = db.conn()
    with c:
        for r in rows:
            doi_key = normalize_doi(r["doi"])
            title_key = normalize_title(r["title"])
            content_hash = r["content_hash"] or ""
            if r["path"] and (force_hash or not content_hash):
                content_hash = pdf_hash(r["path"])
                hashed += bool(content_hash)
            if (doi_key, title_key, content_hash) != (
                    r["doi_key"] or "", r["title_key"] or "", r["content_hash"] or ""):
                c.execute(
                    "UPDATE papers SET doi_key=?,title_key=?,content_hash=? WHERE id=?",
                    (doi_key, title_key, content_hash, r["id"]),
                )
                updated += 1
    return {"papers_scanned": len(rows), "identities_updated": updated,
            "pdfs_hashed": hashed}


def find_duplicate(doi: str = "", title: str = "", path: str | Path = "",
                   exclude_id: str = "") -> dict | None:
    keys = identity(doi, title, path)
    checks = (("doi_key", keys["doi_key"]),
              ("content_hash", keys["content_hash"]),
              ("title_key", keys["title_key"] if len(keys["title_key"]) >= 24 else ""))
    for column, value in checks:
        if not value:
            continue
        row = db.q1(
            f"SELECT id,title,doi,path,status,evidence_level FROM papers "
            f"WHERE {column}=? AND id!=? LIMIT 1", (value, exclude_id),
        )
        if row:
            out = dict(row)
            out["matched_by"] = column
            out.update(keys)
            return out
    if path:
        filename = Path(path).name.casefold()
        for row in db.q(
                "SELECT id,title,doi,path,status,evidence_level FROM papers "
                "WHERE id!=? AND COALESCE(path,'')!=''", (exclude_id,)):
            if Path(row["path"]).name.casefold() == filename:
                out = dict(row)
                out["matched_by"] = "pdf_filename"
                out.update(keys)
                return out
    return None


def _paper_score(r: Any) -> tuple:
    path_ok = bool(r["path"] and Path(r["path"]).is_file())
    return (
        int(r["evidence_level"] == "fulltext" and path_ok),
        int(r["status"] == "done"),
        int(r["pinned"] or 0),
        int(r["n_chunks"] or 0),
        int(bool(r["doi"])),
    )


def _best_text(rows: list[Any], field: str, fallback: str = "") -> str:
    values = [str(r[field] or "").strip() for r in rows if str(r[field] or "").strip()]
    if not values:
        return fallback
    if field == "title":
        human = [v for v in values if len(v) >= 20 and not re.search(r"s\d+\.0-|main$", v, re.I)]
        if human:
            values = human
    return max(values, key=len)


def merge_papers(canonical_id: str, duplicate_id: str) -> dict:
    if canonical_id == duplicate_id:
        return {"canonical_id": canonical_id, "merged": []}
    c = db.conn()
    with c:
        keep = c.execute("SELECT * FROM papers WHERE id=?", (canonical_id,)).fetchone()
        drop = c.execute("SELECT * FROM papers WHERE id=?", (duplicate_id,)).fetchone()
        if not keep or not drop:
            return {"error": "合并对象不存在", "canonical_id": canonical_id,
                    "duplicate_id": duplicate_id}

        rows = [keep, drop]
        keep_has_chunks = int(keep["n_chunks"] or 0) > 0
        drop_has_chunks = int(drop["n_chunks"] or 0) > 0
        keep_fulltext_path = bool(
            keep["evidence_level"] == "fulltext" and keep["path"]
            and Path(keep["path"]).is_file())
        # 已有待摄取全文时，不把摘要级 chunks 搬过去冒充全文学习结果。
        move_learned = drop_has_chunks and not keep_has_chunks and not keep_fulltext_path

        if move_learned:
            c.execute("UPDATE chunks SET paper_id=? WHERE paper_id=?",
                      (canonical_id, duplicate_id))
            c.execute("UPDATE claims SET paper_id=? WHERE paper_id=?",
                      (canonical_id, duplicate_id))
            c.execute("UPDATE kg_edges SET paper_id=? WHERE paper_id=?",
                      (canonical_id, duplicate_id))
        else:
            c.execute("DELETE FROM embeddings WHERE chunk_id IN "
                      "(SELECT id FROM chunks WHERE paper_id=?)", (duplicate_id,))
            c.execute("DELETE FROM chunks WHERE paper_id=?", (duplicate_id,))
            c.execute("DELETE FROM claims WHERE paper_id=?", (duplicate_id,))
            c.execute("DELETE FROM kg_edges WHERE paper_id=?", (duplicate_id,))

        old_paper_node = f"Paper:{duplicate_id.lower()}"
        new_paper_node = f"Paper:{canonical_id.lower()}"
        if move_learned:
            c.execute(
                "INSERT INTO kg_nodes(id,type,name,meta) VALUES(?,'Paper',?,?) "
                "ON CONFLICT(id) DO NOTHING",
                (new_paper_node, canonical_id, keep["meta"] or drop["meta"] or "{}"),
            )
            c.execute("UPDATE kg_edges SET src=? WHERE src=?", (new_paper_node, old_paper_node))
            c.execute("UPDATE kg_edges SET dst=? WHERE dst=?", (new_paper_node, old_paper_node))
        c.execute("DELETE FROM kg_nodes WHERE id=?", (old_paper_node,))
        c.execute("UPDATE tasks SET state='deleted',message='重复文献已合并到 ' || ? "
                  "WHERE kind='ingest_pdf' AND ref=?", (canonical_id, duplicate_id))

        chosen_path = keep["path"] or drop["path"] or ""
        chosen_status = drop["status"] if move_learned else keep["status"]
        chosen_chunks = int(drop["n_chunks"] or 0) if move_learned else int(keep["n_chunks"] or 0)
        chosen_meta = drop["meta"] if move_learned and drop["meta"] else keep["meta"]
        c.execute(
            "UPDATE papers SET doi=?,title=?,authors=?,year=?,journal=?,abstract=?,path=?,"
            "evidence_level=?,pinned=?,status=?,n_chunks=?,error=?,meta=?,done_at=? WHERE id=?",
            (
                _best_text(rows, "doi"), _best_text(rows, "title"),
                _best_text(rows, "authors"), keep["year"] or drop["year"],
                _best_text(rows, "journal"), _best_text(rows, "abstract"), chosen_path,
                "fulltext" if any(r["evidence_level"] == "fulltext" and r["path"] for r in rows)
                else (keep["evidence_level"] or drop["evidence_level"]),
                max(int(keep["pinned"] or 0), int(drop["pinned"] or 0)),
                chosen_status, chosen_chunks,
                None if chosen_status == "done" else (keep["error"] or drop["error"]),
                chosen_meta, keep["done_at"] or drop["done_at"], canonical_id,
            ),
        )
        c.execute("DELETE FROM papers WHERE id=?", (duplicate_id,))

    keys = identity(_best_text(rows, "doi"), _best_text(rows, "title"), chosen_path)
    db.ex("UPDATE papers SET doi_key=?,title_key=?,content_hash=? WHERE id=?",
          (keys["doi_key"], keys["title_key"], keys["content_hash"], canonical_id))
    return {"canonical_id": canonical_id, "merged": [duplicate_id]}


def _duplicate_groups() -> list[list[str]]:
    rows = db.q("SELECT id,doi_key,title_key,content_hash FROM papers")
    parent = {r["id"]: r["id"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    seen: dict[tuple[str, str], str] = {}
    for r in rows:
        keys = (("doi", r["doi_key"] or ""), ("hash", r["content_hash"] or ""))
        title_key = r["title_key"] or ""
        if len(title_key) >= 24:
            keys += (("title", title_key),)
        for kind, value in keys:
            if not value:
                continue
            token = (kind, value)
            if token in seen:
                union(r["id"], seen[token])
            else:
                seen[token] = r["id"]
    groups: dict[str, list[str]] = {}
    for pid in parent:
        groups.setdefault(find(pid), []).append(pid)
    return [ids for ids in groups.values() if len(ids) > 1]


def deduplicate_library(dry_run: bool = False) -> dict:
    """扫描并合并重复论文；不删除磁盘上的 PDF。"""
    with LOCK:
        identity_stats = refresh_identities()
        groups = _duplicate_groups()
        if dry_run:
            return {**identity_stats, "duplicate_groups": groups,
                    "papers_merged": 0, "dry_run": True}
        merged: list[dict] = []
        for ids in groups:
            rows = [db.q1("SELECT * FROM papers WHERE id=?", (pid,)) for pid in ids]
            rows = [r for r in rows if r]
            if len(rows) < 2:
                continue
            keep = max(rows, key=_paper_score)["id"]
            dropped = []
            for r in rows:
                if r["id"] == keep:
                    continue
                result = merge_papers(keep, r["id"])
                if not result.get("error"):
                    dropped.append(r["id"])
            if dropped:
                merged.append({"canonical_id": keep, "removed_records": dropped})
        from . import kg
        kg_stats = kg.canonicalize_existing_nodes()
        return {**identity_stats, "duplicate_groups": groups,
                "papers_merged": sum(len(x["removed_records"]) for x in merged),
                "merges": merged, **kg_stats, "pdf_files_deleted": 0}
