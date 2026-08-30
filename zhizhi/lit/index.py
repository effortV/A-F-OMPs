"""混合检索：BM25 + bge-m3 余弦，RRF 融合。不依赖 faiss / rank_bm25。"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter

import numpy as np

from ..core import db
from ..core.config import CFG
from ..core.llm import LLM

_TOK = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}|\d+\.?\d*")
_lock = threading.Lock()
_vec_cache: dict = {"n": -1, "ids": None, "mat": None}
_bm25_cache: dict = {"n": -1, "ids": None, "df": None, "tf": None, "len": None, "avg": 0.0}


def tokenize(text: str) -> list[str]:
    t = text.lower()
    toks = [w for w in _TOK.findall(t) if len(w) > 1]
    # 中文按字 + 二元组
    cjk = [ch for ch in t if "一" <= ch <= "鿿"]
    toks += cjk + ["".join(p) for p in zip(cjk, cjk[1:])]
    return toks


# ---------------- 向量 ----------------
def store_embeddings(chunk_ids: list[int], vecs: np.ndarray) -> None:
    db.exmany("INSERT OR REPLACE INTO embeddings(chunk_id,vec) VALUES(?,?)",
              [(int(cid), v.astype(np.float32).tobytes())
               for cid, v in zip(chunk_ids, vecs)])
    with _lock:
        _vec_cache["n"] = -1


def _load_vectors():
    with _lock:
        cnt = db.q1("SELECT COUNT(*) c FROM embeddings")["c"]
        if _vec_cache["n"] == cnt and _vec_cache["mat"] is not None:
            return _vec_cache["ids"], _vec_cache["mat"]
        rows = db.q("SELECT chunk_id, vec FROM embeddings ORDER BY chunk_id")
        if not rows:
            _vec_cache.update({"n": 0, "ids": np.array([], int),
                               "mat": np.zeros((0, 1024), np.float32)})
            return _vec_cache["ids"], _vec_cache["mat"]
        ids = np.array([r["chunk_id"] for r in rows], dtype=int)
        mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        _vec_cache.update({"n": cnt, "ids": ids, "mat": mat})
        return ids, mat


def vector_search(query: str, top_k: int = 30,
                  paper_ids: list[str] | None = None) -> list[tuple[int, float]]:
    ids, mat = _load_vectors()
    if len(ids) == 0:
        return []
    qv = LLM("retrieval").embed([query])[0]
    sims = mat @ qv
    if paper_ids:
        keep = {r["id"] for r in db.q(
            "SELECT id FROM chunks WHERE paper_id IN (%s)" %
            ",".join("?" * len(paper_ids)), paper_ids)}
        mask = np.array([i in keep for i in ids])
        sims = np.where(mask, sims, -1e9)
    order = np.argsort(-sims)[:top_k]
    return [(int(ids[i]), float(sims[i])) for i in order if sims[i] > -1e8]


# ---------------- BM25 ----------------
def _load_bm25():
    with _lock:
        cnt = db.q1("SELECT COUNT(*) c FROM chunks")["c"]
        if _bm25_cache["n"] == cnt and _bm25_cache["tf"] is not None:
            return _bm25_cache
        rows = db.q("SELECT id, text FROM chunks ORDER BY id")
        ids, tfs, lens = [], [], []
        df: Counter = Counter()
        for r in rows:
            toks = tokenize(r["text"])
            c = Counter(toks)
            ids.append(r["id"])
            tfs.append(c)
            lens.append(len(toks))
            df.update(c.keys())
        _bm25_cache.update({"n": cnt, "ids": ids, "df": df, "tf": tfs, "len": lens,
                            "avg": (sum(lens) / len(lens)) if lens else 1.0})
        return _bm25_cache


def bm25_search(query: str, top_k: int = 30) -> list[tuple[int, float]]:
    c = _load_bm25()
    if not c["ids"]:
        return []
    N, avg = len(c["ids"]), c["avg"] or 1.0
    qtok = set(tokenize(query))
    k1, b = 1.5, 0.75
    scores = np.zeros(N)
    for t in qtok:
        n = c["df"].get(t, 0)
        if n == 0:
            continue
        idf = math.log(1 + (N - n + 0.5) / (n + 0.5))
        for i in range(N):
            f = c["tf"][i].get(t, 0)
            if f:
                dl = c["len"][i] or 1
                scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avg))
    order = np.argsort(-scores)[:top_k]
    return [(int(c["ids"][i]), float(scores[i])) for i in order if scores[i] > 0]


# ---------------- 融合 ----------------
def hybrid_search(query: str, top_k: int | None = None,
                  paper_ids: list[str] | None = None) -> list[dict]:
    top_k = top_k or int(CFG.get("literature.retrieve_top_k", 8))
    pool = max(top_k * 5, 30)
    try:
        v = vector_search(query, pool, paper_ids)
    except Exception:  # noqa: BLE001  向量服务不可用时降级为纯 BM25
        v = []
    bm = bm25_search(query, pool)
    rr: dict[int, float] = {}
    for rank, (cid, _) in enumerate(v):
        rr[cid] = rr.get(cid, 0.0) + 1.0 / (60 + rank)
    for rank, (cid, _) in enumerate(bm):
        rr[cid] = rr.get(cid, 0.0) + 1.0 / (60 + rank)
    if paper_ids:
        allow = {r["id"] for r in db.q(
            "SELECT id FROM chunks WHERE paper_id IN (%s)" %
            ",".join("?" * len(paper_ids)), paper_ids)}
        rr = {k: s for k, s in rr.items() if k in allow}
    best = sorted(rr.items(), key=lambda kv: -kv[1])[:top_k]
    out = []
    for cid, score in best:
        r = db.q1("SELECT c.id,c.paper_id,c.section,c.page,c.text,p.title,p.year,p.doi "
                  "FROM chunks c JOIN papers p ON p.id=c.paper_id WHERE c.id=?", (cid,))
        if r:
            out.append({"chunk_id": cid, "score": round(score, 5),
                        "paper_id": r["paper_id"], "title": r["title"],
                        "year": r["year"], "doi": r["doi"],
                        "section": r["section"], "page": r["page"],
                        "text": r["text"]})
    return out


def invalidate() -> None:
    with _lock:
        _vec_cache["n"] = -1
        _bm25_cache["n"] = -1
