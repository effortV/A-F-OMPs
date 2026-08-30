"""黑板：SQLite 持久层。四个 Agent 共享，所有产物可追溯。"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .config import CFG

SCHEMA = """
PRAGMA journal_mode=WAL;

-- ============ 文献层 ============
CREATE TABLE IF NOT EXISTS papers (
    id          TEXT PRIMARY KEY,
    source      TEXT,
    doi         TEXT,
    title       TEXT,
    authors     TEXT,
    year        INTEGER,
    journal     TEXT,
    abstract    TEXT,
    path        TEXT,
    evidence_level TEXT DEFAULT 'fulltext',
    pinned      INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'queued',
    n_chunks    INTEGER DEFAULT 0,
    error       TEXT,
    meta        TEXT,
    added_at    REAL,
    done_at     REAL,
    doi_key     TEXT,
    title_key   TEXT,
    content_hash TEXT
);
CREATE INDEX IF NOT EXISTS ix_papers_status ON papers(status);

CREATE TABLE IF NOT EXISTS chunks (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id TEXT,
    idx      INTEGER,
    section  TEXT,
    page     INTEGER,
    text     TEXT
);
CREATE INDEX IF NOT EXISTS ix_chunks_paper ON chunks(paper_id);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY,
    vec      BLOB
);

CREATE TABLE IF NOT EXISTS claims (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id   TEXT,
    descriptor TEXT,
    direction  TEXT,
    membrane   TEXT,
    scope      TEXT,
    statement  TEXT,
    quote      TEXT,
    page       INTEGER,
    confidence REAL
);
CREATE INDEX IF NOT EXISTS ix_claims_desc ON claims(descriptor);

CREATE TABLE IF NOT EXISTS kg_nodes (
    id    TEXT PRIMARY KEY,
    type  TEXT,
    name  TEXT,
    meta  TEXT
);
CREATE TABLE IF NOT EXISTS kg_edges (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    src      TEXT, dst TEXT, relation TEXT,
    paper_id TEXT, quote TEXT, weight REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS ix_edges_src ON kg_edges(src);

CREATE TABLE IF NOT EXISTS contradictions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    topic      TEXT,
    descriptor TEXT,
    side_a     TEXT, side_b TEXT,
    claim_ids  TEXT,
    status     TEXT DEFAULT 'open',
    note       TEXT,
    created_at REAL
);

CREATE TABLE IF NOT EXISTS lit_requests (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    question   TEXT, must_cover TEXT, stance TEXT,
    queries    TEXT, n_found INTEGER, n_ingested INTEGER,
    status     TEXT DEFAULT 'open',
    result     TEXT,
    created_at REAL
);

-- ============ 任务队列 ============
CREATE TABLE IF NOT EXISTS tasks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    kind       TEXT,
    ref        TEXT,
    label      TEXT,
    state      TEXT DEFAULT 'queued',
    progress   REAL DEFAULT 0.0,
    message    TEXT,
    pinned     INTEGER DEFAULT 0,
    created_at REAL, updated_at REAL
);
CREATE INDEX IF NOT EXISTS ix_tasks_state ON tasks(state);

CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);

-- ============ 发现层 ============
CREATE TABLE IF NOT EXISTS cards (
    id          TEXT PRIMARY KEY,
    kind        TEXT,
    engine      TEXT,
    title       TEXT,
    statement   TEXT,
    novelty     TEXT,
    payload     TEXT,
    prereg_hash TEXT,
    prereg      TEXT,
    l1_result   TEXT,
    l2_plan     TEXT,
    l3_plan     TEXT,
    status      TEXT DEFAULT 'proposed',
    review      TEXT,
    created_at  REAL, updated_at REAL
);

CREATE TABLE IF NOT EXISTS descriptors (
    name        TEXT PRIMARY KEY,
    card_id     TEXT,
    hypothesis  TEXT,
    code        TEXT,
    spec        TEXT,
    status      TEXT DEFAULT 'proposed',
    values_path TEXT,
    metrics     TEXT,
    created_at  REAL
);

CREATE TABLE IF NOT EXISTS model_runs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT, params TEXT, metrics TEXT, note TEXT, created_at REAL
);

-- ============ 对话与审计 ============
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, agent TEXT, title TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, role TEXT, content TEXT, extra TEXT, created_at REAL
);
CREATE INDEX IF NOT EXISTS ix_msg_sess ON messages(session_id);

CREATE TABLE IF NOT EXISTS memory (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    agent      TEXT, kind TEXT, content TEXT, created_at REAL
);

CREATE TABLE IF NOT EXISTS llm_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT, model TEXT, kind TEXT,
    prompt_tokens INTEGER, completion_tokens INTEGER, created_at REAL
);

CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT, tool TEXT, args TEXT, ok INTEGER, ms REAL, created_at REAL
);
"""

_local = threading.local()
_schema_lock = threading.Lock()


def _migrate(c: sqlite3.Connection) -> None:
    """轻量前向迁移，兼容已经存在的 zhizhi.db。"""
    cols = {r[1] for r in c.execute("PRAGMA table_info(papers)").fetchall()}
    for name in ("doi_key", "title_key", "content_hash"):
        if name not in cols:
            c.execute(f"ALTER TABLE papers ADD COLUMN {name} TEXT")
    c.execute("CREATE INDEX IF NOT EXISTS ix_papers_doi_key ON papers(doi_key)")
    c.execute("CREATE INDEX IF NOT EXISTS ix_papers_title_key ON papers(title_key)")
    c.execute("CREATE INDEX IF NOT EXISTS ix_papers_content_hash ON papers(content_hash)")
    c.commit()


def conn() -> sqlite3.Connection:
    """线程局部连接（Streamlit / worker 多线程安全）。"""
    c = getattr(_local, "conn", None)
    if c is None:
        Path(CFG.db_path).parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(CFG.db_path, timeout=60, check_same_thread=False)
        c.row_factory = sqlite3.Row
        with _schema_lock:
            c.executescript(SCHEMA)
            _migrate(c)
        _local.conn = c
    return c


def q(sql: str, args: Iterable = ()) -> list[sqlite3.Row]:
    return conn().execute(sql, tuple(args)).fetchall()


def q1(sql: str, args: Iterable = ()):
    return conn().execute(sql, tuple(args)).fetchone()


def ex(sql: str, args: Iterable = ()) -> sqlite3.Cursor:
    c = conn()
    cur = c.execute(sql, tuple(args))
    c.commit()
    return cur


def exmany(sql: str, rows: Iterable[Iterable]) -> None:
    c = conn()
    c.executemany(sql, [tuple(r) for r in rows])
    c.commit()


def rows_to_dicts(rows) -> list[dict]:
    return [dict(r) for r in rows]


# ---- kv 控制位 --------------------------------------------------------
def kv_get(k: str, default: Any = None) -> Any:
    r = q1("SELECT v FROM kv WHERE k=?", (k,))
    return json.loads(r["v"]) if r else default


def kv_set(k: str, v: Any) -> None:
    ex("INSERT INTO kv(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
       (k, json.dumps(v, ensure_ascii=False)))


# ---- 任务 -------------------------------------------------------------
def task_add(kind: str, ref: str, label: str, pinned: int = 0) -> int:
    now = time.time()
    r = q1("SELECT id FROM tasks WHERE kind=? AND ref=? AND state!='deleted'", (kind, ref))
    if r:
        return int(r["id"])
    cur = ex("INSERT INTO tasks(kind,ref,label,state,pinned,created_at,updated_at)"
             " VALUES(?,?,?,'queued',?,?,?)", (kind, ref, label, pinned, now, now))
    return int(cur.lastrowid)


def task_set(task_id: int, **kw) -> None:
    kw["updated_at"] = time.time()
    sets = ",".join(f"{k}=?" for k in kw)
    ex(f"UPDATE tasks SET {sets} WHERE id=?", (*kw.values(), task_id))


def audit(agent: str, tool: str, args: Any, ok: bool, ms: float) -> None:
    ex("INSERT INTO audit(agent,tool,args,ok,ms,created_at) VALUES(?,?,?,?,?,?)",
       (agent, tool, json.dumps(args, ensure_ascii=False, default=str)[:4000],
        int(ok), ms, time.time()))


def log_usage(agent: str, model: str, kind: str, pt: int, ct: int) -> None:
    ex("INSERT INTO llm_usage(agent,model,kind,prompt_tokens,completion_tokens,created_at)"
       " VALUES(?,?,?,?,?,?)", (agent, model, kind, pt, ct, time.time()))


def init() -> None:
    conn()


def jdict(raw, default: dict | None = None) -> dict:
    """安全地把数据库里的 JSON 文本字段读成 dict。

    历史上出现过 l1_result 被写成 JSON 字符串 '""' 的情况，
    json.loads 出来是 str，后面 .get() 直接 AttributeError。
    这里统一兜底：不是 dict 就返回默认值。
    """
    if raw in (None, "", b""):
        return dict(default or {})
    try:
        v = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except Exception:  # noqa: BLE001
        return dict(default or {})
    return v if isinstance(v, dict) else dict(default or {})


def jlist(raw, default: list | None = None) -> list:
    """同上，但用于列表字段。"""
    if raw in (None, "", b""):
        return list(default or [])
    try:
        v = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    except Exception:  # noqa: BLE001
        return list(default or [])
    return v if isinstance(v, list) else list(default or [])
