"""文献摄取工作线程：队列驱动，支持 开始 / 暂停 / 继续 / 重试 / 删除。

核心 59 篇 pinned=1，只能暂停不能删除。
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
import traceback
from pathlib import Path

import numpy as np

from ..core import db, remote
from ..core.config import CFG
from ..core.llm import LLM
from . import dedup, extract, index, pdf

WORKER_API_VERSION = 2
CTL = "lit_worker"           # running | paused | stopped
_threads: list[threading.Thread] = []
_lock = threading.Lock()
_claim_lock = threading.Lock()

FAILURE_LABELS = {
    "missing_text": "缺少 PDF/摘要",
    "page_format": "页码格式兼容错误",
    "needs_ocr": "扫描件需 OCR",
    "model_output": "模型输出格式错误",
    "timeout": "接口超时",
    "other": "其他错误",
}


def failure_category(error: str = "") -> str:
    """Map a persisted ingestion error to a stable, user-facing category."""
    text = str(error or "").strip()
    low = text.casefold()
    if ("无 pdf" in low or "无可用文本" in low or "无摘要" in low
            or "no pdf" in low or "no abstract" in low):
        return "missing_text"
    if ("invalid literal for int" in low and re.search(r"(?:p\.|page)\s*\d+", low)):
        return "page_format"
    if "ocr" in low or "扫描件" in low or "文本层缺失" in low:
        return "needs_ocr"
    if ("无法解析为 json" in low or "jsondecode" in low
            or "不是 json" in low or "schema" in low):
        return "model_output"
    if "timeout" in low or "timed out" in low or "超时" in low:
        return "timeout"
    return "other"


# ---------------- 语料装载 ----------------
def local_paper_id(path: Path) -> str:
    """优先按 PDF 内容生成稳定 ID；同一文件改名后也不会再次学习。"""
    content_hash = dedup.pdf_hash(path)
    if content_hash:
        return "L" + content_hash[:14]
    return "L" + hashlib.md5(path.name.encode("utf-8")).hexdigest()[:14]


def bootstrap_core_corpus() -> dict:
    if remote.enabled():
        return remote.call("module", "worker.bootstrap_core_corpus")
    """把 reference-59 目录的 PDF 注册为核心语料（pinned，不可删）。"""
    d = CFG.reference_dir
    files = sorted(d.glob("*.pdf")) if d.exists() else []
    dedup.refresh_identities()
    added = skipped_duplicate = 0
    for f in files:
        pid = local_paper_id(f)
        dup = dedup.find_duplicate(title=f.stem, path=f, exclude_id=pid)
        if dup:
            pid = dup["id"]
            skipped_duplicate += 1
        if db.q1("SELECT id FROM papers WHERE id=?", (pid,)):
            db.ex("UPDATE papers SET pinned=1, path=? WHERE id=?", (str(f), pid))
        else:
            keys = dedup.identity(title=f.stem, path=f)
            db.ex("INSERT INTO papers(id,source,title,path,pinned,status,evidence_level,"
                  "added_at,doi_key,title_key,content_hash) "
                  "VALUES(?,'local59',?,?,1,'queued','fulltext',strftime('%s','now'),?,?,?)",
                  (pid, f.stem[:250], str(f), keys["doi_key"], keys["title_key"],
                   keys["content_hash"]))
            added += 1
        db.task_add("ingest_pdf", pid, f.stem[:120], pinned=1)
    return {"core_dir": str(d), "pdf_found": len(files), "newly_registered": added,
            "skipped_duplicate": skipped_duplicate,
            "total_pinned": db.q1("SELECT COUNT(*) c FROM papers WHERE pinned=1")["c"]}


def scan_new_pdfs() -> dict:
    if remote.enabled():
        return remote.call("module", "worker.scan_new_pdfs")
    """扫描 store/pdf_new 里手动丢进来的 PDF。"""
    d = CFG.new_pdf_dir
    files = sorted(list(d.glob("*.pdf")))
    dedup.refresh_identities()
    added = skipped_duplicate = 0
    for f in files:
        pid = local_paper_id(f)
        if db.q1("SELECT id FROM papers WHERE path=?", (str(f),)):
            continue
        dup = dedup.find_duplicate(title=f.stem, path=f, exclude_id=pid)
        if dup:
            skipped_duplicate += 1
            if not dup.get("path"):
                db.ex("UPDATE papers SET path=?,evidence_level='fulltext' WHERE id=?",
                      (str(f), dup["id"]))
            continue
        if db.q1("SELECT id FROM papers WHERE id=?", (pid,)):
            db.ex("UPDATE papers SET path=? WHERE id=?", (str(f), pid))
            continue
        keys = dedup.identity(title=f.stem, path=f)
        db.ex("INSERT INTO papers(id,source,title,path,pinned,status,evidence_level,added_at)"
              " VALUES(?,'manual',?,?,0,'queued','fulltext',strftime('%s','now'))",
              (pid, f.stem[:250], str(f)))
        db.ex("UPDATE papers SET doi_key=?,title_key=?,content_hash=? WHERE id=?",
              (keys["doi_key"], keys["title_key"], keys["content_hash"], pid))
        db.task_add("ingest_pdf", pid, f.stem[:120])
        added += 1
    return {"dir": str(d), "pdf_found": len(files), "newly_registered": added,
            "skipped_duplicate": skipped_duplicate}


# ---------------- 单篇处理 ----------------
def _paused(task_id: int) -> bool:
    r = db.q1("SELECT state FROM tasks WHERE id=?", (task_id,))
    return (not r) or r["state"] in ("paused", "deleted")


def process_paper(paper_id: str, task_id: int | None = None,
                  do_extract: bool = True) -> dict:
    p = db.q1("SELECT * FROM papers WHERE id=?", (paper_id,))
    if not p:
        return {"error": "论文不存在"}
    db.ex("UPDATE papers SET status='running', error=NULL WHERE id=?", (paper_id,))
    if task_id:
        db.task_set(task_id, state="running", progress=0.05, message="解析 PDF")

    try:
        n_chunks = int(p["n_chunks"] or 0)
        if n_chunks == 0:
            if p["path"] and Path(p["path"]).exists():
                res = pdf.ingest_file(Path(p["path"]),
                                      target_chars=int(CFG.get("literature.chunk_tokens", 1000)) * 3)
                if res["needs_ocr"]:
                    db.ex("UPDATE papers SET status='needs_ocr', error=? WHERE id=?",
                          ("文本层缺失，疑似扫描件，需 OCR", paper_id))
                    if task_id:
                        db.task_set(task_id, state="failed", progress=1.0,
                                    message="需 OCR，已跳过")
                    return {"paper_id": paper_id, "status": "needs_ocr"}
                blocks = res["chunks"]
                if not p["title"] and res["title_guess"]:
                    db.ex("UPDATE papers SET title=? WHERE id=?", (res["title_guess"], paper_id))
            elif p["abstract"]:
                blocks = [{"section": "abstract", "page": 1, "text": p["abstract"], "idx": 0}]
            else:
                db.ex("UPDATE papers SET status='failed', error='无 PDF 也无摘要' WHERE id=?",
                      (paper_id,))
                if task_id:
                    db.task_set(task_id, state="failed", message="无可用文本")
                return {"paper_id": paper_id, "status": "failed"}

            db.ex("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
            for c in blocks:
                db.ex("INSERT INTO chunks(paper_id,idx,section,page,text) VALUES(?,?,?,?,?)",
                      (paper_id, c["idx"], c["section"], c["page"], c["text"]))
            n_chunks = len(blocks)
            db.ex("UPDATE papers SET n_chunks=? WHERE id=?", (n_chunks, paper_id))

        if task_id and _paused(task_id):
            db.ex("UPDATE papers SET status='paused' WHERE id=?", (paper_id,))
            return {"paper_id": paper_id, "status": "paused"}
        if task_id:
            db.task_set(task_id, progress=0.35, message=f"向量化 {n_chunks} 块")

        rows = db.q("SELECT c.id, c.text FROM chunks c LEFT JOIN embeddings e "
                    "ON e.chunk_id=c.id WHERE c.paper_id=? AND e.chunk_id IS NULL",
                    (paper_id,))
        if rows:
            llm = LLM("bowen")
            ids = [r["id"] for r in rows]
            vecs = llm.embed([r["text"] for r in rows])
            index.store_embeddings(ids, np.asarray(vecs))

        if task_id and _paused(task_id):
            db.ex("UPDATE papers SET status='paused' WHERE id=?", (paper_id,))
            return {"paper_id": paper_id, "status": "paused"}
        if task_id:
            db.task_set(task_id, progress=0.6,
                        message="V3.2 证据预筛 → V4-Pro 机理/图谱审校")

        card = {}
        if do_extract:
            # 先生成并校验新卡片，成功后再替换旧证据；模型失败时保留上一版学习结果。
            def report_extract(stage: str, progress: float) -> None:
                if task_id:
                    db.task_set(task_id, progress=progress, message=stage)

            card = extract.extract_paper(paper_id, persist_result=False,
                                         progress=report_extract)
            if card.get("error"):
                raise RuntimeError(card["error"])
            db.ex("DELETE FROM claims WHERE paper_id=?", (paper_id,))
            db.ex("DELETE FROM kg_edges WHERE paper_id=?", (paper_id,))
            extract.persist(paper_id, card)

        db.ex("UPDATE papers SET status='done', done_at=strftime('%s','now') WHERE id=?",
              (paper_id,))
        if task_id:
            db.task_set(task_id, state="done", progress=1.0,
                        message=f"{n_chunks} 块 / {len(card.get('mechanism_claims', []) or [])} 条主张")
        index.invalidate()
        return {"paper_id": paper_id, "status": "done", "n_chunks": n_chunks,
                "n_claims": len(card.get("mechanism_claims", []) or [])}

    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        db.ex("UPDATE papers SET status='failed', error=? WHERE id=?", (err[:900], paper_id))
        if task_id:
            db.task_set(task_id, state="failed", message=err[:250])
        CFG.logs_dir.joinpath("worker_errors.log").open("a", encoding="utf-8").write(
            f"\n=== {paper_id} ===\n{traceback.format_exc()}\n")
        return {"paper_id": paper_id, "status": "failed", "error": err}


# ---------------- 线程 ----------------
def _claim_next() -> tuple[str, int] | None:
    """原子地领取一个待处理任务，避免多线程重复处理同一篇。"""
    with _claim_lock:
        row = db.q1("SELECT * FROM tasks WHERE kind='ingest_pdf' AND state='queued' "
                    "ORDER BY pinned DESC, id ASC LIMIT 1")
        if not row:
            return None
        db.task_set(int(row["id"]), state="running", message="已领取")
        return row["ref"], int(row["id"])


def _loop() -> None:
    while True:
        state = db.kv_get(CTL, "paused")
        if state == "stopped":
            break
        if state != "running":
            time.sleep(1.0)
            continue
        claimed = _claim_next()
        if not claimed:
            time.sleep(2.0)
            continue
        process_paper(claimed[0], claimed[1])
        time.sleep(0.2)


def ensure_thread() -> None:
    """按配置维持 N 个摄取线程。"""
    n = int(CFG.get("literature.n_workers", 4))
    with _lock:
        alive = [t for t in _threads if t.is_alive()]
        _threads.clear()
        _threads.extend(alive)
        for i in range(len(alive), n):
            t = threading.Thread(target=_loop, daemon=True, name=f"lit-worker-{i}")
            t.start()
            _threads.append(t)


def control(action: str) -> dict:
    if remote.enabled():
        return remote.call("module", "worker.control", action)
    """start | pause | resume | stop"""
    if action in ("start", "resume"):
        db.kv_set(CTL, "running")
        ensure_thread()
    elif action == "pause":
        db.kv_set(CTL, "paused")
    elif action == "stop":
        db.kv_set(CTL, "stopped")
    return status()


def _activate_learning_task(task_id: int, message: str, auto_start: bool) -> dict:
    """Queue one paper and, when requested, wake the shared worker pool."""
    db.task_set(task_id, state="queued", progress=0.0, message=message)
    if auto_start:
        db.kv_set(CTL, "running")
        ensure_thread()
    current = db.q1("SELECT state,progress,message FROM tasks WHERE id=?", (task_id,))
    return {"task_id": task_id,
            "state": current["state"] if current else "queued",
            "progress": float(current["progress"] or 0) if current else 0.0,
            "message": current["message"] if current else message,
            "worker_started": bool(auto_start)}


def status() -> dict:
    if remote.enabled():
        return remote.call("module", "worker.status")
    # Papers are the canonical learning state.  Task rows may include deleted
    # history, which used to inflate the denominator (for example 139/234 while
    # the library actually contained 233 papers).
    counts = {r["status"]: r["c"] for r in db.q(
        "SELECT status, COUNT(*) c FROM papers GROUP BY status")}
    task_counts = {r["state"]: r["c"] for r in db.q(
        "SELECT state, COUNT(*) c FROM tasks WHERE kind='ingest_pdf' "
        "AND state!='deleted' GROUP BY state")}
    pinned_done = db.q1("SELECT COUNT(*) c FROM papers WHERE pinned=1 AND status='done'")["c"]
    pinned_total = db.q1("SELECT COUNT(*) c FROM papers WHERE pinned=1")["c"]
    running = db.q1(
        "SELECT t.ref,COALESCE(NULLIF(p.title,''),t.label) label,t.progress,t.message "
        "FROM tasks t LEFT JOIN papers p ON p.id=t.ref "
        "WHERE t.kind='ingest_pdf' AND t.state='running' LIMIT 1")
    running_rows = db.rows_to_dicts(db.q(
        "SELECT t.ref,COALESCE(NULLIF(p.title,''),t.label) label,t.progress,t.message "
        "FROM tasks t LEFT JOIN papers p ON p.id=t.ref "
        "WHERE t.kind='ingest_pdf' AND t.state='running' ORDER BY t.updated_at"))
    known = {r["ref"] for r in running_rows}
    for p in db.q("SELECT id,title FROM papers WHERE status='running'"):
        if p["id"] not in known:
            running_rows.append({"ref": p["id"], "label": p["title"] or p["id"],
                                 "progress": 0.05, "message": "立即处理"})
    worker_state = db.kv_get(CTL, "paused")
    threads_alive = sum(1 for t in _threads if t.is_alive())
    activity = ("processing" if running_rows else
                "idle" if worker_state == "running" and threads_alive else
                worker_state)
    latest = db.q1(
        "SELECT id,title,done_at FROM papers WHERE status='done' "
        "ORDER BY done_at DESC LIMIT 1")
    latest_added = db.q1("SELECT MAX(added_at) ts FROM papers")
    category_counts = {key: 0 for key in FAILURE_LABELS}
    for row in db.q("SELECT error FROM papers WHERE status='failed'"):
        key = failure_category(row["error"] or "")
        category_counts[key] = category_counts.get(key, 0) + 1
    failure_categories = [
        {"key": key, "label": FAILURE_LABELS[key], "count": category_counts.get(key, 0)}
        for key in FAILURE_LABELS if category_counts.get(key, 0)
    ]
    return {"worker": worker_state,
            "activity": activity,
            "active_papers": len(running_rows),
            "idle_threads": max(0, threads_alive - len(running_rows)),
            "threads_alive": threads_alive,
            "n_workers_configured": int(CFG.get("literature.n_workers", 4)),
            "queue": counts,
            "task_queue": task_counts,
            "running_now": running_rows,
            "total_papers": sum(int(v) for v in counts.values()),
            "latest_added_at": float(latest_added["ts"] or 0) if latest_added else 0.0,
            "latest_done_at": float(latest["done_at"] or 0) if latest else 0.0,
            "last_completed": ({"paper_id": latest["id"], "title": latest["title"] or latest["id"]}
                               if latest else None),
            "failure_categories": failure_categories,
            "core_corpus_progress": f"{pinned_done}/{pinned_total}",
            "core_corpus_done": pinned_done >= pinned_total > 0,
            "current": dict(running) if running else None}


def attach_fulltext(paper_id: str, src_path: str, reingest: bool = True) -> dict:
    """把一份手动提供的 PDF 绑到已有文献记录上（摘要级 -> 全文级）。

    联网检索经常拿不到闭源全文，只能存摘要；有些 Crossref 记录连摘要都没有，
    直接判 failed。这个函数让你补上 PDF 后原地升级，而不是新建一条重复记录。
    """
    import shutil
    p = db.q1("SELECT * FROM papers WHERE id=?", (paper_id,))
    if not p:
        return {"error": f"文献 {paper_id} 不存在"}
    src = Path(src_path)
    if not src.exists():
        return {"error": f"文件不存在: {src_path}"}
    if src.read_bytes()[:5] != b"%PDF-":
        return {"error": "不是有效的 PDF 文件"}
    dedup.refresh_identities()
    duplicate = dedup.find_duplicate(path=src, exclude_id=paper_id)
    if duplicate:
        return {"error": "这份 PDF 已绑定到另一条文献记录，请先合并重复项",
                "existing_id": duplicate["id"], "existing_title": duplicate["title"],
                "matched_by": duplicate["matched_by"]}

    dest = CFG.new_pdf_dir / f"{paper_id}.pdf"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copyfile(src, dest)

    # 清掉摘要级残留，强制重抽
    db.ex("DELETE FROM embeddings WHERE chunk_id IN "
          "(SELECT id FROM chunks WHERE paper_id=?)", (paper_id,))
    db.ex("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
    db.ex("DELETE FROM claims WHERE paper_id=?", (paper_id,))
    db.ex("DELETE FROM kg_edges WHERE paper_id=?", (paper_id,))
    db.ex("UPDATE papers SET path=?, evidence_level='fulltext', status='queued', "
          "n_chunks=0, error=NULL,content_hash=? WHERE id=?",
          (str(dest), dedup.pdf_hash(dest), paper_id))
    tid = int(db.task_add("ingest_pdf", paper_id, (p["title"] or paper_id)[:120]))
    index.invalidate()

    out = {"paper_id": paper_id, "title": (p["title"] or "")[:120],
           "saved_to": str(dest), "size_kb": round(dest.stat().st_size / 1024, 1),
           "previous_level": p["evidence_level"], "now": "fulltext"}
    out["learning"] = _activate_learning_task(
        tid, "已补全文，进入学习队列", auto_start=reingest)
    return out


def register_new_pdf(src_path: str, title: str = "", doi: str = "",
                     reingest: bool = True) -> dict:
    """上传一篇全新的 PDF（不绑定已有记录）。按文件名哈希去重。"""
    import shutil
    src = Path(src_path)
    if not src.exists():
        return {"error": f"文件不存在: {src_path}"}
    if src.read_bytes()[:5] != b"%PDF-":
        return {"error": "不是有效的 PDF 文件"}
    dedup.refresh_identities()
    duplicate = dedup.find_duplicate(doi=doi, title=title or src.stem, path=src)
    if duplicate:
        return {"error": "该论文已存在，请改用 attach_fulltext 绑定或合并重复项",
                "existing_id": duplicate["id"], "existing_title": duplicate["title"],
                "matched_by": duplicate["matched_by"]}

    pid = local_paper_id(src)
    dest = CFG.new_pdf_dir / f"{pid}.pdf"
    if db.q1("SELECT id FROM papers WHERE id=?", (pid,)):
        return {"error": "同名文件已入库", "paper_id": pid}
    if src.resolve() != dest.resolve():
        shutil.copyfile(src, dest)
    keys = dedup.identity(doi, title or src.stem, dest)
    db.ex("INSERT INTO papers(id,source,doi,title,path,pinned,status,evidence_level,"
          "added_at,doi_key,title_key,content_hash) "
          "VALUES(?,'upload',?,?,?,0,'queued','fulltext',strftime('%s','now'),?,?,?)",
          (pid, doi, title or src.stem[:250], str(dest), keys["doi_key"],
           keys["title_key"], keys["content_hash"]))
    tid = int(db.task_add("ingest_pdf", pid, (title or src.stem)[:120]))
    out = {"paper_id": pid, "saved_to": str(dest),
           "size_kb": round(dest.stat().st_size / 1024, 1)}
    out["learning"] = _activate_learning_task(
        tid, "上传完成，进入学习队列", auto_start=reingest)
    return out


def needs_fulltext() -> list[dict]:
    if remote.enabled():
        return remote.call("module", "worker.needs_fulltext")
    """列出所有拿不到全文的文献 —— 这些就是等你手动上传 PDF 的对象。"""
    rows = db.q(
        "SELECT id,title,doi,year,journal,status,evidence_level,error,source "
        "FROM papers WHERE evidence_level='abstract' OR status IN ('failed','needs_ocr') "
        "ORDER BY (status='failed') DESC, added_at")
    out = []
    for r in rows:
        out.append({"paper_id": r["id"], "title": (r["title"] or "")[:130],
                    "doi": r["doi"], "year": r["year"],
                    "journal": (r["journal"] or "")[:60],
                    "status": r["status"], "evidence_level": r["evidence_level"],
                    "source": r["source"], "why": r["error"] or "只有摘要，无全文",
                    "doi_url": f"https://doi.org/{r['doi']}" if r["doi"] else None})
    return out
