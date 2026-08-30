"""博闻 BOWEN —— 文献层工具集。"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

from ..core import db
from ..core.config import CFG
from ..core.llm import LLM
from ..core.tools import (P, obj, report_tool_progress, tool,
                          tool_cancel_requested)
from ..lit import dedup, extract, index, kg, search, worker


LIT_TOOLS_API_VERSION = 3
_EXPAND_EXECUTOR = ThreadPoolExecutor(max_workers=1,
                                      thread_name_prefix="zhizhi-lit-expand")
_EXPAND_LOCK = threading.RLock()
_EXPAND_RECOVERED = False
_EXPAND_META_PREFIX = "lit_expand_meta:"
_SCHEDULE_META_PREFIX = "lit_schedule_meta:"
_SCHEDULE_LOCK = threading.RLock()
_SCHEDULE_WAKE = threading.Event()
_SCHEDULE_THREAD: threading.Thread | None = None
_SCHEDULE_RUNNING: set[str] = set()


class _ScheduleInterrupted(RuntimeError):
    """A scheduled round was paused, resumed, or deleted at a safe boundary."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _meta_patch(key: str, **changes: Any) -> dict:
    """Atomically merge small persistent task metadata stored in SQLite KV."""
    with _EXPAND_LOCK:
        meta = db.kv_get(key, {})
        if not isinstance(meta, dict):
            meta = {}
        meta.update(changes)
        meta["updated_at"] = time.time()
        db.kv_set(key, meta)
        return meta


def _schedule_key(ref: str) -> str:
    return f"{_SCHEDULE_META_PREFIX}{ref}"


def _expansion_key(ref: str) -> str:
    return f"{_EXPAND_META_PREFIX}{ref}"


def _recover_expansion_tasks() -> None:
    """A process restart invalidates the old expansion thread."""
    global _EXPAND_RECOVERED
    with _EXPAND_LOCK:
        if _EXPAND_RECOVERED:
            return
        interrupted = db.q(
            "SELECT ref FROM tasks WHERE kind='lit_expand' "
            "AND state IN ('queued','running')")
        db.ex("UPDATE tasks SET state='failed',progress=1.0,"
              "message='应用重启，后台文献扩充已中断',updated_at=? "
              "WHERE kind='lit_expand' AND state IN ('queued','running')",
              (time.time(),))
        for row in interrupted:
            _meta_patch(_expansion_key(row["ref"]), state="failed",
                        error="应用重启，后台文献扩充已中断")
        _EXPAND_RECOVERED = True


# ================= 语料与队列 =================
@tool("lit_status",
      "文献库与摄取队列总览：核心 59 篇进度、各状态计数、当前正在处理的文献、图谱规模。",
      obj({}), category="lit")
def lit_status() -> dict:
    st = worker.status()
    papers = {r["status"]: r["c"] for r in
              db.q("SELECT status, COUNT(*) c FROM papers GROUP BY status")}
    lvl = {r["evidence_level"]: r["c"] for r in
           db.q("SELECT evidence_level, COUNT(*) c FROM papers GROUP BY evidence_level")}
    return {"worker": st, "papers_by_status": papers, "papers_by_evidence_level": lvl,
            "n_papers": db.q1("SELECT COUNT(*) c FROM papers")["c"],
            "n_chunks": db.q1("SELECT COUNT(*) c FROM chunks")["c"],
            "n_claims": db.q1("SELECT COUNT(*) c FROM claims")["c"],
            "n_contradictions_open": db.q1(
                "SELECT COUNT(*) c FROM contradictions WHERE status='open'")["c"],
            "kg": kg.stats()}


@tool("lit_bootstrap",
      "注册核心语料：扫描 reference-59 目录与 store/pdf_new 手动投放目录，登记为待处理。"
      "核心 59 篇标记为 pinned，只能暂停不能删除。",
      obj({}), category="lit")
def lit_bootstrap() -> dict:
    return {"core": worker.bootstrap_core_corpus(), "manual": worker.scan_new_pdfs()}


@tool("lit_deduplicate",
      "扫描论文库并按规范化 DOI、规范化题名、PDF 内容哈希合并重复记录；"
      "同时合并 NF270/NF 270/NF-270 一类实体别名。不会删除磁盘 PDF。",
      obj({"dry_run": P("boolean", "true 只报告，false 执行合并，默认 true")}),
      category="lit")
def lit_deduplicate(dry_run: bool = True) -> dict:
    result = dedup.deduplicate_library(bool(dry_run))
    index.invalidate()
    return result


@tool("lit_control",
      "控制摄取队列：start 开始 / pause 暂停 / resume 继续 / stop 停止。",
      obj({"action": P("string", "动作", enum=["start", "pause", "resume", "stop"])},
          ["action"]), category="lit")
def lit_control(action: str) -> dict:
    return worker.control(action)


@tool("lit_task_control",
      "对单个摄取任务操作：pause 暂停 / resume 重新排队 / delete 删除（核心 59 篇拒绝删除）"
      " / retry 重试失败任务。",
      obj({"paper_id": P("string", "论文 id"),
           "action": P("string", "动作", enum=["pause", "resume", "delete", "retry"])},
          ["paper_id", "action"]), category="lit")
def lit_task_control(paper_id: str, action: str) -> dict:
    p = db.q1("SELECT id,pinned,title FROM papers WHERE id=?", (paper_id,))
    if not p:
        return {"error": f"论文 {paper_id} 不存在"}
    t = db.q1("SELECT id FROM tasks WHERE kind='ingest_pdf' AND ref=?", (paper_id,))
    if action == "delete":
        if p["pinned"]:
            return {"error": "这是核心 59 篇之一，不允许删除。可以 pause 暂缓处理。",
                    "title": p["title"]}
        db.ex("DELETE FROM embeddings WHERE chunk_id IN "
              "(SELECT id FROM chunks WHERE paper_id=?)", (paper_id,))
        db.ex("DELETE FROM chunks WHERE paper_id=?", (paper_id,))
        db.ex("DELETE FROM claims WHERE paper_id=?", (paper_id,))
        db.ex("DELETE FROM kg_edges WHERE paper_id=?", (paper_id,))
        db.ex("DELETE FROM papers WHERE id=?", (paper_id,))
        if t:
            db.task_set(int(t["id"]), state="deleted")
        index.invalidate()
        return {"deleted": paper_id}
    mapping = {"pause": "paused", "resume": "queued", "retry": "queued"}
    if t:
        db.task_set(int(t["id"]), state=mapping[action], message=f"手动 {action}")
    db.ex("UPDATE papers SET status=? WHERE id=?",
          ("paused" if action == "pause" else "queued", paper_id))
    return {"paper_id": paper_id, "action": action, "queue": worker.status()}


@tool("lit_list_papers",
      "列出文献（可按状态/来源/是否核心筛选），返回 id、题名、年份、状态、证据级别。",
      obj({"status": P("string", "queued|running|done|failed|paused|needs_ocr，留空为全部"),
           "pinned_only": P("boolean", "只看核心 59 篇"),
           "limit": P("integer", "返回条数，默认 40")}), category="lit")
def lit_list_papers(status: str = "", pinned_only: bool = False, limit: int = 40) -> dict:
    where, args = [], []
    if status:
        where.append("status=?")
        args.append(status)
    if pinned_only:
        where.append("pinned=1")
    sql = "SELECT id,title,year,journal,status,evidence_level,pinned,n_chunks,error FROM papers"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY pinned DESC, added_at ASC LIMIT ?"
    args.append(limit)
    rows = db.rows_to_dicts(db.q(sql, args))
    for r in rows:
        r["title"] = (r["title"] or "")[:140]
    return {"n": len(rows), "papers": rows}


@tool("lit_process_now",
      "立刻处理指定文献（不经队列），用于插队或调试。",
      obj({"paper_id": P("string", "论文 id")}, ["paper_id"]), category="lit")
def lit_process_now(paper_id: str) -> dict:
    return worker.process_paper(paper_id)


# ================= 检索与问答 =================
@tool("lit_search",
      "在已入库文献里做混合检索（BM25 + 向量），返回带页码和原文的段落。"
      "回答任何文献问题前先用它取证。",
      obj({"query": P("string", "检索问题或关键词"),
           "top_k": P("integer", "返回段落数，默认 8"),
           "paper_ids": P("array", "限定在这些论文内检索", items={"type": "string"})},
          ["query"]), category="lit")
def lit_search(query: str, top_k: int = 8, paper_ids: list[str] | None = None) -> dict:
    hits = index.hybrid_search(query, top_k, paper_ids)
    return {"query": query, "n": len(hits),
            "passages": [{"paper_id": h["paper_id"], "title": (h["title"] or "")[:120],
                          "year": h["year"], "doi": h["doi"], "section": h["section"],
                          "page": h["page"], "score": h["score"],
                          "text": h["text"][:1800]} for h in hits],
            "citation_rule": "引用时必须写成 [题名, 年份, p.页码]，并逐字保留英文原句。"}


@tool("lit_paper_card",
      "取某篇文献的结构化卡片（膜、化合物、条件、机理主张、局限、反常现象）。",
      obj({"paper_id": P("string", "论文 id")}, ["paper_id"]), category="lit")
def lit_paper_card(paper_id: str) -> dict:
    r = db.q1("SELECT id,title,year,journal,doi,status,evidence_level,meta FROM papers "
              "WHERE id=?", (paper_id,))
    if not r:
        return {"error": "不存在"}
    card = db.jdict(r["meta"])
    claims = db.rows_to_dicts(db.q(
        "SELECT descriptor,direction,membrane,scope,statement,quote,page,confidence "
        "FROM claims WHERE paper_id=?", (paper_id,)))
    return {"paper": {k: r[k] for k in
                      ("id", "title", "year", "journal", "doi", "status", "evidence_level")},
            "card": {k: card.get(k) for k in
                     ("membranes", "compounds", "conditions", "key_findings",
                      "limitations", "anomalies")},
            "claims": claims}


@tool("lit_claims",
      "按 descriptor 检索机理主张，看不同文献在同一因素上的方向是否一致。",
      obj({"descriptor": P("string", "因素关键词，如 log Kow / 分子尺寸 / 电荷 / pH"),
           "limit": P("integer", "默认 30")}), category="lit")
def lit_claims(descriptor: str = "", limit: int = 30) -> dict:
    if descriptor:
        rows = db.q("SELECT c.*, p.title, p.year FROM claims c JOIN papers p ON p.id=c.paper_id "
                    "WHERE lower(c.descriptor) LIKE ? LIMIT ?",
                    (f"%{descriptor.lower()}%", limit))
    else:
        rows = db.q("SELECT c.*, p.title, p.year FROM claims c JOIN papers p ON p.id=c.paper_id "
                    "ORDER BY c.id DESC LIMIT ?", (limit,))
    out = []
    for r in rows:
        out.append({"descriptor": r["descriptor"], "direction": r["direction"],
                    "membrane": r["membrane"], "scope": r["scope"],
                    "statement": r["statement"], "quote": (r["quote"] or "")[:400],
                    "source": f"{(r['title'] or '')[:80]} ({r['year']}) p.{r['page']}"})
    dirs: dict[str, int] = {}
    for o in out:
        dirs[o["direction"]] = dirs.get(o["direction"], 0) + 1
    return {"n": len(out), "direction_counts": dirs, "claims": out}


@tool("lit_contradictions",
      "文献矛盾清单。detect=true 会重新扫描 claims 并用 LLM 研判新的矛盾。"
      "每条矛盾都带调和假设与可证伪预测 —— 这是发现层引擎 3 的燃料。",
      obj({"detect": P("boolean", "是否重新探测，默认 false"),
           "status": P("string", "open|explained|dismissed，留空为全部")}), category="lit")
def lit_contradictions(detect: bool = False, status: str = "") -> dict:
    made = extract.detect_contradictions() if detect else []
    sql = "SELECT * FROM contradictions"
    args: list[Any] = []
    if status:
        sql += " WHERE status=?"
        args.append(status)
    sql += " ORDER BY id DESC LIMIT 40"
    rows = db.rows_to_dicts(db.q(sql, args))
    for r in rows:
        try:
            r["note"] = db.jdict(r["note"])
        except Exception:  # noqa: BLE001
            pass
    return {"newly_detected": len(made), "n": len(rows), "contradictions": rows,
            "read": ("status='open' = 判定为真矛盾，尚无调和解释；"
                     "status='explained' = 提出了调和变量。**两类都要看** —— "
                     "'explained' 里的 reconciliation_hypothesis 和 falsifiable_prediction "
                     "本身就是尚未被验证的新命题，是引擎3 最直接的素材。"),
            "note_field": "每条的 note 字段是 JSON，含 candidate_reconciling_variable / "
                          "reconciliation_hypothesis / falsifiable_prediction。"}


# ================= 文献扩充 =================
def _novelty_scores(works: list[dict]) -> list[float]:
    """每篇候选相对已有语料的新颖度 = 1 - 与语料库最大余弦相似度。"""
    ids, mat = index._load_vectors()
    if len(ids) == 0:
        return [1.0] * len(works)
    texts = [f"{w.get('title','')} {(w.get('abstract') or '')[:1200]}" for w in works]
    try:
        qv = LLM("bowen").embed(texts)
    except Exception:  # noqa: BLE001
        return [1.0] * len(works)
    sims = qv @ mat.T
    return [float(1.0 - s.max()) for s in sims]


@tool("lit_expand_search",
      "★ 文献扩充（路径 A：你给主题/关键词）。LLM 自动扩展中英检索式 + 反例检索式 + "
      "跨领域检索式，多源检索去重打分，然后按【边际收益停机准则】分批收录："
      "每批 10 篇算一次新概念产出率，连续两批低于阈值即停。默认转为独立后台任务，"
      "立即返回任务编号，不阻塞当前 Agent。",
      obj({"topic": P("string", "研究主题或问题"),
           "max_papers": P("integer", "上限，默认取配置 100"),
           "min_relevance": P("number", "最低相关性分，默认 6"),
           "fetch_fulltext": P("boolean", "是否尝试下载 OA 全文，默认 true"),
           "auto_ingest": P("boolean", "收录后是否自动开始摄取，默认 true"),
           "background": P("boolean", "是否后台运行，默认 true；仅维护调试时设 false"),
           "year_min": P("integer", "最早年份")},
          ["topic"]), category="lit", long_running=True)
def lit_expand_search(topic: str, max_papers: int | None = None,
                      min_relevance: float = 6.0, fetch_fulltext: bool = True,
                      auto_ingest: bool = True, year_min: int | None = None,
                      background: bool = True) -> dict:
    if background:
        return queue_literature_expansion(
            topic, max_papers=max_papers, min_relevance=min_relevance,
            fetch_fulltext=fetch_fulltext, auto_ingest=auto_ingest,
            year_min=year_min)
    return _lit_expand_search_sync(
        topic, max_papers=max_papers, min_relevance=min_relevance,
        fetch_fulltext=fetch_fulltext, auto_ingest=auto_ingest,
        year_min=year_min)


def _lit_expand_search_sync(
        topic: str, max_papers: int | None = None,
        min_relevance: float = 6.0, fetch_fulltext: bool = True,
        auto_ingest: bool = True, year_min: int | None = None,
        progress: Callable[[float, str], None] | None = None) -> dict:
    """Blocking implementation used only by the dedicated expansion worker."""
    update = progress or (lambda _fraction, _message: None)
    cfg = CFG.get("literature.expansion")
    max_papers = int(max_papers or cfg["max_papers"])
    update(0.08, "V3.2 正在整理检索式")
    plan = search.expand_queries(topic, stance="both_sides")
    queries = (plan.get("en") or [])[:8] + (plan.get("negative_evidence") or [])[:3] \
        + (plan.get("cross_domain") or [])[:3]
    if not queries:
        queries = [topic]
    update(0.20, f"正在从外部数据库检索 {len(queries)} 条检索式")
    works = search.search_many(queries, per_source=15,
                               year_min=year_min or plan.get("year_min"))
    # Remove library and in-result duplicates before the paid V4 relevance
    # screen.  Repeated scheduled runs therefore neither relearn nor rescore
    # the same paper.
    dedup.refresh_identities()
    new_works: list[dict] = []
    existing_ids: list[str] = []
    seen_candidates: set[str] = set()
    repeated_candidates = 0
    for work in works:
        doi_key = dedup.normalize_doi(work.get("doi", ""))
        title_key = dedup.normalize_title(work.get("title", ""))
        candidate_key = (f"doi:{doi_key}" if doi_key else
                         f"title:{title_key}" if title_key else
                         f"work:{search.work_id('', str(work))}")
        if candidate_key in seen_candidates:
            repeated_candidates += 1
            continue
        seen_candidates.add(candidate_key)
        duplicate = dedup.find_duplicate(work.get("doi", ""), work.get("title", ""))
        if duplicate:
            existing_ids.append(str(duplicate["id"]))
            continue
        new_works.append(work)

    update(0.42, f"去重后 V4-Pro 正在初筛 {len(new_works)} 篇新候选文献")
    scored = search.score_relevance(new_works, topic)
    above_threshold = [w for w in scored if w.get("relevance", 0) >= min_relevance]
    keep = above_threshold

    update(0.62, f"去重后剩余 {len(keep)} 篇，正在计算相对现有语料的新概念增益")
    batch, sat_yield = int(cfg["batch_size"]), float(cfg["saturation_yield"])
    need_sat, min_papers = int(cfg["saturation_batches"]), int(cfg["min_papers"])
    accepted: list[dict] = []
    log: list[dict] = []
    low_streak = 0
    for s in range(0, len(keep), batch):
        chunk = keep[s:s + batch]
        nov = _novelty_scores(chunk)
        yld = float(np.mean([n > 0.20 for n in nov])) if nov else 0.0
        accepted.extend(chunk)
        log.append({"batch": len(log) + 1, "n": len(chunk),
                    "new_concept_yield": round(yld, 3),
                    "mean_novelty": round(float(np.mean(nov)), 3) if nov else None})
        if len(accepted) >= max_papers:
            log[-1]["stop"] = "达到上限"
            accepted = accepted[:max_papers]
            break
        if len(accepted) >= min_papers:
            low_streak = low_streak + 1 if yld < sat_yield else 0
            if low_streak >= need_sat:
                log[-1]["stop"] = f"连续 {need_sat} 批新概念产出率 < {sat_yield}，判定饱和"
                break

    update(0.74, f"正在下载可用全文并将 {len(accepted)} 篇加入摄取队列")
    res = search.enqueue(accepted, fetch_fulltext=fetch_fulltext)
    if auto_ingest:
        worker.control("start")
    update(0.96, "文献已入摄取队列，后续学习由文献 worker 完成")
    return {"topic": topic, "query_plan": plan, "n_queries": len(queries),
            "n_found": len(works), "n_above_threshold": len(above_threshold),
            "n_new_candidates": len(new_works),
            "n_preexisting": len(existing_ids),
            "n_repeated_candidates": repeated_candidates,
            "preexisting_ids": list(dict.fromkeys(existing_ids)),
            "n_accepted": len(accepted), "batches": log, "enqueue": res,
            "accepted_preview": [{"title": w["title"][:110], "year": w.get("year"),
                                  "relevance": w.get("relevance"),
                                  "evidence_type": w.get("evidence_type"),
                                  "doi": w.get("doi")} for w in accepted[:15]],
            "stopping_rule": ("不是固定篇数，而是边际收益停机：新概念产出率连续两批低于阈值即停。"
                              f"本次停在 {len(accepted)} 篇。")}


def queue_literature_expansion(
        topic: str, max_papers: int | None = None,
        min_relevance: float = 6.0, fetch_fulltext: bool = True,
        auto_ingest: bool = True, year_min: int | None = None) -> dict:
    """Queue full literature expansion without holding an Agent tool call."""
    _recover_expansion_tasks()
    topic = str(topic or "").strip()
    if not topic:
        return {"error": "主题不能为空"}
    target = int(max_papers or CFG.get("literature.expansion.max_papers", 40))
    ref = f"LX{time.strftime('%m%d')}-{uuid.uuid4().hex[:8]}"
    task_id = db.task_add("lit_expand", ref, f"文献扩充：{topic[:100]}")
    _meta_patch(
        _expansion_key(ref), ref=ref, task_id=task_id, topic=topic,
        target=target, state="queued", progress=0.0, message="等待后台扩充线程",
        created_at=time.time(), paper_ids=[], duplicate_ids=[])
    _EXPAND_EXECUTOR.submit(
        _run_literature_expansion, task_id, ref, topic, target,
        min_relevance, fetch_fulltext, auto_ingest, year_min)
    return {
        "queued": True,
        "task_id": task_id,
        "task_ref": ref,
        "topic": topic,
        "state": "queued",
        "read": "扩充已转入后台，不阻塞当前 Agent；可在任务监视器查看阶段进度。",
    }


def _run_literature_expansion(
        task_id: int, ref: str, topic: str, max_papers: int | None,
        min_relevance: float, fetch_fulltext: bool, auto_ingest: bool,
        year_min: int | None) -> None:
    def update(fraction: float, message: str) -> None:
        db.task_set(task_id, state="running", progress=max(0.01, min(0.99, fraction)),
                    message=message[:250])
        _meta_patch(_expansion_key(ref), state="running", progress=fraction,
                    message=message)

    try:
        update(0.02, "后台文献扩充已启动")
        result = _lit_expand_search_sync(
            topic, max_papers=max_papers, min_relevance=min_relevance,
            fetch_fulltext=fetch_fulltext, auto_ingest=auto_ingest,
            year_min=year_min, progress=update)
        path = CFG.logs_dir / f"lit_expand_{ref}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        enq = result.get("enqueue") or {}
        paper_ids = list(dict.fromkeys(enq.get("added_ids") or []))
        duplicate_ids = list(dict.fromkeys(
            (result.get("preexisting_ids") or []) + (enq.get("duplicate_ids") or [])))
        _meta_patch(
            _expansion_key(ref), state="done", progress=1.0,
            message="检索与入队完成，正在逐篇学习", completed_at=time.time(),
            result_path=str(path), paper_ids=paper_ids,
            duplicate_ids=duplicate_ids, n_found=result.get("n_found", 0),
            n_above_threshold=result.get("n_above_threshold", 0),
            n_accepted=result.get("n_accepted", 0), n_added=enq.get("added", 0),
            n_fulltext=enq.get("fulltext_obtained", 0),
            n_abstract=enq.get("abstract_only", 0),
            n_skipped_duplicate=(int(result.get("n_preexisting") or 0)
                                 + int(result.get("n_repeated_candidates") or 0)
                                 + int(enq.get("skipped_duplicate") or 0)))
        db.task_set(
            task_id, state="done", progress=1.0,
            message=(f"完成：新增 {enq.get('added', 0)} 篇，"
                     f"跳过重复 {enq.get('skipped_duplicate', 0)} 篇；结果 {path}")[:250])
    except Exception as exc:  # noqa: BLE001
        _meta_patch(_expansion_key(ref), state="failed", progress=1.0,
                    error=f"{type(exc).__name__}: {exc}")
        db.task_set(task_id, state="failed", progress=1.0,
                    message=f"{type(exc).__name__}: {exc}"[:250])


def _paper_learning_summary(paper_ids: list[str]) -> dict:
    ids = list(dict.fromkeys(str(x) for x in paper_ids if x))
    if not ids:
        return {"total": 0, "tracked": 0, "done": 0, "progress": 0.0,
                "by_status": {}, "running": [], "missing": 0}
    marks = ",".join("?" for _ in ids)
    rows = db.q(
        f"SELECT p.id,p.title,p.status,p.evidence_level,p.n_chunks,p.error,"
        f"t.progress,t.message FROM papers p LEFT JOIN tasks t "
        f"ON t.kind='ingest_pdf' AND t.ref=p.id AND t.state!='deleted' "
        f"WHERE p.id IN ({marks})", ids)
    counts: dict[str, int] = {}
    running: list[dict] = []
    for row in rows:
        state = str(row["status"] or "unknown")
        counts[state] = counts.get(state, 0) + 1
        if state == "running":
            running.append({"paper_id": row["id"], "title": row["title"] or row["id"],
                            "progress": float(row["progress"] or 0),
                            "message": row["message"] or "正在学习"})
    done = counts.get("done", 0)
    return {"total": len(ids), "tracked": len(rows), "done": done,
            "progress": done / max(1, len(ids)), "by_status": counts,
            "running": running, "missing": max(0, len(ids) - len(rows))}


@tool("lit_expansion_status",
      "查看最近的文献扩充任务：检索/去重/入队进度，以及新增论文逐篇学习进度。",
      obj({"limit": P("integer", "最多返回多少个任务，默认 8")}), category="lit")
def lit_expansion_status(limit: int = 8) -> dict:
    _recover_expansion_tasks()
    rows = db.q(
        "SELECT * FROM tasks WHERE kind='lit_expand' AND state!='deleted' "
        "ORDER BY created_at DESC LIMIT ?", (max(1, min(int(limit), 50)),))
    tasks = []
    for row in rows:
        meta = db.kv_get(_expansion_key(row["ref"]), {})
        if not isinstance(meta, dict):
            meta = {}
        tracking_available = "paper_ids" in meta
        # Tasks completed before per-paper tracking was introduced still have
        # a full result log.  Recover their aggregate counts instead of
        # incorrectly presenting them as a zero-result expansion.
        if not meta and row["state"] == "done":
            legacy_path = CFG.logs_dir / f"lit_expand_{row['ref']}.json"
            try:
                legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                legacy = {}
            if isinstance(legacy, dict) and legacy:
                legacy_enq = legacy.get("enqueue") or {}
                meta = {
                    "topic": legacy.get("topic", ""),
                    "n_found": legacy.get("n_found", 0),
                    "n_above_threshold": legacy.get("n_above_threshold", 0),
                    "n_accepted": legacy.get("n_accepted", 0),
                    "n_added": legacy_enq.get("added", 0),
                    "n_fulltext": legacy_enq.get("fulltext_obtained", 0),
                    "n_abstract": legacy_enq.get("abstract_only", 0),
                    "n_skipped_duplicate": legacy_enq.get("skipped_duplicate", 0),
                    "result_path": str(legacy_path),
                }
        paper_ids = list(meta.get("paper_ids") or [])
        tasks.append({
            "task_id": int(row["id"]), "ref": row["ref"],
            "topic": meta.get("topic") or str(row["label"] or "").removeprefix("文献扩充："),
            "target": int(meta.get("target") or 0), "state": row["state"],
            "progress": float(row["progress"] or 0),
            "message": row["message"] or meta.get("message") or "",
            "created_at": float(row["created_at"] or 0),
            "completed_at": meta.get("completed_at"),
            "n_found": int(meta.get("n_found") or 0),
            "n_above_threshold": int(meta.get("n_above_threshold") or 0),
            "n_accepted": int(meta.get("n_accepted") or 0),
            "n_added": int(meta.get("n_added") or 0),
            "n_fulltext": int(meta.get("n_fulltext") or 0),
            "n_abstract": int(meta.get("n_abstract") or 0),
            "n_skipped_duplicate": int(meta.get("n_skipped_duplicate") or 0),
            "paper_ids": paper_ids, "learning": _paper_learning_summary(paper_ids),
            "tracking_available": tracking_available,
            "error": meta.get("error", ""), "result_path": meta.get("result_path", ""),
        })
    return {"tasks": tasks, "count": len(tasks)}


def _schedule_config(ref: str) -> dict | None:
    value = db.kv_get(_schedule_key(ref))
    return value if isinstance(value, dict) else None


def _schedule_task(ref: str):
    return db.q1("SELECT * FROM tasks WHERE kind='lit_schedule' AND ref=? "
                 "AND state!='deleted'", (ref,))


@tool("lit_schedule_create",
      "创建定时自主文献学习任务。每轮只筛选尚未入库的新文献，并立即加入统一摄取队列。",
      obj({"topic": P("string", "持续学习主题"),
           "interval_minutes": P("integer", "每隔多少分钟运行，默认 60"),
           "papers_per_run": P("integer", "每轮最多新增学习多少篇，默认 10")},
          ["topic"]), category="lit")
def lit_schedule_create(topic: str, interval_minutes: int = 60,
                        papers_per_run: int = 10) -> dict:
    topic = str(topic or "").strip()
    if not topic:
        return {"error": "学习主题不能为空"}
    interval = max(1, min(int(interval_minutes), 60 * 24 * 30))
    per_run = max(1, min(int(papers_per_run), 100))
    for row in db.q("SELECT ref FROM tasks WHERE kind='lit_schedule' AND state!='deleted'"):
        existing = _schedule_config(row["ref"])
        if (existing and str(existing.get("topic", "")).strip().casefold()
                == topic.casefold()):
            return {"error": "同一主题已经有定时学习任务，请直接继续或调整现有任务。",
                    "existing_ref": row["ref"]}

    ref = f"LS{time.strftime('%m%d')}-{uuid.uuid4().hex[:8]}"
    task_id = db.task_add("lit_schedule", ref, f"定时文献学习：{topic[:90]}")
    now = time.time()
    config = {
        "ref": ref, "task_id": task_id, "topic": topic,
        "interval_minutes": interval, "papers_per_run": per_run,
        "state": "running", "created_at": now, "next_run_at": now,
        "run_active": False, "runs_completed": 0, "paper_ids": [],
        "control_epoch": 0,
        "cumulative_added": 0, "cumulative_duplicates": 0,
    }
    db.kv_set(_schedule_key(ref), config)
    db.task_set(task_id, state="scheduled", progress=0.0,
                message="任务已创建，等待首次运行")
    ensure_literature_scheduler()
    _SCHEDULE_WAKE.set()
    return {"created": True, "ref": ref, "topic": topic,
            "interval_minutes": interval, "papers_per_run": per_run,
            "state": "running", "next_run_at": now}


@tool("lit_schedule_control",
      "控制定时文献学习任务：pause 暂停、resume/start 继续、delete 删除调度任务。"
      "删除调度任务不会删除已经学习的论文。",
      obj({"ref": P("string", "定时任务编号"),
           "action": P("string", "动作", enum=["start", "resume", "pause", "delete"])},
          ["ref", "action"]), category="lit")
def lit_schedule_control(ref: str, action: str) -> dict:
    action = str(action or "").strip().lower()
    task = _schedule_task(ref)
    config = _schedule_config(ref)
    if not task or not config:
        return {"error": f"定时任务 {ref} 不存在或已经删除"}
    if action == "delete":
        # Removing the config is the persistent cancellation token.  A round
        # already inside an external API call will notice this at its next
        # progress boundary and must not continue to later stages.
        db.ex("DELETE FROM kv WHERE k=?", (_schedule_key(ref),))
        db.task_set(int(task["id"]), state="deleted", progress=1.0,
                    message="定时任务已删除；当前轮已请求取消，已入库文献保留")
        _SCHEDULE_WAKE.set()
        return {"ref": ref, "deleted": True,
                "note": "不再启动新一轮；当前轮会在下一个安全阶段终止，已经进入摄取队列的文献不会被删除。"}
    if action == "pause":
        config["state"] = "paused"
        config["control_epoch"] = int(config.get("control_epoch") or 0) + 1
        config["updated_at"] = time.time()
        db.kv_set(_schedule_key(ref), config)
        message = ("暂停请求已生效；当前轮将在下一个安全阶段终止" if ref in _SCHEDULE_RUNNING
                   else "已暂停")
        db.task_set(int(task["id"]), state="paused", message=message)
        _SCHEDULE_WAKE.set()
        return {"ref": ref, "state": "paused", "message": message}
    if action in ("start", "resume"):
        config["state"] = "running"
        config["control_epoch"] = int(config.get("control_epoch") or 0) + 1
        config["next_run_at"] = time.time()
        config["updated_at"] = time.time()
        db.kv_set(_schedule_key(ref), config)
        restarting = ref in _SCHEDULE_RUNNING
        db.task_set(int(task["id"]), state="scheduled", progress=0.0,
                    message=("已继续；正在结束旧轮次后重新调度" if restarting
                             else "已继续，等待下一轮"))
        ensure_literature_scheduler()
        _SCHEDULE_WAKE.set()
        return {"ref": ref, "state": "running", "next_run_at": config["next_run_at"]}
    return {"error": f"不支持动作 {action}"}


@tool("lit_schedule_status",
      "查看全部定时自主文献学习任务、下一轮时间、累计新增与逐篇学习进度。",
      obj({}), category="lit")
def lit_schedule_status() -> dict:
    rows = db.q("SELECT * FROM tasks WHERE kind='lit_schedule' AND state!='deleted' "
                "ORDER BY created_at DESC")
    with _SCHEDULE_LOCK:
        active_refs = set(_SCHEDULE_RUNNING)
    tasks = []
    for row in rows:
        config = _schedule_config(row["ref"])
        if not config:
            continue
        paper_ids = list(config.get("paper_ids") or [])
        tasks.append({
            **config, "task_state": row["state"],
            "progress": float(row["progress"] or 0),
            "message": row["message"] or "", "currently_running": row["ref"] in active_refs,
            "learning": _paper_learning_summary(paper_ids),
        })
    return {"tasks": tasks, "count": len(tasks)}


def _dispatch_due_schedules() -> None:
    now = time.time()
    rows = db.q("SELECT id,ref FROM tasks WHERE kind='lit_schedule' AND state!='deleted'")
    for row in rows:
        ref = str(row["ref"])
        config = _schedule_config(ref)
        if (not config or config.get("state") != "running"
                or float(config.get("next_run_at") or 0) > now):
            continue
        with _SCHEDULE_LOCK:
            if ref in _SCHEDULE_RUNNING:
                continue
            _SCHEDULE_RUNNING.add(ref)
        config["run_active"] = True
        run_control_epoch = int(config.get("control_epoch") or 0)
        config["last_queued_at"] = now
        config["next_run_at"] = now + int(config["interval_minutes"]) * 60
        db.kv_set(_schedule_key(ref), config)
        db.task_set(int(row["id"]), state="queued", progress=0.01,
                    message="本轮已到期，等待文献扩充线程")
        _EXPAND_EXECUTOR.submit(
            _run_schedule_once, int(row["id"]), ref, run_control_epoch)


def _run_schedule_once(task_id: int, ref: str,
                       run_control_epoch: int | None = None) -> None:
    initial = _schedule_config(ref)
    if run_control_epoch is None:
        run_control_epoch = int((initial or {}).get("control_epoch") or 0)

    def update(fraction: float, message: str) -> None:
        current = _schedule_config(ref)
        if not current:
            raise _ScheduleInterrupted("deleted")
        if (current.get("state") != "running"
                or int(current.get("control_epoch") or 0) != run_control_epoch):
            raise _ScheduleInterrupted(
                "paused" if current.get("state") == "paused" else "superseded")
        current["last_message"] = message
        current["last_progress"] = fraction
        db.kv_set(_schedule_key(ref), current)
        db.task_set(task_id, state="running", progress=max(0.01, min(0.99, fraction)),
                    message=message[:250])

    try:
        config = _schedule_config(ref)
        if not config or config.get("state") != "running":
            return
        run_number = int(config.get("runs_completed") or 0) + 1
        update(0.02, f"第 {run_number} 轮自主文献学习已启动")
        result = _lit_expand_search_sync(
            str(config["topic"]), max_papers=int(config["papers_per_run"]),
            min_relevance=6.0, fetch_fulltext=True, auto_ingest=True,
            progress=update)
        enq = result.get("enqueue") or {}
        path = CFG.logs_dir / f"lit_schedule_{ref}_run{run_number}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        current = _schedule_config(ref)
        if not current:  # deleted while the current API call was finishing
            return
        all_ids = list(dict.fromkeys((current.get("paper_ids") or [])
                                     + (enq.get("added_ids") or [])))
        current.update({
            "paper_ids": all_ids, "runs_completed": run_number,
            "cumulative_added": int(current.get("cumulative_added") or 0)
                                + int(enq.get("added") or 0),
            "cumulative_duplicates": int(current.get("cumulative_duplicates") or 0)
                                     + int(result.get("n_preexisting") or 0)
                                     + int(result.get("n_repeated_candidates") or 0)
                                     + int(enq.get("skipped_duplicate") or 0),
            "last_run_at": time.time(), "last_error": "", "run_active": False,
            "next_run_at": time.time() + int(current["interval_minutes"]) * 60,
            "last_result": {
                "n_found": result.get("n_found", 0),
                "n_new_candidates": result.get("n_new_candidates", 0),
                "n_accepted": result.get("n_accepted", 0),
                "n_added": enq.get("added", 0),
                "n_fulltext": enq.get("fulltext_obtained", 0),
                "n_abstract": enq.get("abstract_only", 0),
                "n_duplicates": int(result.get("n_preexisting") or 0)
                                + int(result.get("n_repeated_candidates") or 0)
                                + int(enq.get("skipped_duplicate") or 0),
                "result_path": str(path),
            },
        })
        db.kv_set(_schedule_key(ref), current)
        if current.get("state") == "paused":
            db.task_set(task_id, state="paused", progress=1.0,
                        message=f"本轮新增 {enq.get('added', 0)} 篇；后续轮次已暂停")
        else:
            db.task_set(task_id, state="scheduled", progress=1.0,
                        message=(f"第 {run_number} 轮完成：新增 {enq.get('added', 0)} 篇，"
                                 f"去重跳过 {current['last_result']['n_duplicates']} 篇"))
    except _ScheduleInterrupted as interrupted:
        current = _schedule_config(ref)
        if current:  # A deleted schedule intentionally has no config to update.
            current["run_active"] = False
            current["last_message"] = (
                "本轮已在安全阶段终止" if interrupted.reason == "paused"
                else "旧轮次已终止，等待重新调度")
            db.kv_set(_schedule_key(ref), current)
            if current.get("state") == "paused":
                db.task_set(task_id, state="paused", progress=0.0,
                            message="已暂停；本轮已在安全阶段终止")
            elif current.get("state") == "running":
                db.task_set(task_id, state="scheduled", progress=0.0,
                            message="旧轮次已终止，等待重新调度")
    except Exception as exc:  # noqa: BLE001
        current = _schedule_config(ref)
        if current:
            current["last_error"] = f"{type(exc).__name__}: {exc}"
            current["last_run_at"] = time.time()
            current["run_active"] = False
            current["next_run_at"] = time.time() + int(current["interval_minutes"]) * 60
            db.kv_set(_schedule_key(ref), current)
            state = "paused" if current.get("state") == "paused" else "scheduled"
            db.task_set(task_id, state=state, progress=1.0,
                        message=f"本轮失败，下个周期重试：{type(exc).__name__}: {exc}"[:250])
    finally:
        with _SCHEDULE_LOCK:
            _SCHEDULE_RUNNING.discard(ref)
        _SCHEDULE_WAKE.set()


def _literature_scheduler_loop() -> None:
    while True:
        try:
            _dispatch_due_schedules()
        except Exception:  # noqa: BLE001 - daemon must survive one malformed task
            pass
        _SCHEDULE_WAKE.wait(timeout=10.0)
        _SCHEDULE_WAKE.clear()


def ensure_literature_scheduler() -> None:
    """Start the persistent scheduler daemon once per application process."""
    global _SCHEDULE_THREAD
    with _SCHEDULE_LOCK:
        if _SCHEDULE_THREAD and _SCHEDULE_THREAD.is_alive():
            return
        for row in db.q("SELECT id,ref FROM tasks WHERE kind='lit_schedule' "
                        "AND state!='deleted'"):
            config = _schedule_config(row["ref"])
            if not config:
                continue
            config["run_active"] = False
            db.kv_set(_schedule_key(row["ref"]), config)
            if config.get("state") == "running":
                db.task_set(int(row["id"]), state="scheduled",
                            message="应用启动后已恢复定时任务")
        _SCHEDULE_THREAD = threading.Thread(
            target=_literature_scheduler_loop, name="zhizhi-lit-scheduler", daemon=True)
        _SCHEDULE_THREAD.start()


@tool("lit_request_evidence",
      "★ 文献扩充（路径 B：发现层定向取证）。给一个假设/问题，"
      "系统会同时检索**支持证据与反对证据**（防确认偏误），"
      "先在本地语料检索，不足再联网补充，最后返回双方证据包。",
      obj({"question": P("string", "要取证的假设或问题"),
           "must_cover": P("array", "必须覆盖的要点", items={"type": "string"}),
           "n_external": P("integer", "联网最多补充几篇，默认 15，设 0 表示只查本地")},
          ["question"]), category="lit", long_running=True)
def lit_request_evidence(question: str, must_cover: list[str] | None = None,
                         n_external: int = 15) -> dict:
    rid = db.ex("INSERT INTO lit_requests(question,must_cover,stance,created_at) "
                "VALUES(?,?,'both_sides',strftime('%s','now'))",
                (question, json.dumps(must_cover or [], ensure_ascii=False))).lastrowid

    local = index.hybrid_search(question, top_k=10)
    plan = search.expand_queries(question, must_cover, "both_sides")
    external: dict[str, Any] = {"skipped": True}
    if n_external > 0:
        queries = (plan.get("en") or [])[:5] + (plan.get("negative_evidence") or [])[:3]
        works = search.search_many(queries, per_source=10, year_min=plan.get("year_min"))
        scored = search.score_relevance(works, question)
        support = [w for w in scored if w.get("evidence_type") in ("supports", "neutral")][:n_external // 2 + 1]
        against = [w for w in scored if w.get("evidence_type") == "contradicts"][:n_external // 2 + 1]
        chosen = (support + against)[:n_external]
        external = {"n_found": len(works), "n_enqueued": 0,
                    "supporting": [{"title": w["title"][:110], "year": w.get("year"),
                                    "doi": w.get("doi"), "reason": w.get("score_reason", "")}
                                   for w in support],
                    "contradicting": [{"title": w["title"][:110], "year": w.get("year"),
                                       "doi": w.get("doi"), "reason": w.get("score_reason", "")}
                                      for w in against]}
        eq = search.enqueue(chosen, fetch_fulltext=True, request_id=rid)
        external["n_enqueued"] = eq["added"]
        external["enqueue"] = eq
        worker.control("start")

    result = {"request_id": rid, "question": question,
              "local_evidence": [{"title": (h["title"] or "")[:110], "page": h["page"],
                                  "section": h["section"], "text": h["text"][:900]}
                                 for h in local],
              "query_plan": plan, "external": external,
              "warning": ("外部新收录的文献需要摄取完成后才能进入检索与图谱；"
                          "用 lit_status 看进度。")}
    db.ex("UPDATE lit_requests SET queries=?, n_found=?, n_ingested=?, status='done', "
          "result=? WHERE id=?",
          (json.dumps(plan, ensure_ascii=False),
           external.get("n_found", 0) if isinstance(external, dict) else 0,
           external.get("n_enqueued", 0) if isinstance(external, dict) else 0,
           json.dumps(result, ensure_ascii=False)[:60000], rid))
    return result


def _novelty_cache_key(statement: str, search_web: bool) -> str:
    normalized = re.sub(r"\s+", " ", statement.strip().lower())
    digest = hashlib.sha256(
        f"v2|web={int(bool(search_web))}|{normalized}".encode("utf-8")).hexdigest()
    return f"lit_novelty_cache:{digest}"


@tool("lit_novelty_check",
      "★ 新颖性查重：给一条命题，先查本地语料，再查 OpenAlex，"
      "判定它是 已知复现 / 领域内新 / 跨学科迁移新 / 全新。"
      "相同命题在缓存有效期内直接复用结果，不重复调用 API。"
      "任何自称『新知识』的卡片必须先过这一关。",
      obj({"statement": P("string", "要查重的命题"),
           "search_web": P("boolean", "是否联网查 OpenAlex，默认 true")},
          ["statement"]), category="lit")
def lit_novelty_check(statement: str, search_web: bool = True,
                      max_seconds: float | None = None) -> dict:
    started = time.monotonic()
    budget = (max(8.0, float(max_seconds)) if max_seconds is not None
              else max(30.0, float(CFG.get("literature.novelty_max_seconds", 120))))

    def remaining() -> float:
        return max(0.0, budget - (time.monotonic() - started))

    cache_key = _novelty_cache_key(statement, search_web)
    cached = db.kv_get(cache_key)
    ttl = float(CFG.get("literature.novelty_cache_hours", 24)) * 3600
    if isinstance(cached, dict) and time.time() - float(cached.get("created_at", 0)) <= ttl:
        result = dict(cached.get("result") or {})
        if result:
            result["cached"] = True
            result["cache_age_seconds"] = round(
                max(0.0, time.time() - float(cached["created_at"])), 1)
            report_tool_progress("查重命中缓存，不重复调用 API", 0.96)
            return result

    if tool_cancel_requested():
        return {"cancelled": True, "stage": "before_local_search"}
    report_tool_progress("查重：正在检索本地语料", 0.12)
    local = index.hybrid_search(statement, top_k=6)
    web: list[dict] = []
    if search_web and remaining() > 20:
        try:
            report_tool_progress("查重：V3.2 正在整理 OpenAlex 检索式", 0.26)
            plan = search.expand_queries(
                statement, stance="supports",
                model=CFG.literature_preprocess_model,
                request_timeout=min(35.0, max(10.0, remaining() - 15.0)),
                attempts=1)
            queries = (plan.get("en") or [statement])[:2]
            for i, qq in enumerate(queries):
                if tool_cancel_requested() or remaining() <= 12:
                    break
                report_tool_progress(
                    f"查重：OpenAlex 元数据检索 {i + 1}/{len(queries)}",
                    0.36 + 0.12 * i)
                web.extend(search.s_openalex(
                    qq, 6, timeout=min(12.0, max(5.0, remaining() - 5.0)),
                    attempts=1))
        except Exception:  # noqa: BLE001 - 联网失败仍可做本地保守判断
            web = []

    # One work can be returned by both expanded queries.  Do not pay V4-Pro to
    # judge duplicate metadata.
    unique_web: list[dict] = []
    seen: set[str] = set()
    for work in web:
        key = ((work.get("doi") or "").strip().lower()
               or re.sub(r"[^a-z0-9]+", "", (work.get("title") or "").lower()))
        if key and key not in seen:
            seen.add(key)
            unique_web.append(work)
    web = unique_web

    if tool_cancel_requested():
        return {"cancelled": True, "stage": "before_novelty_judgement",
                "n_local_hits": len(local), "n_web_hits": len(web)}
    if remaining() < 8:
        return {"error": "查重时间预算已耗尽，未调用最终新颖性判断模型。",
                "partial": True, "n_local_hits": len(local),
                "n_web_hits": len(web)}

    # Final novelty classification is a critical scientific judgement and
    # remains on V4-Pro.  Thinking is disabled because the required output is a
    # compact evidence classification, not a long-form mechanism derivation.
    llm = LLM("bowen", model=CFG.llm_model, fallbacks=[CFG.llm_model],
              usage_kind="literature_novelty")
    ctx = "【本地语料命中】\n" + "\n".join(
        f"- ({h['year']}) {h['title']}: {h['text'][:400]}" for h in local)
    ctx += "\n\n【OpenAlex 命中】\n" + "\n".join(
        f"- ({w.get('year')}) {w['title']}: {(w.get('abstract') or '')[:300]}"
        for w in web[:12])
    sys = ("你在判定一条科学命题的新颖性。严格、保守：只要检索结果里有实质等价的表述，"
           "就判为 rediscovery。输出 JSON："
           '{"verdict":"rediscovery|in_field_new|cross_domain_new|novel",'
           '"closest_prior":[{"title":str,"year":int,"why_similar":str}],'
           '"what_is_actually_new":str,"confidence":0-1,"reasoning":str}')
    try:
        report_tool_progress("查重：V4-Pro 正在判断实质等价性", 0.68)
        j = llm.ask_json(
            sys, f"命题：{statement}\n\n检索结果：\n{ctx}",
            temperature=0.1, thinking=False,
            request_timeout=min(70.0, max(8.0, remaining() - 2.0)),
            attempts=1)
    except Exception as e:  # noqa: BLE001
        return {"error": f"判定失败: {e}", "n_local_hits": len(local),
                "n_web_hits": len(web), "partial": True}
    j["n_local_hits"] = len(local)
    j["n_web_hits"] = len(web)
    j["completed"] = True
    j["cached"] = False
    j["statement_hash"] = cache_key.rsplit(":", 1)[-1][:16]
    j["note"] = "rediscovery 不是坏结果 —— 它是语料自洽性的正向证据，但不能当新知识上报。"
    db.kv_set(cache_key, {"created_at": time.time(), "result": j})
    report_tool_progress("查重完成，结果已缓存", 0.96)
    return j


# ================= 知识图谱 =================
@tool("lit_kg_stats", "知识图谱规模与结构统计。", obj({}), category="lit")
def lit_kg_stats() -> dict:
    return kg.stats()


@tool("lit_kg_neighbors",
      "查询图谱中某个实体（化合物/膜/机理/描述符）的邻居与支撑原文引语。",
      obj({"name": P("string", "实体名"),
           "node_type": P("string", "可选类型过滤：Compound/Membrane/Mechanism/"
                          "Descriptor/Condition/Observation/Concept")},
          ["name"]), category="lit")
def lit_kg_neighbors(name: str, node_type: str = "") -> dict:
    return kg.neighbors(name, node_type)


@tool("lit_kg_export", "把知识图谱导出为 GraphML（可用 Gephi/Cytoscape 打开）。",
      obj({}), category="lit")
def lit_kg_export() -> dict:
    p = CFG.abs_path("store/knowledge_graph.graphml")
    kg.export_graphml(str(p))
    return {"path": str(p), **kg.stats()}


# ================= 全文补充（上传） =================
@tool("lit_needs_fulltext",
      "★ 列出所有拿不到全文的文献：只有摘要的、Crossref 连摘要都没有而失败的、"
      "以及扫描件需 OCR 的。这些就是等你手动上传 PDF 的对象，附 DOI 链接方便你去下。",
      obj({}), category="lit")
def lit_needs_fulltext() -> dict:
    rows = worker.needs_fulltext()
    by = {}
    for r in rows:
        by[r["status"]] = by.get(r["status"], 0) + 1
    return {"n": len(rows), "by_status": by, "papers": rows,
            "action": ("拿到 PDF 后用 lit_attach_fulltext(paper_id, file_path) 绑定，"
                       "系统会清掉摘要级残留并重新抽取全文，证据级别升为 fulltext。")}


@tool("lit_attach_fulltext",
      "★ 给一条已有文献记录补上全文 PDF（摘要级 → 全文级）。"
      "会清掉旧的摘要块与主张，立即加入后台学习队列并启动 worker，"
      "不会产生重复记录，也不会阻塞当前对话。",
      obj({"paper_id": P("string", "要补全文的文献 id（从 lit_needs_fulltext 拿）"),
           "file_path": P("string", "本地 PDF 路径"),
           "reingest": P("boolean", "是否立即启动后台学习，默认 true")},
          ["paper_id", "file_path"]), category="lit", long_running=True)
def lit_attach_fulltext(paper_id: str, file_path: str, reingest: bool = True) -> dict:
    return worker.attach_fulltext(paper_id, file_path, reingest=reingest)


@tool("lit_upload_paper",
      "上传一篇全新的 PDF 入库（不绑定已有记录）。可选填题名和 DOI，会先按 DOI 查重；"
      "上传后立即加入后台学习队列并启动 worker。",
      obj({"file_path": P("string", "本地 PDF 路径"),
           "title": P("string", "题名，留空则用文件名"),
           "doi": P("string", "DOI，可选但建议填，用于查重"),
           "reingest": P("boolean", "是否立即启动后台学习，默认 true")},
          ["file_path"]), category="lit", long_running=True)
def lit_upload_paper(file_path: str, title: str = "", doi: str = "",
                     reingest: bool = True) -> dict:
    return worker.register_new_pdf(file_path, title=title, doi=doi, reingest=reingest)


# ================= 图谱自然语言导读 =================
@tool("lit_kg_explain",
      "★ 知识图谱自然语言导读。图看不懂就用它——把图谱翻译成大白话："
      "scope='overview' 讲整张图在说什么、枢纽是谁、哪些因素有方向冲突、有什么局限；"
      "scope='entity' 讲某个膜/化合物/描述符的邻域（带原文引语）；"
      "scope='conflicts' 逐个解读方向冲突的因素。全部基于数据库实际内容，不编造。",
      obj({"scope": P("string", "overview | entity | conflicts",
                      enum=["overview", "entity", "conflicts"]),
           "entity": P("string", "scope=entity 时的实体名，如 NF270 / log Kow / PFOS")}),
      category="lit", long_running=True)
def lit_kg_explain(scope: str = "overview", entity: str = "") -> dict:
    from ..lit import kgviz
    if scope == "entity":
        if not entity:
            return {"error": "scope=entity 需要给 entity 名称"}
        return kgviz.narrate_entity(entity)
    if scope == "conflicts":
        return kgviz.narrate_conflicts()
    return kgviz.narrate_overview()


@tool("lit_kg_facts",
      "图谱事实速览（不调 LLM，秒出）：枢纽化合物/膜/描述符、"
      "各描述符的效应方向票数与是否冲突、关系类型分布、证据级别构成。",
      obj({"top_n": P("integer", "每类返回前几个，默认 12")}), category="lit")
def lit_kg_facts(top_n: int = 12) -> dict:
    from ..lit import kgviz
    return kgviz.graph_facts(top_n=top_n)
