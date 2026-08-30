"""非阻塞 Agent 任务执行器。

不同 Agent / 不同会话在线程池中并行；同一会话严格串行，避免 messages 顺序交叉。
任务状态同时写入通用 tasks 表，页面切换不会中断后台运行。
"""
from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import db, remote
from .config import CFG
from .tools import tool_runtime

_lock = threading.RLock()
_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(
    max_workers=max(1, int(CFG.get("agents.max_workers", 4))),
    thread_name_prefix="zhizhi-agent",
)
_recovered = False


def _default_agent_factory(agent_key: str):
    from ..agents.registry import get_agent
    return get_agent(agent_key)


_agent_factory = _default_agent_factory


def recover_orphaned_jobs() -> None:
    """进程重启后，旧线程已不存在，把遗留 running/queued 标为失败。"""
    if remote.enabled():
        remote.call("jobs", "recover_orphaned_jobs")
        return
    global _recovered
    with _lock:
        if _recovered:
            return
        db.ex("UPDATE tasks SET state='failed',progress=1.0,"
              "message='应用重启，后台 Agent 任务已中断',"
              "updated_at=? WHERE kind='agent_run' AND state IN ('queued','running','cancelling')",
              (time.time(),))
        _recovered = True


def _active_for_session(session_id: str) -> dict | None:
    return next((j for j in _jobs.values()
                 if j["session_id"] == session_id and
                 j["state"] in ("queued", "running", "cancelling")),
                None)


def submit(agent_key: str, session_id: str, prompt: str,
           thinking: bool | None = None) -> dict:
    if remote.enabled():
        return remote.call("jobs", "submit", agent_key, session_id, prompt,
                           thinking=thinking)
    recover_orphaned_jobs()
    if thinking is None:
        thinking = bool(CFG.get("llm.chat_thinking", True))
    with _lock:
        active = _active_for_session(session_id)
        if active:
            return {"error": "同一会话已有任务运行中；可新建会话并行运行。",
                    "job": _public(active)}
        job_id = f"A{time.strftime('%m%d')}-{uuid.uuid4().hex[:8]}"
        now = time.time()
        task_id = db.task_add("agent_run", job_id, f"{agent_key}: {prompt[:100]}")
        job = {
            "id": job_id, "task_id": task_id, "agent": agent_key,
            "session_id": session_id, "prompt": prompt, "thinking": bool(thinking),
            "state": "queued", "created_at": now, "started_at": None, "ended_at": None,
            "status": "等待线程", "progress": 0.0, "step": 0,
            "live_text": "", "final_text": "", "reasoning_tail": "",
            "tools": [], "error": "", "cancel_requested": False,
        }
        _jobs[job_id] = job
        _executor.submit(_run, job_id)
        return {"submitted": True, "job": _public(job)}


def _set(job_id: str, **values: Any) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        if job:
            job.update(values)
        return job


def _run(job_id: str) -> None:
    job = _set(job_id, state="running", started_at=time.time(), status="调用模型",
               progress=0.05)
    if not job:
        return
    db.task_set(job["task_id"], state="running", progress=0.05, message="调用模型")
    agent = _agent_factory(job["agent"])
    if agent is None:
        _finish_failed(job_id, f"未知 Agent: {job['agent']}")
        return
    max_steps = max(1, int(CFG.get("llm.max_tool_iters", 8)))
    try:
        failed = ""
        cancelled = ""
        with tool_runtime(
                progress=lambda message, fraction: _tool_progress(
                    job_id, message, fraction),
                should_cancel=lambda: _is_cancel_requested(job_id)):
            for ev in agent.run(job["session_id"], job["prompt"],
                                thinking=job["thinking"],
                                should_cancel=lambda: _is_cancel_requested(job_id)):
                kind = ev["type"]
                if kind == "delta":
                    with _lock:
                        j = _jobs[job_id]
                        j["live_text"] += ev["text"]
                        j["status"] = "生成回答"
                elif kind == "reasoning":
                    with _lock:
                        j = _jobs[job_id]
                        j["reasoning_tail"] = (j["reasoning_tail"] + ev["text"])[-600:]
                        j["status"] = "深度思考"
                elif kind == "tool_call":
                    with _lock:
                        j = _jobs[job_id]
                        j["tools"].append(ev["name"])
                        j["status"] = f"调用工具 {ev['name']}"
                        j["tool_progress_base"] = float(j.get("progress") or 0.05)
                    db.task_set(job["task_id"], message=f"调用工具 {ev['name']}")
                elif kind == "tool_result":
                    with _lock:
                        j = _jobs[job_id]
                        j["status"] = f"{ev['name']} 完成，继续推理"
                        j["live_text"] = ""
                    db.task_set(job["task_id"], message=f"{ev['name']} 完成")
                elif kind == "llm_done":
                    progress = min(0.9, 0.12 + ev["step"] / max_steps * 0.72)
                    current = float((_jobs.get(job_id) or {}).get("progress") or 0)
                    progress = max(current, progress)
                    _set(job_id, step=ev["step"], progress=progress,
                         status=f"第 {ev['step']} 轮模型完成")
                    db.task_set(job["task_id"], progress=progress,
                                message=f"第 {ev['step']} 轮模型完成")
                elif kind == "text":
                    _set(job_id, final_text=ev["text"], live_text=ev["text"])
                elif kind == "error":
                    failed = ev["text"]
                    break
                elif kind == "cancelled":
                    cancelled = ev["text"]
                    break
        if cancelled or _is_cancel_requested(job_id):
            _finish_cancelled(job_id, cancelled or "用户已停止任务。")
        elif failed:
            _finish_failed(job_id, failed)
        else:
            ended = time.time()
            _set(job_id, state="done", ended_at=ended, status="完成", progress=1.0,
                 reasoning_tail="")
            db.task_set(job["task_id"], state="done", progress=1.0, message="完成")
    except Exception as e:  # noqa: BLE001
        _finish_failed(job_id, f"{type(e).__name__}: {e}")


def _tool_progress(job_id: str, message: str, fraction: float | None) -> None:
    """Persist progress emitted from inside a synchronous long-running tool."""
    with _lock:
        job = _jobs.get(job_id)
        if not job or job["state"] not in ("running", "cancelling"):
            return
        tool_name = job["tools"][-1] if job.get("tools") else "工具"
        status = f"{tool_name} · {message}"
        progress = float(job.get("progress") or 0.05)
        if fraction is not None:
            base = float(job.get("tool_progress_base") or progress)
            # A long tool owns a visible 18-point band.  The final synthesis
            # still retains room to move to 100% when the Agent responds.
            progress = max(progress, min(0.90, base + 0.18 * float(fraction)))
        job["status"] = status
        job["progress"] = progress
        task_id = job["task_id"]
    db.task_set(task_id, progress=progress, message=status[:250])


def _finish_failed(job_id: str, error: str) -> None:
    job = _set(job_id, state="failed", ended_at=time.time(), status="失败",
               error=error[:1000], progress=1.0)
    if job:
        db.task_set(job["task_id"], state="failed", progress=1.0,
                    message=error[:250])


def _is_cancel_requested(job_id: str) -> bool:
    with _lock:
        return bool((_jobs.get(job_id) or {}).get("cancel_requested"))


def cancel(job_id: str) -> dict:
    """Cooperatively stop a queued/running job at the next model/tool boundary."""
    if remote.enabled():
        return remote.call("jobs", "cancel", job_id)
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return {"error": f"任务 {job_id} 不在当前进程中，可能已结束或应用已重启。"}
        if job["state"] not in ("queued", "running", "cancelling"):
            return {"job": _public(job), "already_finished": True}
        job["cancel_requested"] = True
        job["state"] = "cancelling"
        job["status"] = "正在停止"
        task_id = job["task_id"]
    db.task_set(task_id, message="正在停止 Agent 任务")
    return {"cancel_requested": True, "job": _public(job)}


def _finish_cancelled(job_id: str, reason: str) -> None:
    job = _set(job_id, state="cancelled", ended_at=time.time(), status="已停止",
               error=reason[:1000], progress=1.0, reasoning_tail="")
    if job:
        db.task_set(job["task_id"], state="cancelled", progress=1.0,
                    message=reason[:250])


def _public(job: dict) -> dict:
    return {k: job.get(k) for k in (
        "id", "agent", "session_id", "prompt", "state", "created_at", "started_at",
        "ended_at", "status", "progress", "step", "live_text", "final_text",
        "reasoning_tail", "tools", "error")}


def get(job_id: str) -> dict | None:
    if remote.enabled():
        return remote.call("jobs", "get", job_id)
    with _lock:
        job = _jobs.get(job_id)
        return _public(job) if job else None


def list_jobs(agent_key: str = "", session_id: str = "",
              active_only: bool = False, limit: int = 20) -> list[dict]:
    # The UI reads jobs before a user submits a new prompt.  Recover here as
    # well, otherwise rows left by a previous process can look permanently
    # running until the next submission.
    if remote.enabled():
        return remote.call("jobs", "list_jobs", agent_key, session_id,
                           active_only, limit)
    recover_orphaned_jobs()
    with _lock:
        rows = list(_jobs.values())
        if agent_key:
            rows = [j for j in rows if j["agent"] == agent_key]
        if session_id:
            rows = [j for j in rows if j["session_id"] == session_id]
        if active_only:
            rows = [j for j in rows if j["state"] in ("queued", "running", "cancelling")]
        rows.sort(key=lambda j: j["created_at"], reverse=True)
        return [_public(j) for j in rows[:limit]]
