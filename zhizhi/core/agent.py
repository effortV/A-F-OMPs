"""Agent 循环：system prompt + 工具集 + 会话记忆 + 黑板。

run() 是生成器，逐步 yield 事件，UI 可实时展示"它正在调用哪个工具"。
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, Callable, Iterator

from . import db, remote
from .config import CFG
from .llm import LLM, approx_tokens
from .tools import REGISTRY, to_model_text


_ACTIVE_SESSION_PREFIX = "agent_active_session:"


def set_active_session(agent: str, session_id: str) -> str:
    """Persist the last session selected for an agent.

    Streamlit removes widget state when a page is not rendered.  Keeping this
    pointer in SQLite makes switching agents (and even restarting the UI)
    return to the exact conversation the user left.
    """
    if remote.enabled():
        return remote.call("agent", "set_active_session", agent, session_id)
    row = db.q1("SELECT id,title FROM sessions WHERE id=? AND agent=?", (session_id, agent))
    if not row:
        raise ValueError(f"会话 {session_id} 不属于智能体 {agent}")
    if str(row["title"] or "").startswith("[咨询]"):
        raise ValueError("内部咨询会话不能设为用户当前会话")
    db.kv_set(f"{_ACTIVE_SESSION_PREFIX}{agent}", session_id)
    return session_id


def new_session(agent: str, title: str = "", make_active: bool = True) -> str:
    if remote.enabled():
        return remote.call("agent", "new_session", agent, title, make_active)
    sid = f"{agent}-{uuid.uuid4().hex[:8]}"
    db.ex("INSERT INTO sessions(id,agent,title,created_at) VALUES(?,?,?,?)",
          (sid, agent, title or time.strftime("%m-%d %H:%M"), time.time()))
    if make_active:
        set_active_session(agent, sid)
    return sid


def delete_session(agent: str, session_id: str) -> str:
    """Delete one user conversation and select/create a safe replacement."""
    if remote.enabled():
        return remote.call("agent", "delete_session", agent, session_id)
    row = db.q1("SELECT id,title FROM sessions WHERE id=? AND agent=?",
                (session_id, agent))
    if not row:
        raise ValueError(f"会话 {session_id} 不属于智能体 {agent}")
    if str(row["title"] or "").startswith("[咨询]"):
        raise ValueError("内部咨询会话不能从用户界面删除")

    pointer_key = f"{_ACTIVE_SESSION_PREFIX}{agent}"
    was_active = db.kv_get(pointer_key) == session_id
    with db.conn() as connection:
        connection.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        connection.execute("DELETE FROM sessions WHERE id=? AND agent=?",
                           (session_id, agent))
        if was_active:
            connection.execute("DELETE FROM kv WHERE k=?", (pointer_key,))
    return active_session(agent, create=True)


def list_sessions(agent: str, include_internal: bool = False) -> list[dict]:
    if remote.enabled():
        return remote.call("agent", "list_sessions", agent, include_internal)
    sql = "SELECT * FROM sessions WHERE agent=?"
    if not include_internal:
        sql += " AND COALESCE(title,'') NOT LIKE '[咨询]%'"
    sql += " ORDER BY created_at DESC"
    return db.rows_to_dicts(db.q(sql, (agent,)))


def active_session(agent: str, create: bool = True) -> str | None:
    """Return the persisted active session, repairing stale pointers safely."""
    if remote.enabled():
        return remote.call("agent", "active_session", agent, create)
    sid = db.kv_get(f"{_ACTIVE_SESSION_PREFIX}{agent}")
    if sid:
        row = db.q1("SELECT id,title FROM sessions WHERE id=? AND agent=?", (sid, agent))
        if row and not str(row["title"] or "").startswith("[咨询]"):
            return str(row["id"])
    sessions = list_sessions(agent)
    if sessions:
        return set_active_session(agent, str(sessions[0]["id"]))
    return new_session(agent) if create else None


def load_history(session_id: str) -> list[dict]:
    """从 DB 还原可直接喂给模型的 message 列表。"""
    out: list[dict] = []
    for r in db.q("SELECT role,content,extra FROM messages WHERE session_id=? ORDER BY id",
                  (session_id,)):
        extra = json.loads(r["extra"]) if r["extra"] else {}
        msg: dict[str, Any] = {"role": r["role"], "content": r["content"]}
        if extra.get("tool_calls"):
            msg["tool_calls"] = extra["tool_calls"]
        if extra.get("tool_call_id"):
            msg["tool_call_id"] = extra["tool_call_id"]
        if extra.get("name"):
            msg["name"] = extra["name"]
        out.append(msg)
    return out


def save_message(session_id: str, role: str, content: str, extra: dict | None = None) -> None:
    db.ex("INSERT INTO messages(session_id,role,content,extra,created_at) VALUES(?,?,?,?,?)",
          (session_id, role, content or "",
           json.dumps(extra, ensure_ascii=False) if extra else None, time.time()))


def visible_history(session_id: str) -> list[dict]:
    """只给 UI 看的对话（过滤掉 tool 消息体，保留工具调用摘要）。"""
    if remote.enabled():
        return remote.call("agent", "visible_history", session_id)
    out = []
    for r in db.q("SELECT role,content,extra FROM messages WHERE session_id=? ORDER BY id",
                  (session_id,)):
        extra = json.loads(r["extra"]) if r["extra"] else {}
        if r["role"] == "tool":
            out.append({"role": "tool", "name": extra.get("name", "?"),
                        "content": (r["content"] or "")[:2500]})
        elif r["role"] == "assistant":
            out.append({"role": "assistant", "content": r["content"] or "",
                        "calls": [tc["function"]["name"] for tc in extra.get("tool_calls", [])]})
        elif r["role"] == "user":
            out.append({"role": "user", "content": r["content"]})
    return out


class Agent:
    """一个可对话、可调工具、有长期记忆的智能体。"""

    def __init__(self, key: str, cn_name: str, en_name: str, role: str,
                 system_prompt: str, tool_names: list[str]):
        self.key = key
        self.cn_name = cn_name
        self.en_name = en_name
        self.role = role
        self.system_prompt = system_prompt
        self.tool_names = tool_names
        # 面向用户的所有 Agent 回答固定走 V4-Pro，不被低价预处理模型或降级链替换。
        self.llm = LLM(agent=key, model=CFG.llm_model, fallbacks=[CFG.llm_model],
                       usage_kind="agent_chat")

    # ---- 长期记忆 -----------------------------------------------------
    def remember(self, kind: str, content: str) -> None:
        db.ex("INSERT INTO memory(agent,kind,content,created_at) VALUES(?,?,?,?)",
              (self.key, kind, content, time.time()))

    def recall(self, limit: int = 24) -> str:
        rows = db.q("SELECT kind,content FROM memory WHERE agent IN (?,'*')"
                    " ORDER BY id DESC LIMIT ?", (self.key, limit))
        if not rows:
            return ""
        items = "\n".join(f"- [{r['kind']}] {r['content']}" for r in reversed(rows))
        return f"\n\n## 长期记忆（人工批注与既往教训，务必遵守）\n{items}"

    # ---- 上下文压缩 ---------------------------------------------------
    def _compress(self, msgs: list[dict]) -> list[dict]:
        budget = int(CFG.get("llm.memory_budget_tokens", 60000))
        total = sum(approx_tokens(str(m.get("content", ""))) for m in msgs)
        if total <= budget or len(msgs) <= 8:
            return msgs
        head, tail = msgs[:-6], msgs[-6:]
        digest = "\n".join(
            f"{m['role']}: {str(m.get('content',''))[:600]}" for m in head)[:24000]
        try:
            summary = self.llm.ask(
                "你在压缩一段科研对话历史。保留：已确认的结论、数字、卡片编号、"
                "被否决的方案及原因、待办。丢弃：寒暄与重复。用中文条目输出，不超过 600 字。",
                digest, temperature=0.1, thinking=False)
        except Exception:  # noqa: BLE001
            summary = digest[:3000]
        return [{"role": "system", "content": f"## 早前对话摘要\n{summary}"}] + tail

    # ---- 主循环 -------------------------------------------------------
    def run(self, session_id: str, user_msg: str, extra_context: str = "",
            stream: bool = True, thinking: bool | None = None,
            max_iters: int | None = None,
            should_cancel: Callable[[], bool] | None = None) -> Iterator[dict]:
        """yield 事件：
        {'type': 'delta'}   流式文本增量（stream=True 时）
        {'type': 'reasoning'} 思维链增量（模型开启思考时）
        {'type': 'tool_call' | 'tool_result' | 'text' | 'error' | 'done'}
        """
        save_message(session_id, "user", user_msg)
        sys_content = self.system_prompt + self.recall()
        if extra_context:
            sys_content += f"\n\n## 本轮附加上下文\n{extra_context}"
        if thinking is None:
            thinking = bool(CFG.get("llm.chat_thinking", True))

        history = self._compress(load_history(session_id))
        msgs = [{"role": "system", "content": sys_content}] + history
        schemas = REGISTRY.schemas(self.tool_names)
        max_iters = max(1, int(max_iters or CFG.get("llm.max_tool_iters", 8)))
        max_tool_calls = max(1, int(CFG.get("llm.max_tool_calls", 8)))
        max_repeats = max(1, int(CFG.get("llm.max_repeated_tool_calls", 2)))
        max_context = max(4000, int(CFG.get("llm.max_agent_context_tokens", 18000)))
        max_tool_chars = max(1000, int(CFG.get("llm.max_tool_result_chars", 6000)))
        max_seconds = max(30, int(CFG.get("llm.max_agent_run_seconds", 300)))
        started = time.monotonic()
        tool_calls_used = 0
        repeated: dict[str, int] = {}
        budget_notice_added = False

        for step in range(max_iters):
            if should_cancel and should_cancel():
                yield {"type": "cancelled", "text": "用户已停止任务。"}
                return

            # Tool outputs accumulate across rounds. Compact old ones before they
            # inflate every later V4-Pro request and repeatedly charge for the
            # same evidence.
            context_tokens = sum(
                approx_tokens(str(m.get("content", ""))) +
                approx_tokens(json.dumps(m.get("tool_calls", []), ensure_ascii=False))
                for m in msgs)
            if context_tokens > max_context:
                tool_positions = [i for i, m in enumerate(msgs) if m.get("role") == "tool"]
                for i in tool_positions[:-2]:
                    content = str(msgs[i].get("content", ""))
                    if len(content) > 1600:
                        msgs[i]["content"] = (content[:1600] +
                                              "\n…[旧工具结果已压缩，完整结果保存在产出文件中]")

            force_final = (tool_calls_used >= max_tool_calls or
                           time.monotonic() - started >= max_seconds)
            if force_final and not budget_notice_added:
                msgs.append({
                    "role": "system",
                    "content": ("工具或时间预算已经用完。禁止继续调用任何工具；请立即根据"
                                "已有证据给出当前最佳结论，并明确尚缺哪些信息。"),
                })
                budget_notice_added = True
            t_step = time.time()
            try:
                if stream:
                    assembled = None
                    for kind, payload in self.llm.chat_stream(
                            msgs, tools=None if force_final else (schemas or None),
                            thinking=thinking):
                        if should_cancel and should_cancel():
                            yield {"type": "cancelled", "text": "用户已停止任务。"}
                            return
                        if kind == "delta":
                            yield {"type": "delta", "text": payload}
                        elif kind == "reasoning":
                            yield {"type": "reasoning", "text": payload}
                        else:
                            assembled = payload
                    content = (assembled or {}).get("content", "")
                    tool_calls = [tc for tc in (assembled or {}).get("tool_calls", [])
                                  if tc.get("function", {}).get("name")]
                else:
                    m = self.llm.chat(
                        msgs, tools=None if force_final else (schemas or None),
                        thinking=thinking)
                    content = m.content or ""
                    tool_calls = [{"id": tc.id, "type": "function",
                                   "function": {"name": tc.function.name,
                                                "arguments": tc.function.arguments}}
                                  for tc in (m.tool_calls or [])]
            except Exception as e:  # noqa: BLE001
                yield {"type": "error", "text": f"LLM 调用失败：{e}"}
                return
            yield {"type": "llm_done", "seconds": round(time.time() - t_step, 1),
                   "step": step + 1}

            if not tool_calls:
                save_message(session_id, "assistant", content)
                yield {"type": "text", "text": content}
                yield {"type": "done", "steps": step}
                return

            msgs.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            save_message(session_id, "assistant", content, {"tool_calls": tool_calls})
            if content.strip():
                yield {"type": "text", "text": content}

            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except Exception:  # noqa: BLE001
                    args = {}
                yield {"type": "tool_call", "name": name, "args": args}
                t_tool = time.time()
                tool_calls_used += 1
                signature = name + ":" + json.dumps(args, ensure_ascii=False, sort_keys=True)
                seen = repeated.get(signature, 0)
                repeated[signature] = seen + 1
                if should_cancel and should_cancel():
                    yield {"type": "cancelled", "text": "用户已停止任务。"}
                    return
                if tool_calls_used > max_tool_calls:
                    result = {"error": "工具调用预算已耗尽，请直接综合已有结果作答。"}
                elif seen >= max_repeats:
                    result = {"error": f"重复调用 {name} 已被熔断，请使用已有结果作答。"}
                elif time.monotonic() - started >= max_seconds:
                    result = {"error": "本轮运行时间预算已耗尽，请直接综合已有结果作答。"}
                else:
                    result = REGISTRY.call(name, args, agent=self.key)
                text = to_model_text(result, max_chars=max_tool_chars)
                tool_secs = round(time.time() - t_tool, 1)
                msgs.append({"role": "tool", "tool_call_id": tc["id"],
                             "name": name, "content": text})
                save_message(session_id, "tool", text,
                             {"tool_call_id": tc["id"], "name": name})
                yield {"type": "tool_result", "name": name, "result": result,
                       "text": text, "seconds": tool_secs}

        yield {"type": "error", "text": f"达到最大工具轮数 {max_iters}，已停止。"}

    def ask(self, session_id: str, user_msg: str, extra_context: str = "",
            thinking: bool | None = None) -> str:
        """非流式便捷入口：返回最终文本。"""
        final = ""
        for ev in self.run(session_id, user_msg, extra_context,
                           stream=False, thinking=thinking):
            if ev["type"] == "text":
                final = ev["text"]
            elif ev["type"] == "error":
                final += f"\n\n[错误] {ev['text']}"
        return final
