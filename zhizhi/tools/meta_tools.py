"""跨层公共工具：系统总览、Agent 互相咨询、人工批注记忆。"""
from __future__ import annotations

import json
import time
from contextvars import ContextVar

from ..core import db
from ..core.config import CFG
from ..core.tools import P, obj, tool
from ..dataio import loader


_CONSULT_DEPTH: ContextVar[int] = ContextVar("agent_consult_depth", default=0)


@tool("system_overview",
      "系统总览：数据底座、模型状态、文献库进度、卡片统计、Token 消耗。"
      "不知道从哪开始时先调这个。",
      obj({}), category="meta")
def system_overview() -> dict:
    h = loader.data_health()
    cards = {r["status"]: r["c"] for r in
             db.q("SELECT status, COUNT(*) c FROM cards GROUP BY status")}
    desc = {r["status"]: r["c"] for r in
            db.q("SELECT status, COUNT(*) c FROM descriptors GROUP BY status")}
    usage = db.q1("SELECT COALESCE(SUM(prompt_tokens),0) pt, "
                  "COALESCE(SUM(completion_tokens),0) ct FROM llm_usage")
    papers = {r["status"]: r["c"] for r in
              db.q("SELECT status, COUNT(*) c FROM papers GROUP BY status")}
    pinned = db.q1("SELECT COUNT(*) a, SUM(status='done') b FROM papers WHERE pinned=1")
    return {
        "data": {k: h[k] for k in ("n_rows", "n_compounds", "n_membranes",
                                   "n_references", "coverage_pct",
                                   "rows_complete_on_20_features")},
        "literature": {"papers_by_status": papers,
                       "core_corpus": f"{pinned['b'] or 0}/{pinned['a'] or 0}",
                       "n_chunks": db.q1("SELECT COUNT(*) c FROM chunks")["c"],
                       "n_claims": db.q1("SELECT COUNT(*) c FROM claims")["c"],
                       "n_contradictions": db.q1(
                           "SELECT COUNT(*) c FROM contradictions")["c"]},
        "discovery": {"cards_by_status": cards, "descriptors_by_status": desc},
        "llm_usage": {"prompt_tokens": usage["pt"], "completion_tokens": usage["ct"]},
        "agents": {"博闻 BOWEN": "文献层", "量衡 LIANGHENG": "模型层",
                   "格物 GEWU": "发现层(核心)", "验真 YANZHEN": "验证层"},
    }


@tool("agent_consult",
      "向另一个智能体提一个问题并拿回答案（单次咨询，不共享对话历史）。"
      "可选：bowen(文献) / liangheng(模型) / gewu(发现) / yanzhen(验证)。",
      obj({"agent": P("string", "被咨询的智能体",
                      enum=["bowen", "liangheng", "gewu", "yanzhen"]),
           "question": P("string", "问题（说清楚你需要什么，越具体越好）")},
          ["agent", "question"]), category="meta", long_running=True)
def agent_consult(agent: str, question: str) -> dict:
    depth = _CONSULT_DEPTH.get()
    max_depth = max(1, int(CFG.get("llm.consult_max_depth", 1)))
    if depth >= max_depth:
        return {"error": "agent_consult 递归调用已熔断；请直接用当前已有证据回答。",
                "blocked": "recursive_agent_consult", "depth": depth}

    from ..agents.registry import get_agent
    from ..core.agent import new_session
    a = get_agent(agent)
    if a is None:
        return {"error": f"未知智能体 {agent}"}

    token = _CONSULT_DEPTH.set(depth + 1)
    try:
        # Internal consultation must not replace the conversation selected by
        # the user in the consulted agent's UI.
        sid = new_session(agent, title=f"[咨询] {question[:30]}", make_active=False)
        answer, calls = "", []
        consult_iters = max(1, int(CFG.get("llm.consult_max_tool_iters", 4)))
        context = ("这是一次有预算的单次内部咨询。禁止再次调用 agent_consult；"
                   f"最多进行 {consult_iters} 轮，拿到足够证据后立即给出结论。")
        for ev in a.run(sid, question, extra_context=context,
                        max_iters=consult_iters, thinking=False):
            if ev["type"] == "text":
                answer = ev["text"]
            elif ev["type"] == "tool_call":
                calls.append(ev["name"])
            elif ev["type"] in ("error", "cancelled"):
                answer += f"\n[停止] {ev['text']}"
        return {"agent": f"{a.cn_name} {a.en_name}", "session_id": sid,
                "tools_used": calls, "answer": answer}
    finally:
        _CONSULT_DEPTH.reset(token)


@tool("remember_note",
      "把一条人工批注/教训写进长期记忆，之后每轮对话都会带上。"
      "用于记录『被驳回的原因』『不要再犯的错』『课题组的特定约束』。",
      obj({"agent": P("string", "写给哪个智能体；'*' 表示全体"),
           "kind": P("string", "类型：correction 纠正 / constraint 约束 / "
                     "preference 偏好 / fact 事实"),
           "content": P("string", "内容，一句话")},
          ["agent", "kind", "content"]), category="meta")
def remember_note(agent: str, kind: str, content: str) -> dict:
    db.ex("INSERT INTO memory(agent,kind,content,created_at) VALUES(?,?,?,?)",
          (agent, kind, content, time.time()))
    return {"saved": True, "agent": agent, "kind": kind}


@tool("review_card",
      "人工审卡结果登记：通过 / 驳回 / 存疑，并写批注。"
      "驳回时批注会自动写入发现层的长期记忆，避免重复犯同类错。",
      obj({"card_id": P("string", "卡片 id"),
           "decision": P("string", "approve 通过 | reject 驳回 | hold 存疑",
                         enum=["approve", "reject", "hold"]),
           "note": P("string", "批注理由")},
          ["card_id", "decision", "note"]), category="meta")
def review_card(card_id: str, decision: str, note: str) -> dict:
    r = db.q1("SELECT id,title FROM cards WHERE id=?", (card_id,))
    if not r:
        return {"error": "卡片不存在"}
    status = {"approve": "passed", "reject": "rejected", "hold": "parked"}[decision]
    db.ex("UPDATE cards SET status=?, review=?, updated_at=? WHERE id=?",
          (status, f"[{decision}] {note}", time.time(), card_id))
    if decision == "reject":
        db.ex("INSERT INTO memory(agent,kind,content,created_at) VALUES(?,?,?,?)",
              ("gewu", "correction",
               f"卡片《{r['title']}》被人工驳回，理由：{note}。今后避免同类问题。",
               time.time()))
    return {"card_id": card_id, "status": status, "note": note}


@tool("export_report",
      "导出全局发现报告 Markdown：数据体检 + 模型真实能力 + 所有卡片 + 负结果。",
      obj({}), category="meta")
def export_report() -> dict:
    from ..core.config import CFG
    from ..ml import model as M
    lines = ["# 致知 ZHIZHI 发现报告",
             f"\n生成时间：{time.strftime('%Y-%m-%d %H:%M')}\n",
             "## 一、数据底座体检\n", "```json",
             json.dumps(loader.data_health(), ensure_ascii=False, indent=1), "```\n",
             "## 二、模型真实能力\n", "```json",
             json.dumps(M.legacy_report(), ensure_ascii=False, indent=1), "```\n",
             "## 三、发现卡片\n"]
    for r in db.q("SELECT * FROM cards ORDER BY created_at DESC"):
        l1 = db.jdict(r["l1_result"])
        lines += [f"### [{r['status']}] {r['id']} — {r['title']}", "",
                  f"- 引擎：`{r['engine']}`　新颖性：**{r['novelty']}**"
                  f"　预注册：{'有' if r['prereg_hash'] else '**无**'}",
                  f"- L1 判定：{l1.get('verdict', '未跑')}", "",
                  f"> {r['statement']}", ""]
        if r["review"]:
            lines += [f"人工批注：{r['review']}", ""]
    lines += ["## 四、负结果（已排除的解释）\n"]
    for r in db.q("SELECT name,status,hypothesis,metrics FROM descriptors "
                  "WHERE status IN ('failed','redundant','tested')"):
        m = db.jdict(r["metrics"])
        lines.append(f"- `{r['name']}` [{r['status']}] ΔR²={m.get('delta_r2')} "
                     f"— {(r['hypothesis'] or '')[:150]}")
    md = "\n".join(lines)
    p = CFG.abs_path("store/发现报告.md")
    p.write_text(md, encoding="utf-8")
    return {"file": str(p), "n_chars": len(md)}
