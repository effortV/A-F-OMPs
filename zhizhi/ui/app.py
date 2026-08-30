"""致知 ZHIZHI —— Streamlit 控制台（顶部导航版）。

启动：streamlit run zhizhi/ui/app.py    或    python -m zhizhi.cli ui
"""
from __future__ import annotations

import importlib
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from zhizhi.agents.registry import META, all_agents  # noqa: E402
from zhizhi.core import db, jobs  # noqa: E402
from zhizhi.core.agent import (active_session, delete_session, list_sessions,
                               new_session, set_active_session,
                               visible_history)  # noqa: E402
from zhizhi.core.config import CFG  # noqa: E402

st.set_page_config(page_title="致知 ZHIZHI", page_icon="🧭", layout="wide",
                   initial_sidebar_state="collapsed")
db.init()
all_agents()
jobs.recover_orphaned_jobs()

# 上次退出时 worker 是 running 的话，新进程要把线程拉回来，
# 否则界面显示 running 但 0 个活跃线程，队列会静默卡住。
if db.kv_get("lit_worker", "paused") == "running":
    from zhizhi.lit import worker as _worker
    _worker.ensure_thread()

# 定时文献任务配置保存在 SQLite；进程重启后从下一次到期时间继续，
# 不会因为切换页面或 Streamlit rerun 丢失。Streamlit 热更新时可能先重跑
# app.py、但仍缓存旧版工具模块；只在缺少新入口时刷新一次该模块。
from zhizhi.tools import lit_tools as _lit_tools  # noqa: E402
from zhizhi.lit import search as _search_module  # noqa: E402
if getattr(_search_module, "SEARCH_API_VERSION", 0) < 2:
    _search_module = importlib.reload(_search_module)
if getattr(_lit_tools, "LIT_TOOLS_API_VERSION", 0) < 3:
    # Keep the already-running daemon object across a Streamlit hot reload.
    # Replacing the module without restoring it would start a second scheduler
    # thread; both would then compete for the same due task.
    _old_lit_scheduler = getattr(_lit_tools, "_SCHEDULE_THREAD", None)
    _lit_tools = importlib.reload(_lit_tools)
    if _old_lit_scheduler is not None and _old_lit_scheduler.is_alive():
        _lit_tools._SCHEDULE_THREAD = _old_lit_scheduler
from zhizhi.lit import extract as _extract_module, worker as _worker_module  # noqa: E402
if getattr(_extract_module, "EXTRACT_API_VERSION", 0) < 2:
    _extract_module = importlib.reload(_extract_module)
if getattr(_worker_module, "WORKER_API_VERSION", 0) < 2:
    _previous_worker_state = db.kv_get("lit_worker", "paused")
    db.kv_set("lit_worker", "stopped")
    for _thread in list(getattr(_worker_module, "_threads", [])):
        _thread.join(timeout=3.0)
    _worker_module = importlib.reload(_worker_module)
    db.kv_set("lit_worker", _previous_worker_state)
    if _previous_worker_state == "running":
        _worker_module.ensure_thread()
_lit_tools.ensure_literature_scheduler()


def _backfill_first_schedule_paper_ids() -> None:
    """Recover exact IDs for a first run made by the pre-tracking hot module.

    The expansion executor is serial, and all papers from one enqueue batch are
    inserted between last_queued_at and last_run_at.  Only write the mapping
    when that time window contains exactly the recorded number of additions;
    otherwise leave it explicitly untracked rather than guessing.
    """
    rows = db.q("SELECT ref FROM tasks WHERE kind='lit_schedule' AND state!='deleted'")
    for row in rows:
        key = f"lit_schedule_meta:{row['ref']}"
        config = db.kv_get(key, {})
        if (not isinstance(config, dict) or config.get("paper_ids")
                or int(config.get("runs_completed") or 0) != 1):
            continue
        expected = int(config.get("cumulative_added") or 0)
        start = float(config.get("last_queued_at") or 0)
        end = float(config.get("last_run_at") or 0) + 5.0
        if expected <= 0 or start <= 0 or end <= start:
            continue
        candidates = db.q(
            "SELECT id FROM papers WHERE added_at BETWEEN ? AND ? ORDER BY added_at,id",
            (start, end))
        if len(candidates) != expected:
            continue
        config["paper_ids"] = [item["id"] for item in candidates]
        config["paper_ids_backfilled_at"] = time.time()
        config["paper_ids_backfill_method"] = "exact_first_run_time_window"
        db.kv_set(key, config)


_backfill_first_schedule_paper_ids()

# 颜色一律用半透明灰，深浅两套主题都能用；不硬编码具体色值。
st.markdown("""
<style>
section[data-testid="stSidebar"] {display:none;}
/* 铺满整个窗口宽度，不留大片空白 */
.block-container {padding: 1.0rem 2.2rem 2rem 2.2rem !important; max-width: 100% !important;}
html, body, [class*="css"] {font-size: 16.5px;}
.stMarkdown p, .stMarkdown li {font-size: 1.02rem; line-height: 1.72;}
.stCaption, div[data-testid="stCaptionContainer"] p {font-size: .92rem !important;}
div[data-testid="stMetricValue"] {font-size: 1.7rem;}
div[data-testid="stMetricLabel"] p {font-size: .95rem;}
button[data-baseweb="tab"] {font-size: 1.02rem; padding: .55rem 1.0rem;}
.stDataFrame, .stDataFrame td, .stDataFrame th {font-size: .95rem;}
div[data-testid="stExpander"] summary p {font-size: 1.0rem;}
.stButton button, .stDownloadButton button {font-size: .98rem;}
h3 {font-size: 1.5rem !important;}
h4 {font-size: 1.22rem !important;}
.zz-tool {font-family: ui-monospace, Menlo, Consolas, monospace; font-size:.88rem;
          opacity:.72;}
.zz-badge {display:inline-block; padding:.15rem .6rem; border-radius:6px;
           font-size:.85rem; margin-right:.4rem;
           border:1px solid rgba(120,132,148,.42);
           background:rgba(120,132,148,.10);}
.zz-brand {font-size:1.5rem; font-weight:700; letter-spacing:.5px; margin:0;}
.zz-sub {font-size:.88rem; opacity:.68; margin:0;}
.zz-topbar {border-bottom:1px solid rgba(120,132,148,.28); padding-bottom:.5rem;
            margin-bottom:1.0rem;}
div[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:rgba(120,132,148,.30);}
</style>
""", unsafe_allow_html=True)


_AGENT_WIDGET_KEYS = {
    "lit_autorefresh", "litflt", "lit_failflt", "up_title", "up_doi", "exp_topic", "exp_max",
    "lit_sched_topic", "lit_sched_interval", "lit_sched_count",
    "imp_metric", "pred_smi", "pred_nfeat", "e1_n", "e1_lit", "e1_web",
    "e2_lit", "e3_mode", "e3_dom", "e3_nov", "e3_web", "val_eff", "val_sd",
    "val_alpha", "val_power_lvl",
}


def _keep_agent_widget_state() -> None:
    """Detach durable agent controls from Streamlit's hidden-widget cleanup."""
    for key in list(st.session_state):
        if (key in _AGENT_WIDGET_KEYS or key.startswith("deep_") or
                key.startswith("pv_")):
            st.session_state[key] = st.session_state[key]


def persistent_agent_tabs(agent_key: str, labels: list[str]):
    """Tab-like workspaces whose selected view survives agent/page switches."""
    state_key = f"agent_view_{agent_key}"
    db_key = f"agent_active_view:{agent_key}"
    forced = st.session_state.pop(f"agent_view_next_{agent_key}", None)
    stored = forced or st.session_state.get(state_key) or db.kv_get(db_key, labels[0])
    if stored not in labels:
        stored = labels[0]
    widget_key = f"agent_tabnav_{agent_key}"
    if forced in labels:
        # The upload action is handled after the navigation widget is created.
        # Apply its requested view here, before recreating that widget.
        st.session_state[widget_key] = stored
        st.session_state[f"agent_tabradio_{agent_key}"] = stored
    try:
        selected = st.segmented_control(
            "工作区", labels, default=stored, key=widget_key,
            label_visibility="collapsed")
    except Exception:  # noqa: BLE001 兼容没有 segmented_control 的旧版本
        selected = st.radio(
            "工作区", labels, index=labels.index(stored), horizontal=True,
            key=f"agent_tabradio_{agent_key}", label_visibility="collapsed")
    selected = selected or stored
    st.session_state[state_key] = selected
    if selected != db.kv_get(db_key, labels[0]):
        db.kv_set(db_key, selected)

    container_keys = [f"agent_panel_{agent_key}_{i}" for i in range(len(labels))]
    hidden = [f".st-key-{key} {{display:none !important;}}"
              for key, label in zip(container_keys, labels) if label != selected]
    if hidden:
        st.markdown("<style>" + "".join(hidden) + "</style>", unsafe_allow_html=True)
    return [st.container(key=key) for key in container_keys]


_keep_agent_widget_state()


# ============================ 顶栏 ============================
PAGES = ["🧭 总览", "📚 博闻 · 文献层", "⚖️ 量衡 · 模型层", "🔬 格物 · 发现层",
         "🧪 验真 · 验证层", "🗂 卡片审阅台", "🕸 知识图谱", "⚙️ 任务监视器"]


@st.cache_data(ttl=3, show_spinner=False)
def _topbar_stats() -> dict:
    pinned = db.q1("SELECT COUNT(*) a, COALESCE(SUM(status='done'),0) b "
                   "FROM papers WHERE pinned=1")
    running = db.q1("SELECT COUNT(*) c FROM tasks WHERE state IN ('running','queued')")
    u = db.q1("SELECT COALESCE(SUM(prompt_tokens),0) p, "
              "COALESCE(SUM(completion_tokens),0) c FROM llm_usage")
    cards = {r["status"]: r["c"] for r in
             db.q("SELECT status, COUNT(*) c FROM cards GROUP BY status")}
    lit_running = db.rows_to_dicts(db.q(
        "SELECT COALESCE(NULLIF(p.title,''),t.label) label,t.ref,t.message,t.progress "
        "FROM tasks t LEFT JOIN papers p ON p.id=t.ref "
        "WHERE t.kind='ingest_pdf' AND t.state='running' ORDER BY t.id LIMIT 4"))
    return {"pin_done": pinned["b"] or 0, "pin_total": pinned["a"] or 0,
            "active": running["c"], "pt": u["p"], "ct": u["c"], "cards": cards,
            "lit_running": lit_running}


def topbar() -> str:
    st.session_state.setdefault("nav_open", True)
    st.session_state.setdefault("page", PAGES[0])
    s = _topbar_stats()

    st.markdown('<div class="zz-topbar">', unsafe_allow_html=True)
    head = st.columns([0.5, 5.5, 3.4])
    with head[0]:
        arrow = "⌄" if st.session_state["nav_open"] else "›"
        if st.button(arrow, key="nav_toggle", help="展开 / 收起导航栏"):
            st.session_state["nav_open"] = not st.session_state["nav_open"]
            st.rerun()
    with head[1]:
        st.markdown('<p class="zz-brand">🧭 致知 ZHIZHI</p>'
                    '<p class="zz-sub">格物致知 · NF/RO 膜微污染物截留知识发现</p>',
                    unsafe_allow_html=True)
    with head[2]:
        if not st.session_state["nav_open"]:
            st.caption(f"**{st.session_state['page']}**　·　语料 "
                       f"{s['pin_done']}/{s['pin_total']}　·　活动任务 {s['active']}")

    if st.session_state["nav_open"]:
        try:
            picked = st.segmented_control(
                "导航", PAGES, default=st.session_state["page"], key="nav_seg",
                label_visibility="collapsed")
        except Exception:  # noqa: BLE001  老版本 Streamlit 没有 segmented_control
            picked = st.radio("导航", PAGES,
                              index=PAGES.index(st.session_state["page"]),
                              horizontal=True, key="nav_radio",
                              label_visibility="collapsed")
        if picked:
            st.session_state["page"] = picked

        info = st.columns([2.2, 1.2, 2.0, 1.6])
        with info[0]:
            pct = (s["pin_done"] / s["pin_total"]) if s["pin_total"] else 0.0
            st.progress(pct, text=f"必学文献 {s['pin_done']}/{s['pin_total']}")
        info[1].caption(f"活动任务 **{s['active']}**")
        info[2].caption(f"回答 `{CFG.llm_model.split('/')[-1]}` ｜ "
                        f"文献预处理 `{CFG.literature_preprocess_model.split('/')[-1]}` ｜ "
                        f"Token 入 {s['pt']:,} / 出 {s['ct']:,}")
        c = s["cards"]
        info[3].caption(f"卡片 待审 **{c.get('proposed',0)+c.get('tested',0)}** ｜ "
                        f"通过 {c.get('passed',0)} ｜ 证伪 {c.get('refuted',0)}")
        if s["lit_running"]:
            current = "　｜　".join(
                f"**{r['label'][:55]}** [`{r['ref']}`] · {r['message']} "
                f"({int((r['progress'] or 0)*100)}%)"
                for r in s["lit_running"])
            st.caption("📚 正在处理论文：" + current)
    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state["page"]


# ============================ 对话组件 ============================
def agent_chat(key: str) -> None:
    m = META[key]
    sk = f"sid_{key}"
    sessions = list_sessions(key)
    session_ids = {s["id"] for s in sessions}
    sid = st.session_state.get(sk)
    if sid not in session_ids:
        sid = active_session(key)
        sessions = list_sessions(key)
        session_ids = {s["id"] for s in sessions}
    elif sid:
        set_active_session(key, sid)
    if not sid:
        sid = new_session(key)
        sessions = list_sessions(key)
    st.session_state[sk] = sid

    c1, c2, c3, c4 = st.columns([3.0, 1.1, 1.8, 1.4])
    with c1:
        opts = {f"{s['title']}　·　{s['id'][-6:]}": s["id"] for s in sessions}
        if opts:
            names = list(opts)
            desired = next((n for n in names if opts[n] == sid), names[0])
            selector_key = f"sel_{key}"
            if (st.session_state.get(selector_key) not in opts or
                    opts.get(st.session_state.get(selector_key)) != sid):
                # Runs before widget creation, so a newly created/restored
                # session cannot be overwritten by a stale selectbox value.
                st.session_state[selector_key] = desired

            def _session_changed() -> None:
                chosen = opts.get(st.session_state.get(selector_key))
                if chosen:
                    st.session_state[sk] = chosen
                    set_active_session(key, chosen)

            st.selectbox("会话", names, key=selector_key, on_change=_session_changed,
                         label_visibility="collapsed")
            sid = st.session_state[sk]
    active_now = jobs.list_jobs(session_id=sid, active_only=True)
    if c2.button("＋ 新会话", key=f"new_{key}", use_container_width=True):
        st.session_state[sk] = new_session(key)
        st.rerun()
    if c3.button("🗑 删除这个对话", key=f"del_{key}", use_container_width=True,
                 disabled=bool(active_now), help="当前会话运行中时不能删除"):
        if st.session_state.get(sk):
            st.session_state[sk] = delete_session(key, st.session_state[sk])
        st.rerun()
    deep = c4.toggle("深度思考", value=True, key=f"deep_{key}",
                     help="开启后模型会先做思维链推理，质量更高但首字延迟从 2s 涨到 30s+")

    box = st.container(height=470, border=True)
    with box:
        for msg in visible_history(sid):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant", avatar=m["icon"]):
                    if msg["content"]:
                        st.markdown(msg["content"])
                    if msg.get("calls"):
                        st.markdown("<span class='zz-tool'>⚙ " + " · ".join(msg["calls"])
                                    + "</span>", unsafe_allow_html=True)
            else:
                with st.expander(f"⚙ {msg['name']} 返回", expanded=False):
                    st.code(msg["content"], language="json")

    _agent_job_panel(key, sid)
    st.caption("后台并行已开启：可以切换页面，或新建另一个会话/运行其它 Agent；"
               "同一会话会保持串行以防消息交叉。")

    prompt = st.chat_input(
        (f"当前会话运行中；可点『新会话』并行…" if active_now
         else f"和 {m['cn']} {m['en']} 对话…"),
        key=f"in_{key}", disabled=bool(active_now))
    if prompt:
        result = jobs.submit(key, sid, prompt, thinking=deep)
        if result.get("error"):
            st.error(result["error"])
        else:
            st.toast(f"{m['cn']} 已在后台开始运行")
        st.rerun()


def _agent_job_panel_impl(key: str, sid: str) -> None:
    recent = jobs.list_jobs(agent_key=key, session_id=sid, limit=3)
    if not recent:
        return
    active = [j for j in recent if j["state"] in ("queued", "running", "cancelling")]
    latest = active or recent[:1]
    with st.container(border=True):
        for j in latest:
            icon = {"queued": "⏳", "running": "⚙️", "cancelling": "⏹️",
                    "cancelled": "⏹️", "done": "✅", "failed": "❌"}.get(j["state"], "•")
            st.progress(float(j["progress"] or 0),
                        text=f"{icon} {j['status']} · 任务 {j['id']}")
            if j.get("tools"):
                st.caption("工具：" + " → ".join(j["tools"][-6:]))
            if j.get("reasoning_tail") and j["state"] == "running":
                st.caption("💭 " + j["reasoning_tail"][-300:])
            if j.get("live_text"):
                st.markdown(j["live_text"][-4000:])
            if j.get("error"):
                st.error(j["error"])
            if j["state"] in ("queued", "running", "cancelling"):
                if st.button("⏹ 停止此任务", key=f"job_cancel_{j['id']}",
                             use_container_width=True,
                             disabled=j["state"] == "cancelling"):
                    jobs.cancel(j["id"])
                    st.rerun()
        if st.button("🔄 刷新后台状态 / 同步会话", key=f"job_refresh_{key}_{sid}",
                     use_container_width=True):
            st.rerun()


_fragment = getattr(st, "fragment", getattr(st, "experimental_fragment", None))
_agent_job_panel = (_fragment(run_every=1.0)(_agent_job_panel_impl)
                    if _fragment else _agent_job_panel_impl)


def quick_ask(key: str, label: str, prompt: str, help: str = "") -> None:
    if st.button(label, key=f"qa_{key}_{abs(hash(label)) % 999983}",
                 use_container_width=True, help=help or None):
        st.session_state[f"sid_{key}"] = (st.session_state.get(f"sid_{key}")
                                          or new_session(key))
        st.session_state[f"pending_{key}"] = prompt
        st.rerun()


def run_pending(key: str) -> None:
    p = st.session_state.pop(f"pending_{key}", None)
    if not p:
        return
    result = jobs.submit(key, st.session_state[f"sid_{key}"], p)
    if result.get("error") and "同一会话" in result["error"]:
        # 快捷任务自动开新会话，避免被当前同 Agent 会话阻塞。
        st.session_state[f"sid_{key}"] = new_session(key, title=p[:40])
        result = jobs.submit(key, st.session_state[f"sid_{key}"], p)
    if result.get("error"):
        st.error(result["error"])
    else:
        st.toast(f"{META[key]['cn']} 已在后台开始处理")
    st.rerun()


def _fmt_epoch(value) -> str:
    try:
        return time.strftime("%m-%d %H:%M:%S", time.localtime(float(value)))
    except (TypeError, ValueError, OSError):
        return "—"


def _lit_expansion_progress_impl() -> None:
    from zhizhi.tools.lit_tools import lit_expansion_status

    tasks = lit_expansion_status(limit=6).get("tasks", [])
    st.markdown("##### 本次扩充与逐篇学习进度")
    if not tasks:
        st.caption("还没有扩充任务。点击上方「开始扩充」后，这里会实时显示检索、去重、入队和学习进度。")
        return
    state_names = {"queued": "等待检索", "running": "正在检索/入队",
                   "done": "检索已完成", "failed": "扩充失败"}
    for task in tasks:
        learning = task.get("learning") or {}
        with st.container(border=True):
            st.markdown(f"**{task['topic']}**　`{task['ref']}`")
            st.caption(f"{state_names.get(task['state'], task['state'])} · "
                       f"创建于 {_fmt_epoch(task.get('created_at'))}")
            search_progress = float(task.get("progress") or 0)
            st.progress(search_progress,
                        text=(f"扩充阶段 {int(search_progress * 100)}% · "
                              f"{task.get('message') or '等待开始'}"))
            m = st.columns(6)
            m[0].metric("目标上限", task.get("target") or "—")
            m[1].metric("检索发现", task.get("n_found") or 0)
            m[2].metric("相关候选", task.get("n_above_threshold") or 0)
            m[3].metric("去重跳过", task.get("n_skipped_duplicate") or 0)
            m[4].metric("实际新增", task.get("n_added") or 0)
            m[5].metric("已学完", learning.get("done") or 0)
            total = int(learning.get("total") or 0)
            if total:
                counts = learning.get("by_status") or {}
                status_text = " / ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                st.progress(float(learning.get("progress") or 0),
                            text=(f"逐篇学习 {learning.get('done', 0)}/{total} · "
                                  f"{status_text or '等待摄取'}"))
                for paper in (learning.get("running") or [])[:4]:
                    st.caption(f"⚙ 正在学习：{paper['title'][:110]} · "
                               f"{int(float(paper.get('progress') or 0) * 100)}% · "
                               f"{paper.get('message') or ''}")
            elif (task["state"] == "done" and task.get("n_added")
                  and not task.get("tracking_available")):
                st.info(f"这是升级前创建的任务：日志确认实际新增 {task['n_added']} 篇、"
                        f"去重跳过 {task.get('n_skipped_duplicate', 0)} 篇；旧任务未保存逐篇 ID，"
                        "因此只能显示总数。升级后创建的任务会逐篇显示学习进度。")
            elif task["state"] == "done":
                st.info("本轮没有新增论文：候选可能均已学习、低于相关性阈值，或未检索到可用结果。")
            if task.get("error"):
                st.error(task["error"])
    st.caption("“上限篇数”是本轮最多新增数，不等于一定能找到这么多；实际新增会受相关性、去重和检索结果限制。")


def _lit_schedule_action_buttons(task: dict, key_scope: str) -> None:
    """Render identical, working schedule controls on every relevant page."""
    from zhizhi.tools.lit_tools import lit_schedule_control

    state = task.get("state")
    controls = st.columns([1, 1, 3])
    action = None
    if state == "paused":
        if controls[0].button("▶ 开始 / 继续",
                              key=f"{key_scope}_resume_{task['ref']}",
                              use_container_width=True):
            action = "resume"
    elif controls[0].button("⏸ 暂停", key=f"{key_scope}_pause_{task['ref']}",
                            use_container_width=True,
                            help="正在执行时会在下一个检索/筛选/下载阶段边界安全终止"):
        action = "pause"
    if controls[1].button("🗑 删除任务", key=f"{key_scope}_delete_{task['ref']}",
                          use_container_width=True,
                          help="停止当前及后续轮次；不删除已经入库的文献"):
        action = "delete"
    if not action:
        return
    result = lit_schedule_control(task["ref"], action)
    if result.get("error"):
        st.error(result["error"])
        return
    labels = {"pause": "自动读取任务已暂停", "resume": "自动读取任务已开始 / 继续",
              "delete": "自动读取任务已删除"}
    st.session_state[f"_lit_schedule_flash_{key_scope}"] = labels[action]
    st.rerun()


def _lit_schedule_control_panel(key_scope: str) -> None:
    """Compact schedule panel shared by the ingest queue and task monitor."""
    from zhizhi.tools.lit_tools import lit_schedule_status

    flash_key = f"_lit_schedule_flash_{key_scope}"
    if st.session_state.get(flash_key):
        st.success(st.session_state.pop(flash_key))
    tasks = lit_schedule_status().get("tasks", [])
    if not tasks:
        st.caption("当前没有自动读取任务。可在「博闻 · 文献扩充」中创建。")
        return
    for task in tasks:
        running = bool(task.get("currently_running"))
        state = task.get("state")
        state_text = ("本轮执行中" if running and state != "paused" else
                      "正在安全暂停" if running and state == "paused" else
                      "已暂停" if state == "paused" else "等待下一轮")
        with st.container(border=True):
            st.markdown(f"**{task.get('topic', '')}**　`{task['ref']}`　{state_text}")
            next_text = "本轮结束后重新计时" if running else _fmt_epoch(task.get("next_run_at"))
            st.caption(f"每 {task.get('interval_minutes')} 分钟 · "
                       f"每轮最多 {task.get('papers_per_run')} 篇 · "
                       f"已完成 {task.get('runs_completed', 0)} 轮 · 下次 {next_text}")
            if running:
                progress = float(task.get("progress") or 0)
                st.progress(progress, text=f"{int(progress * 100)}% · "
                            f"{task.get('message') or '正在执行'}")
            _lit_schedule_action_buttons(task, key_scope)


def _lit_schedule_progress_impl() -> None:
    from zhizhi.tools.lit_tools import lit_schedule_status

    flash_key = "_lit_schedule_flash_lit_sched"
    if st.session_state.get(flash_key):
        st.success(st.session_state.pop(flash_key))
    tasks = lit_schedule_status().get("tasks", [])
    st.markdown("##### 定时任务状态")
    if not tasks:
        st.caption("尚未创建定时任务。")
        return
    for task in tasks:
        learning = task.get("learning") or {}
        state = task.get("state")
        running = bool(task.get("currently_running"))
        with st.container(border=True):
            state_text = ("本轮执行中" if running and state != "paused" else
                          "正在安全暂停" if running and state == "paused" else
                          "已暂停" if state == "paused" else "等待下一轮")
            st.markdown(f"**{task['topic']}**　`{task['ref']}`　{state_text}")
            next_text = "本轮完成后重新计时" if running else _fmt_epoch(task.get("next_run_at"))
            st.caption(f"每 {task['interval_minutes']} 分钟 · 每轮最多新增 {task['papers_per_run']} 篇 · "
                       f"已完成 {task.get('runs_completed', 0)} 轮 · 下次 {next_text}")
            if running:
                progress = float(task.get("progress") or 0)
                st.progress(progress, text=f"本轮 {int(progress * 100)}% · {task.get('message') or ''}")
            m = st.columns(5)
            m[0].metric("累计新增", task.get("cumulative_added") or 0)
            m[1].metric("累计去重", task.get("cumulative_duplicates") or 0)
            m[2].metric("已学完", learning.get("done") or 0)
            m[3].metric("待学习", max(0, int(learning.get("total") or 0)
                                        - int(learning.get("done") or 0)))
            last = task.get("last_result") or {}
            m[4].metric("上轮新增", last.get("n_added") or 0)
            if learning.get("total"):
                counts = learning.get("by_status") or {}
                status_text = " / ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                st.progress(float(learning.get("progress") or 0),
                            text=f"累计逐篇学习 {learning.get('done', 0)}/{learning['total']} · {status_text}")
            if task.get("last_error"):
                st.warning(f"上轮失败（下个周期会自动重试）：{task['last_error']}")
            _lit_schedule_action_buttons(task, "lit_sched")
    st.caption("暂停或删除会阻止后续轮次，并让正在执行的一轮在下一个安全阶段边界终止；"
               "不会删除已经学过或已经进入摄取队列的文献。")


_lit_expansion_progress = (_fragment(run_every=2.0)(_lit_expansion_progress_impl)
                           if _fragment else _lit_expansion_progress_impl)
_lit_schedule_progress = (_fragment(run_every=3.0)(_lit_schedule_progress_impl)
                          if _fragment else _lit_schedule_progress_impl)


_FIG_SEQ = {"n": 0}


def show_fig(figinfo: dict, caption: str = "", slot: str = "") -> None:
    """展示 plots.py 产出的 png/svg 双份图。

    key 用「文件名 + 页面槽位 + 本次渲染序号」三重保证唯一：
    同一张图在一次渲染里被展示两次也不会撞 key。
    """
    if not figinfo or "png" not in figinfo:
        return
    st.image(figinfo["png"], caption=caption or None, use_container_width=True)
    _FIG_SEQ["n"] += 1
    seq = _FIG_SEQ["n"]
    cols = st.columns(2)
    for i, k in enumerate(("png", "svg")):
        path = figinfo.get(k)
        if path and Path(path).exists():
            cols[i].download_button(f"下载 {k.upper()}", Path(path).read_bytes(),
                                    file_name=Path(path).name,
                                    key=f"dl_{k}_{slot}_{Path(path).stem}_{seq}",
                                    use_container_width=True)


# ============================ 页面：总览 ============================
def page_overview() -> None:
    from zhizhi.tools.meta_tools import system_overview
    o = system_overview()
    c = st.columns(6)
    c[0].metric("数据行", o["data"]["n_rows"])
    c[1].metric("化合物 × 膜", f"{o['data']['n_compounds']}×{o['data']['n_membranes']}")
    c[2].metric("组合覆盖率", f"{o['data']['coverage_pct']}%")
    c[3].metric("核心语料", o["literature"]["core_corpus"])
    c[4].metric("机理主张", o["literature"]["n_claims"])
    c[5].metric("发现卡片", sum(o["discovery"]["cards_by_status"].values()) or 0)

    st.divider()
    left, right = st.columns([1, 1.15])
    with left:
        st.subheader("生产模型")
        pb1, pb2 = st.columns(2)
        if pb1.button("训练（标准）", key="ov_prod", use_container_width=True):
            with st.spinner("训练中…"):
                from zhizhi.ml import production as PROD
                st.session_state["prod"] = PROD.train_production(mode="base")
        if pb2.button("🚀 增强模式", key="ov_prod_enh", use_container_width=True):
            with st.spinner("训练 8 个模型中…"):
                from zhizhi.ml import production as PROD
                st.session_state["prod"] = PROD.train_production(mode="enhanced")
        p = st.session_state.get("prod")
        if not p:
            st.caption("32 列（12 子结构 + 20 特征），XGBoost。点上方按钮训练。")
        else:
            st.dataframe(pd.DataFrame([
                {"集合": "训练集", "R²": p["train"]["r2"], "RMSE": p["train"]["rmse"],
                 "n": p["train"]["n"]},
                {"集合": "测试集", "R²": p["test"]["r2"], "RMSE": p["test"]["rmse"],
                 "n": p["test"]["n"]},
            ]), use_container_width=True, hide_index=True)

    with right:
        st.subheader("语料与发现进展")
        lit = o["literature"]
        dis = o["discovery"]
        g1, g2 = st.columns(2)
        g1.markdown("**文献层**")
        g1.dataframe(pd.DataFrame([
            {"项": "核心语料", "值": lit["core_corpus"]},
            {"项": "全文分块", "值": lit["n_chunks"]},
            {"项": "机理主张", "值": lit["n_claims"]},
            {"项": "文献矛盾", "值": lit["n_contradictions"]},
        ]), use_container_width=True, hide_index=True)
        g2.markdown("**发现层**")
        cards = dis["cards_by_status"] or {}
        desc = dis["descriptors_by_status"] or {}
        g2.dataframe(pd.DataFrame([
            {"项": "待审卡片", "值": cards.get("proposed", 0) + cards.get("tested", 0)},
            {"项": "已通过", "值": cards.get("passed", 0)},
            {"项": "已证伪", "值": cards.get("refuted", 0)},
            {"项": "描述符", "值": sum(desc.values())},
        ]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("四个智能体")
    cols = st.columns(4)
    for col, (k, m) in zip(cols, META.items()):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {m['icon']} {m['cn']} {m['en']}")
                st.caption(m["role"])
                st.write(m["desc"])


# ============================ 页面：博闻 ============================
def page_bowen() -> None:
    from zhizhi.lit import worker
    m = META["bowen"]
    st.markdown(f"### {m['icon']} {m['cn']} {m['en']}　<span class='zz-sub'>"
                f"{m['role']} · {m['desc']}</span>", unsafe_allow_html=True)
    st.caption(f"模型路由：章节/证据块筛选与基础元数据 → "
               f"`{CFG.literature_preprocess_model}`；机理主张、知识图谱三元组、"
               f"候选相关性判断与回答 → `{CFG.llm_model}`。")
    run_pending("bowen")

    tabs = persistent_agent_tabs(
        "bowen", ["📥 摄取队列", "📤 补全文 / 上传", "🔎 文献扩充", "💬 对话"])

    with tabs[0]:
        st_ = worker.status()
        q = st_["queue"]
        upstream_active = db.rows_to_dicts(db.q(
            "SELECT id,kind,ref,label,state,progress,message,updated_at FROM tasks "
            "WHERE kind IN ('lit_expand','lit_schedule') AND state IN ('queued','running') "
            "ORDER BY updated_at DESC"))
        scheduled_waiting = db.rows_to_dicts(db.q(
            "SELECT ref,label,message FROM tasks WHERE kind='lit_schedule' "
            "AND state='scheduled' ORDER BY updated_at DESC"))
        if upstream_active:
            with st.container(border=True):
                st.markdown("**🔎 上游文献检索 / 扩充正在运行**")
                for task in upstream_active:
                    kind = "定时自主学习" if task["kind"] == "lit_schedule" else "立即扩充"
                    progress = float(task.get("progress") or 0)
                    st.progress(
                        progress,
                        text=(f"{kind} · {task['label']} · {int(progress * 100)}% · "
                              f"{task.get('message') or '等待开始'}"))
                st.caption("这些文献仍在检索、相关性初筛、去重或下载阶段；完成入队后，"
                           "下方的排队、处理中和已学完数量会自动更新。")
        elif scheduled_waiting:
            next_times = []
            for task in scheduled_waiting[:3]:
                config = db.kv_get(f"lit_schedule_meta:{task['ref']}", {})
                if isinstance(config, dict):
                    next_times.append(
                        f"{config.get('topic') or task['label']}：{_fmt_epoch(config.get('next_run_at'))}")
            if next_times:
                st.caption("⏱ 定时自主学习已开启；下一轮：" + "　｜　".join(next_times))
        schedule_count = _lit_tools.lit_schedule_status().get("count", 0)
        with st.expander("⏱ 自动读取任务（暂停 / 开始 / 删除）",
                         expanded=bool(schedule_count)):
            st.caption("这里控制自动检索与入队任务；下方的开始/暂停/停止只控制逐篇摄取 worker。")
            _lit_schedule_control_panel("ingest_schedule")
        last_upload = st.session_state.get("lit_last_upload")
        if last_upload:
            n_ok = sum(1 for r in last_upload if not (r.get("error") or r.get("错误")))
            if n_ok:
                st.success(f"刚刚上传的 {n_ok} 篇文献已进入学习队列，worker 已立即启动。")
            with st.expander("查看本次上传结果", expanded=bool(n_ok != len(last_upload))):
                st.dataframe(pd.DataFrame(last_upload), use_container_width=True,
                             hide_index=True)
        c = st.columns(6)
        c[0].metric("核心语料", st_["core_corpus_progress"])
        c[1].metric("文献总数", st_.get("total_papers", sum(q.values())))
        c[2].metric("已学完", q.get("done", 0))
        c[3].metric("待学 / 处理中", f"{q.get('queued', 0)} / {q.get('running', 0)}")
        c[4].metric("失败", q.get("failed", 0))
        activity_text = {
            "processing": f"处理中 {st_['active_papers']} 篇",
            "idle": "空闲",
            "paused": "已暂停",
            "stopped": "已停止",
        }.get(st_["activity"], st_["activity"])
        c[5].metric("worker", f"{activity_text} ({st_['threads_alive']}/"
                              f"{st_['n_workers_configured']})")
        latest = st_.get("last_completed") or {}
        if latest:
            st.caption(f"最近学完：**{latest.get('title', '')[:120]}** "
                       f"[`{latest.get('paper_id', '')}`] · "
                       f"{_fmt_epoch(st_.get('latest_done_at'))}　｜　"
                       f"最近入库：{_fmt_epoch(st_.get('latest_added_at'))}")
        b = st.columns(6)
        if b[0].button("▶ 开始", key="lit_q_start", use_container_width=True):
            worker.bootstrap_core_corpus()
            worker.control("start")
            st.rerun()
        if b[1].button("⏸ 暂停", key="lit_q_pause", use_container_width=True):
            worker.control("pause")
            st.rerun()
        if b[2].button("⏹ 停止", key="lit_q_stop", use_container_width=True):
            worker.control("stop")
            st.rerun()
        if b[3].button("↻ 注册语料", key="lit_q_register", use_container_width=True):
            st.json(worker.bootstrap_core_corpus())
            st.json(worker.scan_new_pdfs())
        if b[4].button("🔄 刷新", key="lit_q_refresh", use_container_width=True):
            st.rerun()
        auto = b[5].toggle("自动刷新", value=False, key="lit_autorefresh")

        total = int(st_.get("total_papers") or sum(q.values()) or 1)
        done = int(q.get("done", 0))
        st.progress(
            done / total,
            text=(f"最新学习进度 {done}/{total} · 排队 {q.get('queued', 0)} · "
                  f"处理中 {q.get('running', 0)} · 失败 {q.get('failed', 0)}"))
        running_now = st_.get("running_now", [])
        if running_now:
            st.markdown(f"**当前正在处理 {len(running_now)} 篇论文**")
        for r in running_now:
            with st.container(border=True):
                st.markdown(f"**⚙ 正在处理：{r['label'][:100]}**")
                st.caption(f"文献 ID：`{r['ref']}`")
                st.progress(float(r["progress"] or 0),
                            text=f"{r['message']} · {int((r['progress'] or 0)*100)}%")
        if not running_now:
            if st_["activity"] == "idle":
                queued = q.get("queued", 0)
                failed = q.get("failed", 0)
                if queued:
                    detail = f"队列还有 {queued} 篇，正在等待 worker 领取。"
                elif upstream_active:
                    detail = "上游检索仍在进行，尚未产生新的摄取任务；入队后 worker 会自动领取。"
                else:
                    detail = "排队为 0，所以此刻没有可处理的论文。"
                st.info(f"当前没有正在处理的论文。{st_['threads_alive']}/"
                        f"{st_['n_workers_configured']} 个 worker 已启动但处于空闲状态；"
                        f"{detail} 失败 {failed} 篇，可在下方筛选 `failed` 后逐篇重试。")
            elif st_["activity"] == "paused":
                st.warning("当前没有正在处理的论文：文献 worker 已暂停。点击「开始」可继续。")
            elif st_["activity"] == "stopped":
                st.warning("当前没有正在处理的论文：文献 worker 已停止。点击「开始」可重新启动。")
            else:
                st.caption("当前没有正在处理的论文。")

        failure_categories = st_.get("failure_categories") or []
        if failure_categories:
            with st.expander("⚠ 失败原因分类与处理建议"):
                failure_df = pd.DataFrame(failure_categories)
                advice = {
                    "missing_text": "前往「补全文 / 上传」补 PDF 后再学习",
                    "page_format": "页码兼容已修复，可在下方筛选后逐篇重试",
                    "needs_ocr": "需要先对扫描版 PDF 做 OCR",
                    "model_output": "可重试；若反复出现再检查模型返回",
                    "timeout": "接口恢复后重试",
                    "other": "展开单篇错误后判断是否重试",
                }
                failure_df["处理建议"] = failure_df["key"].map(advice).fillna("逐篇检查")
                st.dataframe(failure_df[["label", "count", "处理建议"]],
                             use_container_width=True, hide_index=True)
                st.caption("这里按当前数据库实时统计。系统不会自动重试，避免未经确认再次产生 API 费用。")

        with st.expander("🧹 重复文献与实体别名维护"):
            st.caption("按规范化 DOI、规范化题名和 PDF SHA-256 去重；"
                       "同时合并 NF270 / NF 270 / NF-270 等实体别名。不会删除 PDF 文件。")
            d1, d2 = st.columns(2)
            if d1.button("只扫描重复项", key="lit_dedup_scan", use_container_width=True):
                from zhizhi.lit.dedup import deduplicate_library
                st.session_state["dedup_result"] = deduplicate_library(dry_run=True)
            if d2.button("扫描并合并", key="lit_dedup_apply", use_container_width=True):
                from zhizhi.lit.dedup import deduplicate_library
                st.session_state["dedup_result"] = deduplicate_library(dry_run=False)
            if st.session_state.get("dedup_result"):
                st.json(st.session_state["dedup_result"])

        rows = db.rows_to_dicts(db.q(
            "SELECT p.id,p.title,p.year,p.status,p.evidence_level,p.pinned,p.n_chunks,"
            "p.error,(SELECT COUNT(*) FROM claims c WHERE c.paper_id=p.id) n_claims "
            "FROM papers p ORDER BY p.pinned DESC, p.added_at ASC"))
        if rows:
            dfp = pd.DataFrame(rows)
            dfp["title"] = dfp["title"].fillna("").str[:80]
            dfp["failure_key"] = dfp["error"].fillna("").map(worker.failure_category)
            dfp["failure_reason"] = dfp["failure_key"].map(worker.FAILURE_LABELS)
            f1, f2 = st.columns([1, 3])
            flt = f1.selectbox("筛选", ["全部", "queued", "running", "done", "failed",
                                        "paused", "needs_ocr"], key="litflt")
            view = dfp if flt == "全部" else dfp[dfp["status"] == flt]
            if flt == "failed":
                present_keys = [item["key"] for item in failure_categories]
                options = ["全部原因"] + [worker.FAILURE_LABELS[key] for key in present_keys]
                selected_reason = f1.selectbox("失败原因", options, key="lit_failflt")
                if selected_reason != "全部原因":
                    selected_key = next(
                        key for key in present_keys
                        if worker.FAILURE_LABELS[key] == selected_reason)
                    view = view[view["failure_key"] == selected_key]
            f2.caption(f"共 {len(view)} 篇")
            display_columns = ["id", "title", "year", "status", "n_chunks", "n_claims",
                               "evidence_level", "pinned"]
            if flt == "failed":
                display_columns.append("failure_reason")
            display_columns.append("error")
            st.dataframe(view[display_columns],
                         use_container_width=True, hide_index=True, height=260)
            if not view.empty:
                with st.expander("单篇操作（暂停 / 重试 / 删除 / 立即处理）"):
                    pid = st.selectbox("选择文献", view["id"].tolist(), key="pid_sel")
                    o = st.columns(4)
                    from zhizhi.tools.lit_tools import lit_process_now, lit_task_control
                    if o[0].button("⏸ 暂停", key="lit_one_pause", use_container_width=True):
                        st.json(lit_task_control(pid, "pause"))
                    if o[1].button("↻ 重试", key="lit_one_retry", use_container_width=True):
                        st.json(lit_task_control(pid, "retry"))
                    if o[2].button("🗑 删除", key="lit_one_delete", use_container_width=True):
                        st.json(lit_task_control(pid, "delete"))
                    if o[3].button("⚡ 立即处理", key="lit_one_now", use_container_width=True):
                        with st.spinner("处理中…"):
                            st.json(lit_process_now(pid))
            else:
                st.info("当前筛选条件下没有文献。")
        if auto:
            time.sleep(4)
            st.rerun()

    # ---- 补全文 / 上传 ----
    with tabs[1]:
        st.markdown("**联网检索拿不到全文的文献**——闭源期刊常常连摘要都没有。"
                    "在这里补上 PDF，系统会原地升级为全文级并立即进入后台学习队列，"
                    "不会产生重复记录。")
        need = worker.needs_fulltext()
        st.caption(f"待补全文 **{len(need)}** 篇")
        if need:
            nd = pd.DataFrame(need)
            st.dataframe(nd[["paper_id", "title", "year", "journal", "status",
                             "evidence_level", "why", "doi_url"]],
                         use_container_width=True, hide_index=True, height=240,
                         column_config={"doi_url": st.column_config.LinkColumn("DOI 链接")})
            u1, u2 = st.columns([2, 3])
            with u1:
                labels = {f"{r['title'][:60]} [{r['paper_id']}]": r["paper_id"]
                          for r in need}
                sel = st.selectbox("选一篇补全文", list(labels), key="att_sel")
            with u2:
                att_seq = st.session_state.get("att_file_seq", 0)
                up = st.file_uploader("上传该文献的 PDF", type=["pdf"],
                                      key=f"att_file_{att_seq}")
            if up is not None and st.button("绑定并立即学习", key="att_go",
                                            use_container_width=True):
                tmp = CFG.new_pdf_dir / f"_upload_{time.time_ns()}.pdf"
                tmp.write_bytes(up.getbuffer())
                with st.spinner("保存 PDF 并加入学习队列…"):
                    from zhizhi.tools.lit_tools import lit_attach_fulltext
                    try:
                        res = lit_attach_fulltext(labels[sel], str(tmp), reingest=True)
                    except Exception as exc:  # noqa: BLE001
                        res = {"error": str(exc)}
                    finally:
                        tmp.unlink(missing_ok=True)
                summary = {"文件": up.name, "文献 ID": res.get("paper_id", labels[sel]),
                           "题名": res.get("title", ""),
                           "队列状态": res.get("learning", {}).get("state", ""),
                           "错误": res.get("error", "")}
                st.session_state["lit_last_upload"] = [summary]
                if not res.get("error"):
                    st.session_state["att_file_seq"] = att_seq + 1
                    st.session_state["agent_view_next_bowen"] = "📥 摄取队列"
                    st.rerun()
                st.error(res["error"])

        st.divider()
        st.markdown("**上传全新文献**（不绑定已有记录，支持一次多选）")
        n1, n2 = st.columns(2)
        new_title = n1.text_input("题名（单篇上传时可填；留空用文件名）", key="up_title")
        new_doi = n2.text_input("DOI（单篇上传时可填，用于查重）", key="up_doi")
        upload_seq = st.session_state.get("up_file_seq", 0)
        newfiles = st.file_uploader("PDF 文件（可同时选择多篇）", type=["pdf"],
                                    accept_multiple_files=True,
                                    key=f"up_files_{upload_seq}")
        if len(newfiles) > 1:
            st.caption("批量上传时，每篇题名采用原 PDF 文件名，DOI 留空并由学习流程提取元数据。")
        if newfiles and st.button(f"上传 {len(newfiles)} 篇并立即学习", key="up_go",
                                  use_container_width=True):
            results = []
            with st.spinner(f"正在保存 {len(newfiles)} 篇 PDF 并加入学习队列…"):
                from zhizhi.tools.lit_tools import lit_upload_paper
                for i, uploaded in enumerate(newfiles):
                    tmp = CFG.new_pdf_dir / f"_new_{time.time_ns()}_{i}.pdf"
                    tmp.write_bytes(uploaded.getbuffer())
                    title = ((new_title or "").strip() if len(newfiles) == 1 else "")
                    title = title or Path(uploaded.name).stem
                    doi = ((new_doi or "").strip() if len(newfiles) == 1 else "")
                    try:
                        res = lit_upload_paper(str(tmp), title=title, doi=doi,
                                               reingest=True)
                    except Exception as exc:  # noqa: BLE001 单篇失败不阻塞整批
                        res = {"error": str(exc)}
                    finally:
                        tmp.unlink(missing_ok=True)
                    results.append({
                        "文件": uploaded.name,
                        "文献 ID": res.get("paper_id", res.get("existing_id", "")),
                        "题名": title,
                        "队列状态": res.get("learning", {}).get("state", ""),
                        "错误": res.get("error", ""),
                    })
            st.session_state["lit_last_upload"] = results
            if any(not r["错误"] for r in results):
                st.session_state["up_file_seq"] = upload_seq + 1
                st.session_state["agent_view_next_bowen"] = "📥 摄取队列"
                st.rerun()
            st.error("本批文件均未入队，请查看错误信息后重试。")
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        st.caption("也可以直接把一批 PDF 拷到 `store/pdf_new/`，"
                   "然后回摄取队列点「↻ 注册语料」。")

    # ---- 文献扩充 ----
    with tabs[2]:
        e1, e2, e3 = st.columns([3, 1, 1])
        topic = e1.text_input("主题 / 关键词（LLM 自动扩成中英检索式 + 反例检索式 + "
                              "跨领域检索式）", key="exp_topic")
        maxp = e2.number_input("上限篇数", 10, 100, 40, key="exp_max")
        if e3.button("🔎 开始扩充", key="lit_expand", use_container_width=True) and topic:
            from zhizhi.tools.lit_tools import lit_expand_search
            res = lit_expand_search(topic, max_papers=int(maxp), background=True)
            if res.get("queued"):
                st.success(f"文献扩充已进入后台任务 {res.get('task_ref')}，可切换页面继续工作。")
            elif res.get("error"):
                st.error(res["error"])
            st.json({k: v for k, v in res.items() if k != "query_plan"})
        _lit_expansion_progress()

        st.divider()
        st.markdown("#### ⏱ 定时自主学习")
        st.caption("按设定周期检索新文献；DOI、规范化题名和 PDF 内容哈希统一去重。"
                   "已经学过、正在排队或同轮重复的论文都会跳过。")
        s1, s2, s3, s4 = st.columns([3, 1, 1, 1])
        schedule_topic = s1.text_input("持续学习主题", key="lit_sched_topic",
                                       placeholder="例如：微污染物 去除 纳滤 反渗透膜")
        schedule_interval = s2.number_input(
            "间隔（分钟）", min_value=1, max_value=43200, value=60, step=5,
            key="lit_sched_interval")
        schedule_count = s3.number_input(
            "每轮最多篇数", min_value=1, max_value=100, value=10, step=1,
            key="lit_sched_count")
        if s4.button("▶ 创建并开始", key="lit_sched_create", use_container_width=True):
            from zhizhi.tools.lit_tools import lit_schedule_create
            result = lit_schedule_create(
                schedule_topic, interval_minutes=int(schedule_interval),
                papers_per_run=int(schedule_count))
            if result.get("error"):
                st.error(result["error"])
            else:
                st.toast(f"定时学习任务 {result['ref']} 已创建，首轮立即开始")
                st.rerun()
        st.caption("首轮会立即运行；以后从上一轮结束时重新计时。高频任务会增加检索和模型费用。")
        _lit_schedule_progress()

        st.divider()
        q = st.columns(3)
        with q[0]:
            quick_ask("bowen", "📈 汇报语料进度",
                      "汇报当前文献库进度，核心 59 篇还差几篇，已抽出多少条机理主张，"
                      "图谱规模如何，还有多少篇缺全文。")
        with q[1]:
            quick_ask("bowen", "⚔ 探测文献矛盾",
                      "运行矛盾探测，把发现的矛盾逐条列出来，每条给出双方原文引语、"
                      "条件差异、以及你提出的调和假设。")
        with q[2]:
            quick_ask("bowen", "🕸 图谱概况",
                      "给我知识图谱的规模与结构概况，并列出被最多文献提及的 5 个描述符"
                      "及其效应方向分布。")

    with tabs[3]:
        agent_chat("bowen")


# ============================ 页面：量衡 ============================
def page_liangheng() -> None:
    m = META["liangheng"]
    st.markdown(f"### {m['icon']} {m['cn']} {m['en']}　<span class='zz-sub'>"
                f"{m['role']} · {m['desc']}</span>", unsafe_allow_html=True)
    run_pending("liangheng")

    tabs = persistent_agent_tabs(
        "liangheng", ["🎯 生产模型", "🔮 预测器", "🔍 外推诊断", "💬 对话"])

    # ---- 生产模型 ----
    with tabs[0]:
        a1, a2 = st.columns([1, 1])
        if a1.button("训练并落盘（标准）", key="prod_train", use_container_width=True):
            with st.spinner("训练中…"):
                from zhizhi.ml import production as PROD
                st.session_state["prod"] = PROD.train_production(mode="base")
        if a2.button("🚀 增强模式", key="prod_enh", use_container_width=True):
            with st.spinner("训练 8 个模型中，约 30 秒…"):
                from zhizhi.ml import production as PROD
                st.session_state["prod"] = PROD.train_production(mode="enhanced")
        p = st.session_state.get("prod")
        if p:
            k = st.columns(4)
            k[0].metric("训练 R²", p["train"]["r2"])
            k[1].metric("测试 R²", p["test"]["r2"])
            k[2].metric("测试 RMSE", p["test"]["rmse"])
            k[3].metric("特征数", p["n_features"])
            with st.expander("模型文件与特征清单"):
                st.json({kk: p.get(kk) for kk in
                         ("saved_to", "also_saved", "params", "column_order", "features")
                         if p.get(kk) is not None})

        st.divider()
        st.markdown("**特征重要性**")
        METRICS = {"weight": "weight — 被用作分裂判据的次数",
                   "shap": "SHAP — 对每条预测的实际贡献",
                   "permutation": "置换 — 打乱该列后 R² 跌幅",
                   "gain": "gain — 每次分裂的平均增益",
                   "cover": "cover — 分裂覆盖的样本量"}
        i0, i1, i2 = st.columns([2, 1, 1])
        met = i0.selectbox("重要性口径", list(METRICS),
                           format_func=lambda k: METRICS[k], key="imp_metric")
        if i1.button("计算并出图", key="prod_imp", use_container_width=True):
            with st.spinner("计算中…"):
                from zhizhi.tools.prod_tools import ml_feature_importance
                st.session_state["imp"] = ml_feature_importance(metric=met)
        if i2.button("误差分析图（散点 + 残差）", key="prod_err",
                     use_container_width=True):
            with st.spinner("出图中…"):
                from zhizhi.tools.prod_tools import ml_error_plots
                st.session_state["errfig"] = ml_error_plots(
                    mode="production", residual_vs_feature="compound size (nm)")
        if st.session_state.get("imp"):
            r = st.session_state["imp"]
            st.success(f"**{r['metric']}** 口径下排第一的是 **{r['rank_1']}**"
                       f"　—　{r['metric_meaning_selected']}")
            show_fig(r.get("figure", {}), slot="imp")
            with st.expander("五口径排名对照表"):
                st.dataframe(pd.DataFrame(r["combined"]).head(25),
                             use_container_width=True, hide_index=True)
                st.caption(r["read"])
        if st.session_state.get("errfig"):
            e = st.session_state["errfig"]
            show_fig(e.get("parity", {}), slot="err_parity")
            show_fig(e.get("residuals", {}), slot="err_resid")

    # ---- 预测器 ----
    with tabs[1]:
        st.caption("给 SMILES 自动算 12 个子结构；其余特征填你知道的，不知道的留空"
                   "（XGBoost 原生处理缺失，但缺太多会有警告）。")
        from zhizhi.dataio import loader
        pc = st.columns([2, 1, 1])
        smi = pc[0].text_input("SMILES", value="CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                               key="pred_smi")
        n_show = pc[1].number_input("显示前几个特征", 4, 20, 12, key="pred_nfeat")
        vals: dict = {"SMILES": smi} if smi else {}
        from zhizhi.ml import model as MM
        b = MM.get_bundle()
        keys = [c for c in loader.FEATURES if c in b.X.columns][: int(n_show)]
        grid = st.columns(4)
        for i, kk in enumerate(keys):
            med = float(b.X[kk].median()) if b.X[kk].notna().any() else 0.0
            v = grid[i % 4].number_input(kk[:30], value=round(med, 4), format="%.4f",
                                         key=f"pv_{i}")
            vals[kk] = v
        if st.button("预测截留率", key="pred_go", use_container_width=True):
            from zhizhi.tools.prod_tools import ml_predict_smiles
            res = ml_predict_smiles([vals])
            pr = res["predictions"][0]
            k = st.columns(4)
            k[0].metric("预测截留率", f"{pr['pred_removal_pct']} %")
            k[1].metric("集成不确定度", f"± {pr['ensemble_std']}")
            k[2].metric("缺失特征", f"{pr['n_missing_features']}/{res['n_features_used']}")
            k[3].metric("可靠", "是" if pr["reliable"] else "否")
            if pr["warning"]:
                st.warning(pr["warning"])
            for w in res.get("input_warnings", []):
                st.error(w)

        st.divider()
        st.markdown("**批量预测**：上传含 `SMILES` 列和任意特征列的 Excel/CSV。")
        bf = st.file_uploader("批量文件", type=["xlsx", "csv"], key="batch_file")
        if bf is not None and st.button("批量预测", key="batch_go",
                                        use_container_width=True):
            dfb = (pd.read_csv(bf) if bf.name.endswith(".csv") else pd.read_excel(bf))
            from zhizhi.tools.prod_tools import ml_predict_smiles
            res = ml_predict_smiles(dfb.to_dict("records"))
            outdf = pd.concat([dfb.reset_index(drop=True),
                               pd.DataFrame(res["predictions"])], axis=1)
            st.dataframe(outdf, use_container_width=True, hide_index=True)
            st.download_button("下载结果 CSV",
                               outdf.to_csv(index=False).encode("utf-8-sig"),
                               file_name="predictions.csv", key="batch_dl",
                               use_container_width=True)

    # ---- 外推诊断（供发现层使用的分组口径，与日常预测分开）----
    with tabs[2]:
        st.caption("这一页回答的是另一个问题：结论能不能推广到没见过的分子 / 膜 / 课题组。"
                   "发现层的残差考古依赖这里的分组 OOF 口径。")
        if st.button("口径对照表（多种设定横比）", key="prod_cmp",
                     use_container_width=True):
            with st.spinner("逐个口径训练中，约 1-2 分钟…"):
                from zhizhi.tools.prod_tools import ml_compare_variants
                st.session_state["cmp"] = ml_compare_variants()
        if st.session_state.get("cmp"):
            st.dataframe(pd.DataFrame(st.session_state["cmp"]["table"]),
                         use_container_width=True, hide_index=True)
            st.caption(st.session_state["cmp"]["read"])
        st.divider()
        d = st.columns(4)
        with d[0]:
            quick_ask("liangheng", "📉 残差全景",
                      "先跑 ml_data_qc 排除 SMILES 错标，再跑 ml_residuals，"
                      "告诉我哪些膜/化合物存在系统性偏差、方向是什么。")
        with d[1]:
            quick_ask("liangheng", "🔪 特征组消融",
                      "跑 ml_ablate，告诉我哪组特征最关键、哪组接近无用甚至在拖后腿，"
                      "以及每组消融后哪类分子退化最严重。")
        with d[2]:
            quick_ask("liangheng", "🚀 外推压力测试",
                      "跑 ml_extrapolate（含 membrane_class），"
                      "解释三种分组 R² 落差分别揭示什么失效来源。")
        with d[3]:
            quick_ask("liangheng", "🔍 SHAP 解释",
                      "跑 ml_explain(scope='global') 和 scope='interaction'，"
                      "给出前 12 个特征的重要性与方向，以及最强的 5 组交互。")
        if st.button("导出全部预测结果 CSV", key="exp_pred", use_container_width=True):
            from zhizhi.tools.prod_tools import ml_export_predictions
            st.json(ml_export_predictions())

    with tabs[3]:
        agent_chat("liangheng")


# ============================ 页面：格物 ============================
def page_gewu() -> None:
    m = META["gewu"]
    st.markdown(f"### {m['icon']} {m['cn']} {m['en']}　<span class='zz-sub'>"
                f"{m['role']} · {m['desc']}</span>", unsafe_allow_html=True)
    run_pending("gewu")

    tabs = persistent_agent_tabs(
        "gewu", ["🧩 引擎1 残差考古", "🗺 引擎2 图谱覆盖", "🌐 引擎3 跨学科迁移",
                 "🧬 描述符仓库", "💬 对话"])

    with tabs[0]:
        st.caption("取分组 OOF 残差最大的 20% 样本，在【标准化特征空间 ⊕ Morgan 指纹】上聚类。"
                   "自动标注该簇是否被单篇文献主导（>60% 判为协议差异而非机理）、"
                   "是否与 SMILES 错标重合。")
        a, b_, c_ = st.columns([1.4, 1, 1])
        nclu = a.slider("簇数", 3, 10, 6, key="e1_n")
        e1_lit = b_.toggle("自动查文献", value=True, key="e1_lit",
                           help="为每个簇检索原文段落，并查该因素在文献里的历史效应方向")
        e1_web = b_.toggle("联网补文献", value=False, key="e1_web")
        if c_.button("跑残差聚类", key="disc_e1", use_container_width=True):
            with st.spinner("聚类 + 文献取证中…" if e1_lit else "聚类中…"):
                from zhizhi.tools.disc_tools import disc_residual_clusters
                st.session_state["e1"] = disc_residual_clusters(
                    n_clusters=int(nclu), with_literature=e1_lit, search_web=e1_web)
        if st.session_state.get("e1"):
            r = st.session_state["e1"]
            st.caption(f"分组 CV R²={r['cv_r2']}　阈值 |残差|≥{r['abs_residual_threshold']}"
                       f"　选中 {r['n_selected']} 条　已剔除错标行 {r['n_qc_excluded_rows']}")
            for c in r["clusters"]:
                with st.expander(
                        f"簇 {c['cluster']}　n={c['n']}　平均残差 {c['mean_residual']:+.1f}"
                        f"　{c['direction']}"
                        + ("　⚠ 单篇文献主导" if c["single_reference_dominated"] else "")):
                    cc = st.columns(3)
                    cc[0].metric("化合物数", c["n_compounds"])
                    cc[1].metric("涉及膜数", c["n_membranes"])
                    cc[2].metric("Mw 范围", f"{c['mw_range'][0]}–{c['mw_range'][1]}")
                    st.markdown("**机理特征（固定三项）**")
                    st.dataframe(pd.DataFrame(c.get("mechanism_profile", [])),
                                 use_container_width=True, hide_index=True)
                    st.markdown("**其它模型驱动特征**（不归入机理特征）")
                    st.dataframe(pd.DataFrame(c["feature_profile"]),
                                 use_container_width=True, hide_index=True)
                    st.markdown("**代表分子**")
                    st.dataframe(pd.DataFrame(c["exemplars"]),
                                 use_container_width=True, hide_index=True)
                    lit = c.get("literature") or {}
                    if lit:
                        st.markdown("**📚 文献层证据**（自动取证）")
                        if lit.get("conflict_with_data"):
                            st.error(
                                f"⚠ 数据方向与文献方向**相反**："
                                f"数据显示 {lit['data_direction']}，"
                                f"文献主流是 {lit['literature_direction']}"
                                f"（{lit.get('descriptor_probed')}："
                                f"{lit.get('direction_counts')}）。"
                                "这是最有价值的信号，必须写进命题。")
                        elif lit.get("direction_counts"):
                            st.info(f"「{lit.get('descriptor_probed')}」历史方向分布："
                                    f"{lit['direction_counts']}")
                        for pg in (lit.get("passages") or [])[:4]:
                            st.markdown(
                                f"- **{str(pg.get('title',''))[:70]}** "
                                f"({pg.get('year')}) p.{pg.get('page')}\n\n"
                                f"  <span class='zz-tool'>{str(pg.get('text',''))[:380]}"
                                "</span>", unsafe_allow_html=True)
                        if not lit.get("passages"):
                            st.caption("本地语料没查到相关段落 —— 可开「联网补文献」，"
                                       "或这本身说明该现象未见报道。")
            with st.expander("归因协议（Agent 必须遵守）"):
                for line in r["attribution_protocol"]:
                    st.markdown(f"- {line}")
        quick_ask("gewu", "→ 交给格物做自由归因",
                  "执行引擎1：跑 disc_residual_clusters，对每个系统性残差簇做自由归因"
                  "（随机噪声 / 缺条件变量 / 测量协议差异 / 未建模机理）。"
                  "先用 ml_data_qc 和残差-缺失相关排除数据问题；"
                  "对判定为『未建模机理』的簇写出命题：『现有 2D 特征语言无法表达 ___，"
                  "因为 ___』，并给出可计算的描述符草案。")

    with tabs[1]:
        st.caption("289 化合物 × 51 膜 = 14739 个组合，实际只填了 6.3%。"
                   "空白格按 0.35·预测不确定度 + 0.25·判别力 + 0.20·外推新颖度 "
                   "+ 0.15·残差异常度 + 0.05·可行性 打分。")
        q1_, q2_ = st.columns([1, 3])
        e2_lit = q1_.toggle("交叉查文献", value=True, key="e2_lit",
                            help="数据集里空白 ≠ 文献里没人做过。已有报道的会标出来")
        if q2_.button("扫高价值空白", key="disc_e2", use_container_width=True):
            with st.spinner("打分 + 文献交叉核对中…" if e2_lit else "打分中…"):
                from zhizhi.tools.disc_tools import disc_coverage_map
                st.session_state["e2"] = disc_coverage_map(
                    top_n=25, with_literature=e2_lit)
        if st.session_state.get("e2"):
            r = st.session_state["e2"]
            mx = r["matrix"]
            mc = st.columns(4)
            mc[0].metric("可用化合物", mx["n_compounds_considered"])
            mc[1].metric("可用膜", mx["n_membranes_considered"])
            mc[2].metric("空白组合", mx["blank_cells"])
            mc[3].metric("全局覆盖率", f"{mx['global_coverage_pct']}%")
            lx = r.get("literature_crosscheck") or {}
            if lx.get("checked"):
                st.info(f"已对前 {lx['checked']} 个高分空白查了文献，其中 "
                        f"**{lx['likely_already_studied']}** 个很可能已有报道。"
                        f"{lx['read']}")
            rows = []
            for x in r["high_value_blanks"]:
                lit = x.get("literature") or {}
                rows.append({**{k: v for k, v in x.items() if k != "literature"},
                             "文献已报道": ("是" if lit.get("likely_already_studied")
                                        else ("否" if lit else "未查")),
                             "最相关文献": lit.get("top_hit")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.warning(r["caveat"])
        quick_ask("gewu", "→ 交给格物挑组合",
                  "执行引擎2：跑 disc_coverage_map，从高价值空白里挑出 5 个"
                  "机理上有理由预期反常、且能分开竞争假设的组合，逐个说明理由。")

    with tabs[2]:
        st.caption("**领域不再限于预设池** —— 默认让 Agent 自己根据当前残差异常和文献矛盾"
                   "反推「哪个外领域已经解决过结构类似的问题」，优先挑离膜科学远、"
                   "但机制可类比、且有成熟定量描述符的领域。")
        mode = st.radio("模式", ["auto_propose（Agent 自主提域）", "rotate（轮转池）",
                                 "manual（我指定）"], key="e3_mode", horizontal=True)
        dom = ""
        if mode.startswith("manual"):
            dom = st.text_input("外领域（任意文本，如「晶体工程」「血脑屏障渗透」"
                                "「湍流输运」）", key="e3_dom")
        elif mode.startswith("rotate"):
            pool = list(CFG.get("discovery.cross_domain_pool") or [])
            dom = st.selectbox("轮转池", ["（自动挑没扫过的）"] + pool, key="e3_pool")
            dom = "" if dom.startswith("（") else dom
        o1, o2, o3 = st.columns(3)
        e3_nov = o1.toggle("扫完自动查重（本地语料 + OpenAlex）", value=True, key="e3_nov")
        e3_web = o2.toggle("查重联网读取 OpenAlex 元数据", value=True, key="e3_web")
        e3_expand = o3.toggle("后台扩充相关文献", value=False, key="e3_expand")
        x1, x2 = st.columns(2)
        if x1.button("① 先看它想扫哪些领域", key="e3_propose",
                     use_container_width=True):
            with st.spinner("Agent 正在反推候选领域…"):
                from zhizhi.tools.disc_tools import disc_propose_domains
                try:
                    st.session_state["e3p"] = disc_propose_domains()
                    st.session_state.pop("e3p_error", None)
                except Exception as exc:  # noqa: BLE001
                    st.session_state["e3p_error"] = (
                        f"候选领域返回结构异常：{type(exc).__name__}: {exc}")
        if x2.button("② 深扫（产出可证伪假设）", key="disc_e3",
                     use_container_width=True):
            with st.spinner("跨界迁移推理 + 元数据查重中；核心推理最长允许 1000 秒…"):
                from zhizhi.tools.disc_tools import disc_crossdomain_scan
                scan_mode = mode.split("（")[0]
                scan_domain = dom
                if scan_mode == "auto_propose" and not scan_domain:
                    scan_domain = ((st.session_state.get("e3p") or {})
                                   .get("recommended") or "")
                scan_result = disc_crossdomain_scan(
                    domain=scan_domain, mode=scan_mode,
                    auto_novelty_check=e3_nov, search_web=e3_web,
                    expand_literature=e3_expand)
                st.session_state["e3"] = scan_result
                if (not st.session_state.get("e3p")
                        and scan_result.get("domain_proposal")):
                    proposal = dict(scan_result["domain_proposal"])
                    proposal.setdefault("recommended", scan_result.get("domain"))
                    st.session_state["e3p"] = proposal
        if st.session_state.get("e3p"):
            p = st.session_state["e3p"]
            st.markdown(f"**推荐：{p.get('recommended')}**")
            st.dataframe(pd.DataFrame(p.get("candidates") or []),
                         use_container_width=True, hide_index=True)
            st.caption(p.get("reasoning", ""))
            if mode.startswith("auto_propose"):
                st.caption(f"步骤②将复用该推荐领域：{p.get('recommended')}，不会再次提域。")
        if st.session_state.get("e3p_error"):
            st.error(st.session_state["e3p_error"])
        if st.session_state.get("e3"):
            r = st.session_state["e3"]
            status = r.get("status")
            if not status:  # 兼容升级前留在 session_state 里的旧结果
                status = ("cancelled" if r.get("cancelled") else
                          "error" if r.get("error") else
                          "success" if r.get("scan_valid") else "invalid")
            head = (f"领域：**{r.get('domain') or '未选定'}**"
                    f"（模式 {r.get('mode') or '未知'}）")
            elapsed = r.get("elapsed_seconds")
            timing = f"，耗时 {elapsed} 秒" if elapsed is not None else ""
            if status == "success":
                st.success(f"{head}　扫描有效{timing}")
            elif status == "invalid":
                fields = r.get("missing_fields") or []
                st.error(f"{head}　模型输出不完整，缺字段：{', '.join(fields)}{timing}")
            elif status == "timeout":
                st.warning(f"{head}　在 {r.get('stage') or '推理阶段'}超时{timing}。"
                           "再次点击步骤②会复用同一领域，只重跑深层推理。")
                if r.get("error"):
                    st.caption(r["error"])
            elif status == "cancelled":
                st.info(f"{head}　已在 {r.get('stage') or '当前阶段'}取消{timing}")
            else:
                st.error(f"{head}　在 {r.get('stage') or '未知阶段'}失败{timing}："
                         f"{r.get('error') or '未提供错误详情'}")
            for k_, label in (("donor_concept", "迁移概念"), ("mapping", "映射到膜截留"),
                              ("falsifiable_prediction", "可证伪预测"),
                              ("discriminating_test", "判别性检验"),
                              ("why_not_already_known", "现有特征为何表达不了"),
                              ("risk", "风险")):
                if r.get(k_):
                    st.markdown(f"**{label}**　{r[k_]}")
            if r.get("computable_descriptor"):
                with st.expander("可计算描述符定义"):
                    st.json(r["computable_descriptor"])
            nv = r.get("novelty_check") or {}
            if nv and not nv.get("error"):
                NOVMAP = {"rediscovery": ("🔁 已知复现", st.warning),
                          "in_field_new": ("🆕 领域内新", st.info),
                          "cross_domain_new": ("🌐 跨学科迁移新", st.success),
                          "novel": ("✨ 全新", st.success)}
                label, fn = NOVMAP.get(nv.get("verdict"), ("？", st.info))
                fn(f"**查重结论：{label}**　（本地命中 {nv.get('n_local_hits')} 条 / "
                   f"OpenAlex 命中 {nv.get('n_web_hits')} 条）　"
                   f"{nv.get('what_is_actually_new','')}")
                with st.expander("最接近的既有工作"):
                    st.json(nv.get("closest_prior", []))
            if r.get("literature_expansion_task"):
                task = r["literature_expansion_task"]
                if task.get("queued"):
                    st.info(f"相关文献扩充已转入后台任务 **{task.get('task_ref')}**；"
                            "当前扫描结果无需等待，可在任务监视器查看进度。")
                else:
                    st.warning(task.get("error", "后台文献扩充未成功排队"))
        quick_ask("gewu", "🌐 跨界扫描并落卡",
                  "做一次跨学科扫描（自主提域模式，不要局限于预设池），"
                  "产出可证伪预测 + 可计算描述符 + 判别性检验，"
                  "扫描会自动完成新颖性查重；若结果中 novelty_check_completed=true，"
                  "不要再次调用 lit_novelty_check，直接 disc_create_card 出卡。")

    with tabs[3]:
        from zhizhi.desc import store as dstore
        rows = dstore.listing()
        if rows:
            st.dataframe(pd.DataFrame([
                {"名称": r["name"], "状态": r["status"],
                 "ΔR²": db.jdict(r["metrics"]).get("delta_r2")
                 if r["metrics"] else None,
                 "假设": (r["hypothesis"] or "")[:90]} for r in rows]),
                use_container_width=True, hide_index=True)
        else:
            st.caption("还没有描述符。")
        with st.expander("3D 构象原语库（描述符代码里可直接调 prim.*）"):
            from zhizhi.desc import primitives as prim
            st.json(prim.AVAILABLE)
        quick_ask("gewu", "🧬 走一遍完整描述符闭环",
                  "完整执行一次描述符发现闭环：①跑引擎1找一个最值得解释的残差簇；"
                  "②基于它提出一个 3D 构象描述符假设；③disc_prereg 预注册；"
                  "④写 compute(smiles) 代码用 disc_compute_descriptor 计算；"
                  "⑤ml_add_descriptor 检验；⑥lit_novelty_check 查重；⑦disc_create_card 出卡。"
                  "每一步都要报告结果，失败也要如实说并出卡。")

    with tabs[4]:
        agent_chat("gewu")


# ============================ 页面：验真 ============================
def page_yanzhen() -> None:
    m = META["yanzhen"]
    st.markdown(f"### {m['icon']} {m['cn']} {m['en']}　<span class='zz-sub'>"
                f"{m['role']} · {m['desc']}</span>", unsafe_allow_html=True)
    run_pending("yanzhen")

    tabs = persistent_agent_tabs(
        "yanzhen", ["📋 验证调度", "🔢 L3 实验重复数估算", "💬 对话"])
    with tabs[0]:
        st.caption("成本序 L1（分钟/免费）< L2（机时，有 xtb 廉价前置）< L3（周级/耗材）。"
                   "只有上一层通过才投下一层；L1 就证伪的直接结案，不浪费机时和耗材。")
        if st.button("刷新调度队列", key="val_sched", use_container_width=True):
            from zhizhi.tools.val_tools import val_schedule
            st.session_state["vsched"] = val_schedule()
        s = st.session_state.get("vsched")
        if s and s.get("queue"):
            st.dataframe(pd.DataFrame(s["queue"]), use_container_width=True,
                         hide_index=True)
            st.info(s.get("principle", ""))
        elif s:
            st.caption(s.get("error", "暂无待验证卡片"))
        q = st.columns(3)
        with q[0]:
            quick_ask("yanzhen", "📋 排验证队列",
                      "跑 val_schedule，对每张卡片给出下一步该做什么以及为什么。")
        with q[1]:
            quick_ask("yanzhen", "❌ 负结果清单",
                      "跑 val_negative_results，列出所有被证伪/降级的命题，"
                      "并说明它们各自排除了什么解释。")
        with q[2]:
            quick_ask("yanzhen", "🧪 给最新卡片出 L2+L3",
                      "找到最新一张卡片，为它生成 L2 分子动力学方案"
                      "（含 xtb 廉价先行路线）和 L3 判别性实验设计，然后导出验证工单。")

    with tabs[1]:
        st.info("这里只服务于 **L3 湿实验设计**：估算两组各需多少个独立实验重复。"
                "它不用于 ML 模型验证，也不用于 L2 分子动力学。")
        st.caption("做实验**之前**算：要想可靠看出 X 个百分点的截留差异，每组得重复几次。"
                   "效应量小于测量噪声 1.5 倍时，应换判别力更强的设计。")
        p = st.columns(4)
        eff = p[0].number_input("预期效应量（百分点）", 1.0, 60.0, 10.0, key="val_eff",
                                help="你的假设预测两组会差多少")
        sd = p[1].number_input("测量标准差（百分点）", 0.5, 20.0, 3.0, key="val_sd",
                               help="你实验台单次测量的重复性，用自己的平行样算")
        alpha = p[2].number_input("α", 0.001, 0.2, 0.05, key="val_alpha",
                                  help="假阳性率")
        power = p[3].number_input("功效", 0.5, 0.99, 0.8, key="val_power_lvl",
                                  help="真有差异时能检出的概率")
        if st.button("算重复数", key="val_power", use_container_width=True):
            from zhizhi.tools.val_tools import val_power_analysis
            r = val_power_analysis(eff, sd, alpha, power)
            k = st.columns(4)
            k[0].metric("每组重复", r["n_per_group"])
            k[1].metric("总实验数", r["total_runs"])
            k[2].metric("标准化效应量 d", r["cohens_d"])
            k[3].metric("n=3 时最小可检出", f"{r['minimum_detectable_effect_at_n3_pct']} pp")
            st.info(r["read"])
            with st.expander("适用前提"):
                for assumption in r.get("assumptions", []):
                    st.markdown(f"- {assumption}")

    with tabs[2]:
        agent_chat("yanzhen")


# ============================ 页面：卡片审阅台 ============================
def page_cards() -> None:
    st.markdown("### 🗂 卡片审阅台")
    st.caption("人只做一件事：审卡。通过 / 驳回 / 存疑，驳回理由会写进格物的长期记忆。")
    rows = db.rows_to_dicts(db.q("SELECT * FROM cards ORDER BY created_at DESC"))
    if not rows:
        st.info("还没有卡片。去发现层让格物跑一遍三引擎。")
        return
    f = st.columns(4)
    stf = f[0].multiselect("状态", sorted({r["status"] for r in rows}), key="cf1")
    enf = f[1].multiselect("引擎", sorted({r["engine"] or "" for r in rows}), key="cf2")
    nvf = f[2].multiselect("新颖性", sorted({r["novelty"] or "未查重" for r in rows}),
                           key="cf3")
    f[3].metric("卡片总数", len(rows))
    view = [r for r in rows
            if (not stf or r["status"] in stf)
            and (not enf or (r["engine"] or "") in enf)
            and (not nvf or (r["novelty"] or "未查重") in nvf)]

    NOV = {"rediscovery": "🔁 已知复现", "in_field_new": "🆕 领域内新",
           "cross_domain_new": "🌐 跨学科迁移新", "novel": "✨ 全新",
           None: "⚪ 未查重", "": "⚪ 未查重"}
    for r in view:
        with st.container(border=True):
            head = st.columns([5, 1, 1])
            head[0].markdown(f"#### {r['title']}")
            head[1].markdown(f"`{r['status']}`")
            head[2].markdown(f"`{r['id']}`")
            st.markdown(
                f"<span class='zz-badge'>{r['engine']}</span>"
                f"<span class='zz-badge'>{NOV.get(r['novelty'], r['novelty'])}</span>"
                f"<span class='zz-badge'>预注册 {'✅' if r['prereg_hash'] else '❌'}</span>"
                f"<span class='zz-badge'>L1 {'✅' if r['l1_result'] else '—'}</span>"
                f"<span class='zz-badge'>L2 {'✅' if r['l2_plan'] else '—'}</span>"
                f"<span class='zz-badge'>L3 {'✅' if r['l3_plan'] else '—'}</span>",
                unsafe_allow_html=True)
            st.markdown(f"> {r['statement']}")
            if r["review"]:
                st.info(f"人工批注：{r['review']}")
            tabs = st.tabs(["证据", "预注册", "L1 结果", "L2 方案", "L3 设计", "审阅"])
            with tabs[0]:
                st.json(db.jdict(r["payload"]))
            with tabs[1]:
                if r["prereg"]:
                    st.caption(f"哈希 `{r['prereg_hash']}`")
                    st.json(db.jdict(r["prereg"]))
                else:
                    st.warning("无预注册协议 —— 统计结论只能算探索性。")
            with tabs[2]:
                st.json(db.jdict(r["l1_result"]))
            with tabs[3]:
                st.markdown(r["l2_plan"] or "_未生成_")
            with tabs[4]:
                st.markdown(r["l3_plan"] or "_未生成_")
            with tabs[5]:
                note = st.text_area("批注", key=f"nt_{r['id']}", height=80)
                bb = st.columns(4)
                from zhizhi.tools.meta_tools import review_card
                if bb[0].button("✅ 通过", key=f"ap_{r['id']}", use_container_width=True):
                    review_card(r["id"], "approve", note or "通过")
                    st.rerun()
                if bb[1].button("❌ 驳回", key=f"rj_{r['id']}", use_container_width=True):
                    review_card(r["id"], "reject", note or "驳回")
                    st.rerun()
                if bb[2].button("🤔 存疑", key=f"hd_{r['id']}", use_container_width=True):
                    review_card(r["id"], "hold", note or "存疑")
                    st.rerun()
                if bb[3].button("📄 导出工单", key=f"ex_{r['id']}",
                                use_container_width=True):
                    from zhizhi.tools.val_tools import val_export_workorder
                    st.success(f"已导出：{val_export_workorder(r['id'])['file']}")


# ============================ 页面：知识图谱 ============================
_ENTITY_TYPES = {
    "化合物": "Compound", "膜": "Membrane", "描述符": "Descriptor", "机理": "Mechanism"
}


@st.cache_data(ttl=5, show_spinner=False)
def _entity_options(ntype: str) -> list[dict]:
    from zhizhi.lit import kg
    return kg.entity_choices(ntype, limit=1200)


def entity_picker(key: str, default: str = "NF270") -> str:
    """按实体类型分组的可搜索下拉框，避免要求用户记住节点名。"""
    a, b = st.columns([1, 2.4])
    default_type = "膜" if default.upper().replace("-", "").replace(" ", "") == "NF270" else "化合物"
    labels = list(_ENTITY_TYPES)
    type_label = a.selectbox("实体类型", labels, index=labels.index(default_type),
                             key=f"{key}_type")
    rows = _entity_options(_ENTITY_TYPES[type_label])
    if not rows:
        return b.text_input("实体名", value=default, key=f"{key}_manual")
    names = [r["name"] for r in rows]
    degree = {r["name"]: r["degree"] for r in rows}
    default_norm = default.upper().replace("-", "").replace(" ", "")
    idx = next((i for i, name in enumerate(names)
                if name.upper().replace("-", "").replace(" ", "") == default_norm), 0)
    return b.selectbox("搜索实体", names, index=idx, key=f"{key}_name",
                       format_func=lambda name: f"{name}　·　{degree[name]} 条关系")


def page_graph() -> None:
    from zhizhi.lit import kg, kgviz
    st.markdown("### 🕸 知识图谱")
    s = kg.stats()
    c = st.columns(4)
    c[0].metric("节点", s["n_nodes"])
    c[1].metric("边", s["n_edges"])
    c[2].metric("机理主张", db.q1("SELECT COUNT(*) c FROM claims")["c"])
    c[3].metric("矛盾", db.q1("SELECT COUNT(*) c FROM contradictions")["c"])

    tabs = st.tabs(["🌐 全局图", "🔎 实体邻域", "🔥 矛盾热力图", "📖 图谱导读",
                    "⚔ 矛盾清单", "📤 导出"])

    # ---- 自然语言导读：看完图之后来这里读解释 ----
    with tabs[3]:
        st.caption("一堆点和线看不出名堂？这里把图谱翻译成大白话——"
                   "这张图从多少篇文献抽出来、枢纽是谁、哪些因素方向冲突、"
                   "你该去看什么、以及它的局限在哪。全部基于数据库实际内容。")
        n1, n3 = st.columns(2)
        if n1.button("📖 整张图在说什么", key="kg_narr_all",
                     use_container_width=True):
            with st.spinner("生成导读中，约 25 秒…"):
                st.session_state["kgnarr"] = ("overview", kgviz.narrate_overview())
        if n3.button("📖 逐个解读方向冲突", key="kg_narr_c",
                     use_container_width=True):
            with st.spinner("生成中…"):
                st.session_state["kgnarr"] = ("conflicts", kgviz.narrate_conflicts())
        st.markdown("**按类型搜索一个实体的导读**")
        ent_n = entity_picker("kg_narr", "NF270")
        if st.button("📖 讲讲这个实体", key="kg_narr_e", use_container_width=True):
            with st.spinner("生成中…"):
                st.session_state["kgnarr"] = ("entity", kgviz.narrate_entity(ent_n))
        nr = st.session_state.get("kgnarr")
        if nr:
            scope, res = nr
            if res.get("error"):
                st.error(res["error"])
            else:
                st.markdown(res["narrative"])
                if scope == "overview":
                    with st.expander("导读依据的原始事实（不是 LLM 编的）"):
                        st.json(res["facts"])
                elif scope == "entity":
                    st.caption(f"实体 {res['entity']}　·　{res['n_edges']} 条关系　·　"
                               f"{res['n_claims']} 条相关主张")
                elif scope == "conflicts":
                    st.caption("方向冲突的因素：" +
                               "、".join(res.get("conflicted_descriptors", [])))
        st.divider()
        st.markdown("**事实速览**（不调 LLM，秒出）")
        if st.button("刷新事实", key="kg_facts_btn", use_container_width=True):
            st.session_state["kgfacts"] = kgviz.graph_facts()
        kf = st.session_state.get("kgfacts")
        if kf:
            fc = st.columns(3)
            fc[0].markdown("**枢纽化合物**")
            fc[0].dataframe(pd.DataFrame(kf["hubs"]["Compound"][:8]),
                            hide_index=True, use_container_width=True)
            fc[1].markdown("**枢纽膜**")
            fc[1].dataframe(pd.DataFrame(kf["hubs"]["Membrane"][:8]),
                            hide_index=True, use_container_width=True)
            fc[2].markdown("**描述符方向票数**")
            fc[2].dataframe(pd.DataFrame(kf["descriptors"][:8]),
                            hide_index=True, use_container_width=True)

    with tabs[0]:
        f1, f2, f3, f4 = st.columns([2.2, 1, 1, 1])
        types = f1.multiselect("节点类型", list(kgviz.TYPE_COLOR),
                               default=["Compound", "Membrane", "Descriptor",
                                        "Mechanism", "Concept"], key="kg_types")
        mind = f2.slider("最小连接数", 1, 8, 3, key="kg_mindeg")
        maxn = f3.slider("最多节点", 50, 500, 220, key="kg_maxn")
        lay = f4.selectbox("布局", ["spring", "kamada"], key="kg_layout")
        if st.button("绘制", key="kg_draw", use_container_width=True):
            with st.spinner("布局计算中…"):
                fig, info = kgviz.plotly_graph(types or None, mind, maxn, lay)
                st.session_state["kgfig"] = (fig, info)
        if st.session_state.get("kgfig"):
            fig, info = st.session_state["kgfig"]
            if fig is None:
                st.warning(info.get("hint", "无可绘制节点"))
            else:
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"显示 {info['n_nodes']} 节点 / {info['n_edges']} 边"
                           "（文献节点已隐藏；节点大小 = 连接数）")

    with tabs[1]:
        st.caption("先选类型，再在下拉框中直接搜索；实体别名会在入库维护时自动归一化。")
        ent = entity_picker("kg_nb", "NF270")
        n2, n3 = st.columns([1, 3])
        hops = n2.slider("跳数", 1, 2, 1, key="kg_hops")
        if n3.button("查邻域", key="kg_nb_btn", use_container_width=True) and ent:
            with st.spinner("检索中…"):
                st.session_state["kgnb"] = kgviz.neighborhood_figure(ent, hops)
        if st.session_state.get("kgnb"):
            fig, info = st.session_state["kgnb"]
            if fig is None:
                st.error(info.get("error"))
            else:
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**支撑原文引语**")
                for qte in info.get("quotes", []):
                    st.markdown(f"- `{qte['relation']}` → {qte['other'].split(':')[-1]}  \n"
                                f"  <span class='zz-tool'>“{qte['quote']}”</span>",
                                unsafe_allow_html=True)

    with tabs[2]:
        st.caption("同一行里同时出现蓝（升高截留）和红（降低截留）= "
                   "该因素在不同膜上效应反号，这是引擎3 的直接素材。")
        mc = st.slider("最少主张数", 1, 8, 2, key="kg_minclaims")
        if st.button("绘制热力图", key="kg_heat", use_container_width=True):
            st.session_state["kgheat"] = kgviz.contradiction_heatmap(mc)
        if st.session_state.get("kgheat"):
            fig, info = st.session_state["kgheat"]
            if fig is None:
                st.warning(info.get("error"))
            else:
                st.plotly_chart(fig, use_container_width=True)
                st.error("方向冲突的描述符：" +
                         "、".join(info["conflicting_descriptors"] or ["（无）"]))

    with tabs[4]:
        rows = db.rows_to_dicts(db.q(
            "SELECT * FROM contradictions ORDER BY id DESC LIMIT 30"))
        for r in rows:
            with st.expander(f"[{r['status']}] {r['topic']}"):
                st.markdown(f"**A 组**：{r['side_a']}\n\n**B 组**：{r['side_b']}")
                if r["note"]:
                    try:
                        n = db.jdict(r["note"])
                        st.markdown(f"**调和变量**：{n.get('candidate_reconciling_variable')}")
                        st.markdown(f"**调和假设**：{n.get('reconciliation_hypothesis')}")
                        st.success(f"**可证伪预测**：{n.get('falsifiable_prediction')}")
                    except Exception:  # noqa: BLE001
                        st.write(r["note"])
        if st.button("🔎 重新探测矛盾", key="kg_detect", use_container_width=True):
            with st.spinner("研判中，约 5 分钟…"):
                from zhizhi.tools.lit_tools import lit_contradictions
                st.json(lit_contradictions(detect=True))

    with tabs[5]:
        e1, e2 = st.columns(2)
        mind2 = e1.slider("最小连接数（静态图）", 1, 8, 4, key="kg_expdeg")
        if e2.button("导出 PNG + SVG", key="kg_exp_img", use_container_width=True):
            with st.spinner("绘制中…"):
                st.session_state["kgimg"] = kgviz.export_static(min_degree=mind2)
        if st.session_state.get("kgimg"):
            show_fig(st.session_state["kgimg"], "知识图谱（静态，可直接进论文）")
        if st.button("导出 GraphML（Gephi / Cytoscape）", key="kg_export",
                     use_container_width=True):
            from zhizhi.tools.lit_tools import lit_kg_export
            st.success(lit_kg_export())


# ============================ 页面：任务监视器 ============================
def page_tasks() -> None:
    from zhizhi.lit import worker
    st.markdown("### ⚙️ 任务监视器")
    st_ = worker.status()
    c = st.columns(6)
    c[0].metric("文献 worker", st_["worker"])
    c[1].metric("文献线程", f"{st_['threads_alive']}/{st_['n_workers_configured']}")
    agent_running = db.q1(
        "SELECT COUNT(*) c FROM tasks WHERE kind='agent_run' AND state IN ('queued','running')")
    c[2].metric("并行 Agent", f"{agent_running['c']}/{CFG.get('agents.max_workers', 4)}")
    for i, (k, v) in enumerate(list(st_["queue"].items())[:3]):
        c[3 + i].metric(k, v)
    st.caption("下面四个控制按钮只控制文献摄取；Agent 任务彼此独立并在后台线程池运行。")
    b = st.columns(4)
    if b[0].button("▶ 开始", key="task_start", use_container_width=True):
        worker.control("start")
        st.rerun()
    if b[1].button("⏸ 暂停", key="task_pause", use_container_width=True):
        worker.control("pause")
        st.rerun()
    if b[2].button("⏹ 停止", key="task_stop", use_container_width=True):
        worker.control("stop")
        st.rerun()
    if b[3].button("🔄 刷新", key="task_refresh", use_container_width=True):
        st.rerun()

    st.markdown("#### ⏱ 自动读取任务")
    st.caption("自动读取调度与文献摄取 worker 相互独立，可在这里单独暂停、开始或删除。")
    _lit_schedule_control_panel("task_monitor_schedule")

    t = st.tabs(["任务队列", "Token 消耗", "工具调用审计", "产出文件"])
    with t[0]:
        show_deleted = st.toggle("显示已删除任务", value=False, key="task_show_deleted")
        where = "" if show_deleted else "WHERE state!='deleted'"
        rows = db.rows_to_dicts(db.q(
            "SELECT id,kind,ref,label,state,progress,message,pinned,"
            "datetime(updated_at,'unixepoch','localtime') updated "
            f"FROM tasks {where} ORDER BY updated_at DESC LIMIT 200"))
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True,
                         hide_index=True, height=420)
    with t[1]:
        u = db.rows_to_dicts(db.q(
            "SELECT agent, model, kind, SUM(prompt_tokens) pt, "
            "SUM(completion_tokens) ct, COUNT(*) n FROM llm_usage "
            "GROUP BY agent, model, kind ORDER BY pt DESC"))
        if u:
            st.dataframe(pd.DataFrame(u), use_container_width=True, hide_index=True)
    with t[2]:
        a = db.rows_to_dicts(db.q(
            "SELECT datetime(created_at,'unixepoch','localtime') t, agent, tool, ok, "
            "round(ms) ms, substr(args,1,120) args FROM audit ORDER BY id DESC LIMIT 100"))
        if a:
            st.dataframe(pd.DataFrame(a), use_container_width=True,
                         hide_index=True, height=420)
    with t[3]:
        from zhizhi.ml import plots
        figs = plots.list_figures()
        st.caption(f"图片 {len(figs)} 张（store/figures/）")
        for f in figs[:8]:
            st.image(f, caption=Path(f).name, use_container_width=True)
        if st.button("📄 导出发现报告", key="sb_report", use_container_width=True):
            from zhizhi.tools.meta_tools import export_report
            st.success(export_report()["file"])


# ============================ 路由 ============================
ROUTES = {
    "🧭 总览": page_overview,
    "📚 博闻 · 文献层": page_bowen,
    "⚖️ 量衡 · 模型层": page_liangheng,
    "🔬 格物 · 发现层": page_gewu,
    "🧪 验真 · 验证层": page_yanzhen,
    "🗂 卡片审阅台": page_cards,
    "🕸 知识图谱": page_graph,
    "⚙️ 任务监视器": page_tasks,
}

ROUTES[topbar()]()
