from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote

import httpx
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/") + "/api"
if os.getenv("STREAMLIT_PAGE_CONFIGURED", "").casefold() not in {"1", "true", "yes", "on"}:
    st.set_page_config(page_title="NF-Atlas 纳滤智能体", page_icon="🧪", layout="wide")
CLOUD_MODE = os.getenv("STREAMLIT_CLOUD_MODE", "").casefold() in {"1", "true", "yes", "on"}
REMOTE_BACKEND = os.getenv("STREAMLIT_REMOTE_BACKEND", "").casefold() in {"1", "true", "yes", "on"}


def _api_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    token = os.getenv("NF_API_ACCESS_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    client_id = os.getenv("CF_ACCESS_CLIENT_ID", "").strip()
    client_secret = os.getenv("CF_ACCESS_CLIENT_SECRET", "").strip()
    if client_id and client_secret:
        headers["CF-Access-Client-Id"] = client_id
        headers["CF-Access-Client-Secret"] = client_secret
    return headers


def api(method: str, path: str, **kwargs: Any) -> Any:
    try:
        headers = _api_headers()
        headers.update(kwargs.pop("headers", {}) or {})
        response = httpx.request(method, API_URL + path, timeout=600, headers=headers, **kwargs)
        response.raise_for_status()
        if response.status_code == 204:
            return None
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except Exception:
            detail = exc.response.text
        raise RuntimeError(str(detail)) from exc
    except httpx.HTTPError as exc:
        raise RuntimeError(f"无法连接后端 {API_URL}：{exc}") from exc


def api_bytes(path: str) -> bytes:
    try:
        response = httpx.get(API_URL + path, timeout=600, headers=_api_headers())
        response.raise_for_status()
        return response.content
    except httpx.HTTPError as exc:
        raise RuntimeError(f"训练数据导出失败：{exc}") from exc


@st.cache_data(ttl=5, show_spinner=False)
def load_conversations() -> list[dict[str, Any]]:
    return api("GET", "/conversations")


def ensure_conversation() -> str:
    if st.session_state.get("conversation_id"):
        return st.session_state.conversation_id
    conversations = load_conversations()
    if conversations:
        st.session_state.conversation_id = conversations[0]["id"]
    else:
        created = api("POST", "/conversations", json={"title": "新对话"})
        st.session_state.conversation_id = created["id"]
        load_conversations.clear()
    return st.session_state.conversation_id


def render_sidebar() -> str:
    with st.sidebar:
        st.title("🧪 NF-Atlas")
        st.caption("纳滤领域知识 + LLM 垂直 Agent")
        if CLOUD_MODE and REMOTE_BACKEND:
            st.caption("☁️ 云前端 · 数据、文献和后台任务保存在服务器主库；前端休眠不会丢失")
        if st.session_state.pop("conversation_deleted", None):
            st.success("对话已删除；知识库和已入库文献仍然保留。")
        conversations = load_conversations()
        if st.button("＋ 新建对话", use_container_width=True):
            created = api("POST", "/conversations", json={"title": "新对话"})
            st.session_state.conversation_id = created["id"]
            for key in ("chat_discovery", "manual_discovery", "active_job"):
                st.session_state.pop(key, None)
            load_conversations.clear()
            st.rerun()
        current_id = ensure_conversation()
        labels = {item["id"]: item["title"] for item in conversations}
        labels.setdefault(current_id, "新对话")
        selected = st.selectbox(
            "历史对话",
            options=list(labels),
            index=list(labels).index(current_id),
            format_func=lambda item: labels[item],
        )
        if selected != current_id:
            st.session_state.conversation_id = selected
            st.session_state.pop("chat_discovery", None)
            st.rerun()
        with st.expander("管理当前对话"):
            st.caption("删除对话会移除消息和对应训练轨迹；已入库文献、知识库和知识成果继续保留。")
            confirmed = st.checkbox(
                f"确认删除“{labels.get(selected, '当前对话')}”",
                key=f"confirm-delete-{selected}",
            )
            if st.button(
                "删除当前对话",
                type="secondary",
                use_container_width=True,
                disabled=not confirmed,
                key=f"delete-conversation-{selected}",
            ):
                try:
                    api("DELETE", f"/conversations/{selected}")
                    st.session_state.pop("conversation_id", None)
                    for key in ("chat_discovery", "manual_discovery", "active_job"):
                        st.session_state.pop(key, None)
                    load_conversations.clear()
                    st.session_state.conversation_deleted = True
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
        st.divider()
        st.caption("对话、证据、知识库和索引版本均持久化；重启后可继续。")
        return selected


def render_status() -> None:
    try:
        health = api("GET", "/health")
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    cols = st.columns(5)
    cols[0].metric("后端", "在线")
    chat_model = str(health.get("chat_llm_model") or "deepseek-ai/DeepSeek-V4-Pro").split("/")[-1]
    cols[1].metric("对话模型", chat_model if health["llm_configured"] else "未配置")
    cols[2].metric("图谱", "Neo4j" if health["neo4j_configured"] else "SQL 回退")
    cols[3].metric("数据库", health["database"])
    cols[4].metric("后台任务", health["queue_mode"])
    channels = [
        "OpenAlex✓" if health.get("openalex_configured") else "OpenAlex未配置",
        "Unpaywall✓" if health.get("unpaywall_configured") else "Unpaywall未配置",
        "Elsevier已填Key" if health.get("elsevier_configured") else "Elsevier未配置",
    ]
    st.caption("文献渠道：" + " · ".join(channels) + "。已填 Key 不等于机构已授权，实际状态以每次检索诊断为准。")


def render_current_document_progress(job: dict[str, Any]) -> None:
    counts = job.get("counts") or {}
    current = counts.get("current_document") or {}
    execution = counts.get("execution_state") or "legacy"
    queue_name = counts.get("worker_queue") or "—"
    state_labels = {
        "running": "正在执行",
        "queued": "排队中",
        "pausing": "正在安全暂停",
        "paused": "已暂停",
        "cancelling": "正在安全取消",
        "cancel_requested": "等待取消",
        "cancelled": "已取消",
        "completed": "已完成",
        "failed": "失败",
        "legacy": "旧任务/等待恢复",
    }
    st.caption(f"执行状态：{state_labels.get(execution, execution)} · 队列：{queue_name}")
    if not current:
        if job.get("status") not in {"completed", "failed", "awaiting_selection"}:
            st.info("任务尚未进入具体文献，或这是升级前创建的旧任务；Worker 恢复后会开始记录单篇进度。")
        return
    title = current.get("title") or "题名暂缺"
    position = int(current.get("position") or 0)
    total = int(current.get("total") or counts.get("selected") or 0)
    st.markdown(f"**当前文献：{title}**")
    details = st.columns([1, 1.5, 1, 1, 1])
    details[0].metric("批次位置", f"{position}/{total}" if total else "—")
    details[1].metric("内容模式", current.get("content_mode") or "判断中")
    details[2].metric("解析切片", int(current.get("chunks") or 0))
    details[3].metric("结构化事实", int(current.get("facts") or 0))
    details[4].metric("单篇状态", current.get("state") or "processing")
    document_progress = float(current.get("document_progress") or 0)
    st.progress(
        max(0.0, min(1.0, document_progress)),
        text=f"单篇进度 {document_progress:.0%} · {current.get('stage') or job.get('stage')}",
    )
    st.caption(
        f"DOI：{current.get('doi') or '—'} · 全文来源：{current.get('fulltext_source') or '尚未确定'} · "
        f"源文件：{'已保存' if current.get('has_saved_file') else '尚未保存/题名摘要模式'}"
    )


def render_job(
    job_id: str,
    *,
    key_prefix: str = "job",
    embedded: bool = False,
    job_data: dict[str, Any] | None = None,
    show_logs: bool = True,
) -> None:
    if job_data is None:
        try:
            job = api("GET", f"/jobs/{job_id}")
        except RuntimeError as exc:
            st.error(str(exc))
            return
    else:
        job = job_data
    st.progress(float(job["progress"]), text=f"{job['stage']} · {job['status']}")
    counts = job.get("counts") or {}
    render_current_document_progress(job)
    cols = st.columns(6)
    for col, (key, label) in zip(
        cols,
        [
            ("selected", "选择"),
            ("downloaded", "全文"),
            ("metadata_only", "题名摘要"),
            ("parsed", "解析"),
            ("indexed", "入库"),
            ("failed", "失败"),
        ],
        strict=True,
    ):
        col.metric(label, counts.get(key, 0))
    def render_logs() -> None:
        for entry in job.get("log", [])[-100:]:
            st.write(f"- `{entry.get('stage')}` {entry.get('message')}")
        if job.get("error_message"):
            st.error(job["error_message"])

    if not show_logs:
        pass
    elif embedded:
        st.markdown("**任务日志**")
        render_logs()
    else:
        with st.expander("任务日志", expanded=job["status"] == "failed"):
            render_logs()
    if job["status"] not in {"completed", "failed"}:
        if st.button("刷新任务状态", key=f"{key_prefix}-refresh-{job_id}"):
            st.rerun()


def render_job_controls(job: dict[str, Any], *, key_prefix: str) -> None:
    """Render cooperative task controls without deleting imported knowledge."""

    job_id = job["id"]
    control_state = job.get("control_state") or "active"
    status = job.get("status")
    execution_state = str((job.get("counts") or {}).get("execution_state") or "legacy")
    controls = st.columns([1, 1, 1.3, 1.2, 4])
    if status not in {"completed", "failed", "awaiting_selection"}:
        startable = control_state in {"paused", "pause_requested"} or execution_state != "running"
        if startable:
            if controls[0].button(
                "开始/继续",
                key=f"{key_prefix}-start-{job_id}",
                type="primary" if control_state in {"paused", "pause_requested"} else "secondary",
                use_container_width=True,
            ):
                try:
                    result = api("POST", f"/jobs/{job_id}/start")
                    st.session_state.processing_notice = result.get("message") or "任务已开始"
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
        else:
            if controls[0].button("暂停", key=f"{key_prefix}-pause-{job_id}", use_container_width=True):
                try:
                    result = api("POST", f"/jobs/{job_id}/pause")
                    st.session_state.processing_notice = result.get("message") or "暂停请求已提交"
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
    else:
        controls[0].caption("任务已结束")
    controls[1].button("刷新", key=f"{key_prefix}-refresh-{job_id}", use_container_width=True)
    confirmed = controls[2].checkbox("确认删除", key=f"{key_prefix}-confirm-delete-{job_id}")
    if controls[3].button(
        "删除任务",
        key=f"{key_prefix}-delete-{job_id}",
        disabled=not confirmed,
        use_container_width=True,
    ):
        try:
            result = api("DELETE", f"/jobs/{job_id}")
            st.session_state.processing_notice = (
                result.get("message") or "任务已删除；已入库文献和知识继续保留"
            )
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))
    controls[4].caption("暂停/删除在当前单篇安全点生效；已经入库的文献、切片、事实和图谱不会删除。")


@st.fragment(run_every=5)
def render_processing_center(knowledge_base_id: str) -> None:
    heading, refresh = st.columns([5, 1])
    heading.subheader("后台文献处理中心")
    if refresh.button("立即刷新", key=f"processing-refresh-{knowledge_base_id}", use_container_width=True):
        st.rerun(scope="fragment")
    notice = st.session_state.pop("processing_notice", None)
    if notice:
        st.success(notice)
    try:
        jobs = api("GET", f"/knowledge-bases/{knowledge_base_id}/jobs?limit=100")
    except RuntimeError as exc:
        st.error(str(exc))
        return
    execution_jobs = [
        item
        for item in jobs
        if item.get("status") not in {"completed", "failed", "awaiting_selection"}
    ]
    running = [item for item in execution_jobs if (item.get("counts") or {}).get("execution_state") == "running"]
    queued = [item for item in execution_jobs if item not in running]
    awaiting = [item for item in jobs if item.get("status") == "awaiting_selection"]
    recent_completed = [item for item in jobs if item.get("status") in {"completed", "failed"}][:10]
    metrics = st.columns(5)
    metrics[0].metric("正在处理任务", len(running))
    metrics[1].metric("排队/等待恢复", len(queued))
    metrics[2].metric("等待你选文献", len(awaiting))
    metrics[3].metric("队列待处理文献", sum(int((item.get("counts") or {}).get("selected") or 0) for item in queued))
    metrics[4].metric("最近完成/失败", len(recent_completed))
    st.caption("该区域每 5 秒自动刷新。整批进度与当前单篇进度分开显示；上传、公开网址和单篇修复进入优先队列。")

    if not running:
        st.info("目前没有 Worker 正在执行具体任务；如存在排队任务，它们会按优先队列和普通队列依次启动。")
    for index, job in enumerate(running, 1):
        with st.container(border=True):
            st.markdown(f"#### 正在执行 {index} · {job.get('query') or '文献处理任务'}")
            render_job(
                job["id"],
                key_prefix=f"processing-running-{index}",
                embedded=True,
                job_data=job,
                show_logs=False,
            )
            render_job_controls(job, key_prefix=f"processing-running-controls-{index}")

    if queued:
        with st.expander(f"排队或等待恢复的任务（{len(queued)}）", expanded=not running):
            for index, job in enumerate(queued[:50], 1):
                counts = job.get("counts") or {}
                queue_name = counts.get("worker_queue") or "等待 Worker 识别"
                selected = int(counts.get("selected") or job.get("requested_count") or 0)
                with st.container(border=True):
                    st.write(
                        f"**{job.get('query') or '文献任务'}** · {job.get('stage')} · "
                        f"{selected} 篇 · `{queue_name}`"
                    )
                    render_job_controls(job, key_prefix=f"processing-queued-{index}")
    if awaiting:
        with st.expander(f"等待你确认候选（{len(awaiting)}）"):
            for index, job in enumerate(awaiting[:30], 1):
                counts = job.get("counts") or {}
                with st.container(border=True):
                    st.write(
                        f"**{job.get('query')}** · 新候选 {counts.get('new', 0)} 篇 · "
                        f"已有 {counts.get('existing', 0)} 篇"
                    )
                    render_job_controls(job, key_prefix=f"processing-awaiting-{index}")
    if recent_completed:
        with st.expander("最近完成或失败的任务"):
            for job in recent_completed:
                icon = "✅" if job.get("status") == "completed" else "❌"
                st.write(f"- {icon} **{job.get('query')}** · {job.get('stage')}")


def render_connector_status(connector_status: dict[str, Any]) -> None:
    for name, connector in connector_status.items():
        state = connector.get("status")
        errors = connector.get("errors") or []
        http_code = errors[0].get("status_code") if errors else None
        service_code = errors[0].get("service_code") if errors else None
        if state not in {"error", "degraded", "not_configured"}:
            continue
        if name == "elsevier" and service_code == "AUTHORIZATION_ERROR":
            st.warning(
                f"Elsevier 返回 HTTP {http_code}（AUTHORIZATION_ERROR）：新 Key 已发送，但该应用/当前出口网络"
                "尚无 ScienceDirect Search API 权限。请从学校订阅网络重试，或联系 Elsevier/API 管理员开通。"
            )
        elif name == "elsevier" and service_code == "AUTHENTICATION_ERROR":
            st.warning(
                f"Elsevier 返回 HTTP {http_code}（AUTHENTICATION_ERROR）：请求者配置不足。"
                "请检查 Developer Portal 中的 Key/API 产品，并使用学校订阅 IP 或官方 Insttoken。"
            )
        elif name == "elsevier" and http_code == 401:
            st.warning(
                "Elsevier 返回 HTTP 401：Key 无效、已停用，或尚未启用当前 API。"
                "请在 Developer Portal 检查应用；系统不会使用或保存学校账号密码。"
            )
        elif name == "elsevier" and http_code == 403:
            st.warning(
                "Elsevier 返回 HTTP 403：API Key 已被识别，但当前服务器公网 IP/Insttoken "
                "没有相应订阅全文权限。请从学校订阅网络运行，或向 Elsevier 申请官方 ELSEVIER_INSTTOKEN。"
            )
        elif http_code == 429:
            st.warning(f"{name} 返回 HTTP 429。系统已按该渠道限速和自动重试，本次结果可能仍不完整；稍后重试即可。")
        elif state == "error":
            suffix = f"（HTTP {http_code}）" if http_code else ""
            st.error(f"{name} API 调用失败{suffix}；其他正常渠道仍会保留结果。")
        elif state == "degraded":
            suffix = f"（HTTP {http_code}）" if http_code else ""
            st.warning(f"{name} 部分请求失败{suffix}，本次结果可能不完整。")
        elif state == "not_configured" and name in {"openalex", "unpaywall", "elsevier"}:
            st.info(f"{name} 未配置，本次未使用该渠道。")


def render_discovery_results(state_key: str, *, default_count: int) -> None:
    discovery = st.session_state.get(state_key)
    if not discovery:
        return
    cols = st.columns(3)
    cols[0].metric("检索候选", discovery["total_found"])
    cols[1].metric("知识库已有", discovery["existing_count"])
    cols[2].metric("可新增", discovery["new_count"])
    render_connector_status(discovery.get("connector_status") or {})
    with st.expander("查看 DeepSeek 扩展词表"):
        st.json(discovery["expanded_terms"], expanded=False)

    new_candidates = [item for item in discovery["candidates"] if not item["already_exists"]][:200]
    existing = [item for item in discovery["candidates"] if item["already_exists"]]
    if existing:
        with st.expander(f"已有文献推荐（{len(existing)}，不会重复下载、解析或计费）"):
            for item in existing[:50]:
                st.write(f"- {item['title']} · DOI: {item.get('doi') or '—'} · {item.get('duplicate_reason')}")
    option_map = {
        item["candidate_id"]: (
            f"{item.get('publication_year') or '—'} | {item['title']} | "
            f"DOI {item.get('doi') or '—'} | 相关性 {item['relevance_score']:.2f} | {item['source']}"
        )
        for item in new_candidates
    }
    default_ids = list(option_map)[: min(default_count, len(option_map))]
    selected_ids = st.multiselect(
        "选择本次新增文献（0～200 篇）",
        options=list(option_map),
        default=default_ids,
        format_func=lambda candidate_id: option_map[candidate_id],
        max_selections=200,
        key=f"selection-{state_key}-{discovery['job_id']}",
    )
    selected_ids = list(dict.fromkeys(selected_ids))
    st.caption(f"本次新增 {len(selected_ids)} 篇；数量只计算四级查重后的新文献。无合法全文时将用题名+摘要入库。")
    label = "确认并开始入库" if selected_ids else "确认新增 0 篇"
    if st.button(label, type="primary", key=f"confirm-{state_key}-{discovery['job_id']}"):
        try:
            job = api(
                "POST",
                f"/jobs/{discovery['job_id']}/selection",
                json={"candidate_ids": selected_ids, "requested_count": len(selected_ids)},
            )
            st.session_state.active_job = job["id"]
            st.session_state.pop(state_key, None)
            st.success("已提交后台处理。" if selected_ids else "不新增，继续使用现有知识库。")
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))


def render_messages(conversation_id: str) -> None:
    try:
        messages = api("GET", f"/conversations/{conversation_id}/messages")
    except RuntimeError as exc:
        st.error(str(exc))
        return
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("model_name"):
                model_call = next(
                    (call for call in message.get("tool_calls", []) if call.get("tool") == "conversation_llm"),
                    {},
                )
                thinking_label = (
                    "深度思考：开启" if model_call.get("deep_thinking") else "深度思考：关闭"
                    if model_call
                    else ""
                )
                st.caption(" · ".join(item for item in (message["model_name"], thinking_label) if item))
            if message.get("evidence"):
                with st.expander(f"查看 {len(message['evidence'])} 条检索证据"):
                    for item in message["evidence"]:
                        source = item.get("doi") or item.get("title") or "来源未知"
                        page = item.get("page") or item.get("page_start") or "未解析"
                        st.markdown(f"**{source} · p.{page}**")
                        st.caption(item.get("quote") or item.get("source_sentence") or "")
            if message.get("tool_calls"):
                with st.expander("Agent 检索、路由与证据审计轨迹"):
                    for call in message["tool_calls"]:
                        tool = call.get("tool", "unknown")
                        details = {key: value for key, value in call.items() if key != "tool"}
                        st.markdown(f"- **{tool}**")
                        if details:
                            st.json(details, expanded=False)
            if message["role"] == "assistant":
                with st.expander("评价或修订此回答"):
                    with st.form(f"feedback-{message['id']}"):
                        rating = st.select_slider("评分", options=[1, 2, 3, 4, 5], value=4)
                        revision = st.text_area("人工修订（可选）")
                        approved = st.checkbox("审核后允许纳入训练候选集")
                        if st.form_submit_button("保存反馈"):
                            try:
                                api(
                                    "POST",
                                    "/feedback",
                                    json={
                                        "message_id": message["id"],
                                        "rating": rating,
                                        "human_revision": revision or None,
                                        "approved_for_training": approved,
                                    },
                                )
                                st.success("反馈已保存")
                            except RuntimeError as exc:
                                st.error(str(exc))


def render_chat_module(conversation: dict[str, Any]) -> None:
    conversation_id = conversation["id"]
    st.subheader("纳滤知识发现图谱 Agent")
    st.caption(
        "每次提问先扫描整个现有知识库（不是只看本轮新增文献），再用纳滤中英文词表联查 Neo4j、Chroma 和全文关键词；"
        "DeepSeek 负责跨文献比较、因果分析与可验证假设。只有最终送入模型的证据会按相关性和独立文献覆盖度压缩。"
    )
    mode_col, thinking_col, literature_col, count_col = st.columns([1.3, 1.1, 2, 1])
    mode_label = mode_col.selectbox(
        "研究模式",
        ["深度科研（推荐）", "证据严格", "快速问答"],
        help="深度科研会扩大混合检索并进行知识缺口评估；证据严格会要求更多独立全文来源；快速问答减少检索数量。",
    )
    mode_map = {
        "深度科研（推荐）": "deep_research",
        "证据严格": "evidence_strict",
        "快速问答": "rapid",
    }
    deep_thinking = thinking_col.toggle(
        "V4-Pro 深度思考",
        value=True,
        help="默认开启；关闭后仍使用 DeepSeek-V4-Pro，但不启用 enable_thinking，可降低等待时间和用量。",
    )
    proactive = literature_col.toggle("证据不足或你要求找文献时，主动检索新文献", value=True)
    desired = count_col.number_input("预选新文献", min_value=0, max_value=200, value=50, step=10)
    knowledge_discovery = st.toggle(
        "启用跨文献知识发现（规律、矛盾、机理假设和验证实验）",
        value=True,
        help="生成内容会保存为 AI综合/AI假设，不会冒充论文事实，并在本对话下方提供审核。",
    )
    if desired == 0:
        st.caption("选择 0：不发现或新增文献，只使用现有知识库问答。")
    if st.session_state.get("active_job"):
        with st.expander("当前入库任务", expanded=False):
            render_job(st.session_state.active_job, key_prefix="chat", embedded=True)
    if st.session_state.get("chat_discovery"):
        st.markdown("#### Agent 主动发现的候选")
        render_discovery_results("chat_discovery", default_count=int(st.session_state.get("chat_desired", desired)))
    st.divider()
    render_messages(conversation_id)
    question = st.chat_input("询问纳滤材料、性能、条件、机理、综述，或直接说‘帮我找相关文献’……")
    if question:
        with st.spinner("扫描全库，进行双语扩展、图谱/向量/全文联查与跨文献知识发现……"):
            try:
                result = api(
                    "POST",
                    "/chat",
                    json={
                        "conversation_id": conversation_id,
                        "question": question,
                        "proactive_literature": proactive,
                        "desired_new_count": int(desired),
                        "knowledge_discovery": knowledge_discovery,
                        "research_mode": mode_map[mode_label],
                        "deep_thinking": deep_thinking,
                    },
                )
                if result.get("discovery"):
                    st.session_state.chat_discovery = result["discovery"]
                    st.session_state.chat_desired = int(desired)
                if result.get("literature_error"):
                    st.session_state.chat_literature_error = result["literature_error"]
                load_conversations.clear()
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
    if st.session_state.pop("chat_literature_error", None):
        st.warning("主动检索暂时失败；回答已保存，可在文献模块重试。")

    st.divider()
    render_knowledge_insights(
        conversation["knowledge_base_id"],
        conversation_id=conversation_id,
        embedded=True,
    )


def render_knowledge_insights(
    knowledge_base_id: str,
    *,
    conversation_id: str | None = None,
    embedded: bool = False,
) -> None:
    st.subheader("本对话生成的知识、关系与可验证假设" if embedded else "跨文献知识成果")
    st.info("这些内容来自 Agent 对多篇知识库证据的综合。论文事实、AI归纳和AI假设始终分层保存；只有人工审阅或实验验证后才能升级状态。")
    status_filter = st.selectbox(
        "状态筛选",
        ["全部", "ai_synthesis", "ai_hypothesis", "reviewed", "validated", "rejected"],
        key=f"insight-status-{knowledge_base_id}",
    )
    path = f"/knowledge-bases/{knowledge_base_id}/insights?limit=300"
    if status_filter != "全部":
        path += "&status=" + quote(status_filter)
    if conversation_id:
        path += "&conversation_id=" + quote(conversation_id)
    try:
        insights = api("GET", path)
    except RuntimeError as exc:
        st.error(str(exc))
        return
    if not insights:
        st.info("当前对话还没有知识发现成果。可以提出材料比较、机理解释、性能优化、矛盾分析或新实验假设问题。")
        return
    summary = st.columns(4)
    summary[0].metric("总条目", len(insights))
    summary[1].metric("AI假设", sum(item["insight_type"] == "hypothesis" for item in insights))
    summary[2].metric("矛盾", sum(item["insight_type"] == "contradiction" for item in insights))
    summary[3].metric("已验证", sum(item["status"] == "validated" for item in insights))
    for item in insights:
        label = f"{item['insight_type']} | {item['title']} | {item['status']} | 置信度 {item['confidence']:.2f}"
        with st.expander(label):
            st.markdown(f"**推断/假设**：{item['claim']}")
            st.markdown(f"**推理链**：{item['rationale']}")
            if item.get("boundary_conditions"):
                st.markdown("**边界条件**：" + "；".join(item["boundary_conditions"]))
            if item.get("assumptions"):
                st.markdown("**关键假设**：" + "；".join(item["assumptions"]))
            references = item.get("evidence_refs") or []
            if references:
                st.markdown("**证据关系**")
                for reference in references:
                    stance = "反对/冲突" if reference.get("stance") == "contradicts" else "支持"
                    source = reference.get("doi") or reference.get("title") or "来源未知"
                    page = reference.get("page") if reference.get("page") is not None else "未解析"
                    st.markdown(f"- {stance} · {source} · p.{page} · {reference.get('evidence_mode') or 'unknown'}")
                    if reference.get("quote"):
                        st.caption(reference["quote"])
            if item.get("validation_plan"):
                st.markdown("**建议验证/证伪实验**")
                st.json(item["validation_plan"], expanded=False)
            st.caption(f"新颖性评分 {item['novelty_score']:.2f} · 生成模型 {item.get('model_name') or '—'}")
            with st.form(f"review-insight-{item['id']}"):
                review_status = st.selectbox(
                    "审核结果",
                    ["reviewed", "validated", "rejected"],
                    format_func=lambda value: {"reviewed": "人工已审阅", "validated": "实验/外部证据已验证", "rejected": "驳回"}[value],
                )
                note = st.text_area("审核依据；选择‘已验证’时必须填写实验或外部证据")
                if st.form_submit_button("保存审核"):
                    try:
                        api(
                            "POST",
                            f"/insights/{item['id']}/review",
                            json={"status": review_status, "review_note": note.strip() or None},
                        )
                        st.success("审核状态已同时写入 PostgreSQL 和 Neo4j。")
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(str(exc))


def render_manual_discovery(conversation_id: str) -> None:
    st.caption("输入主题后，DeepSeek 会补充中英文同义词、材料、工艺、分离体系和性能指标；题录 API 用于发现，公开网页/PDF/HTML 将直接读取。")
    with st.form("manual-discovery"):
        query = st.text_input("采集主题", placeholder="界面聚合 / 盐湖提锂 / 抗污染膜")
        cols = st.columns(4)
        desired = cols[0].number_input("希望新增", min_value=50, max_value=200, value=50, step=10)
        year_from = cols[1].number_input("起始年份（0=不限）", min_value=0, max_value=2100, value=0)
        year_to = cols[2].number_input("结束年份（0=不限）", min_value=0, max_value=2100, value=0)
        citation = cols[3].checkbox("扩展参考/被引", value=True)
        submitted = st.form_submit_button("扩词、检索并四级查重", type="primary")
    if submitted:
        if not query.strip():
            st.warning("请输入采集主题")
        else:
            with st.spinner("正在限速访问多源学术索引、检查公开网页并去重……"):
                try:
                    st.session_state.manual_discovery = api(
                        "POST",
                        "/discover",
                        json={
                            "conversation_id": conversation_id,
                            "query": query,
                            "limit": min(500, max(100, int(desired) * 3)),
                            "year_from": year_from or None,
                            "year_to": year_to or None,
                            "include_citation_expansion": citation,
                        },
                    )
                    st.session_state.manual_desired = int(desired)
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
    render_discovery_results("manual_discovery", default_count=int(st.session_state.get("manual_desired", 50)))


def render_automations(conversation: dict[str, Any]) -> None:
    st.caption("每轮自动扩词、检索、查重并处理 50～200 篇；任务通过 Redis/RQ 定时运行，关闭页面也不会停止。停止会在当前单篇安全点结束；删除任务不会删除已经入库的文献和知识。")
    with st.form("automation-create"):
        query = st.text_input("持续监测主题", placeholder="thin-film composite nanofiltration antifouling")
        cols = st.columns(3)
        batch_size = cols[0].number_input("每轮新增上限", min_value=50, max_value=200, value=50, step=10)
        interval = cols[1].number_input("轮询间隔（分钟）", min_value=5, max_value=10080, value=60, step=5)
        max_total = cols[2].number_input("总量上限（0=一直运行）", min_value=0, max_value=100000, value=0, step=50)
        create = st.form_submit_button("启动持续自动采集", type="primary")
    if create:
        if not query.strip():
            st.warning("请输入持续监测主题")
        else:
            try:
                api(
                    "POST",
                    "/automations",
                    json={
                        "conversation_id": conversation["id"],
                        "query": query,
                        "batch_size": int(batch_size),
                        "interval_minutes": int(interval),
                        "max_total": int(max_total) or None,
                    },
                )
                st.success("自动采集已进入 Redis/RQ 队列")
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
    try:
        tasks = api("GET", f"/automations?knowledge_base_id={conversation['knowledge_base_id']}")
    except RuntimeError as exc:
        st.error(str(exc))
        return
    if not tasks:
        st.info("还没有持续自动采集任务。")
        return
    for task in tasks:
        with st.container(border=True):
            top = st.columns([4, 1, 1, 1])
            top[0].markdown(f"**{task['name']}**  \n`{task['query']}`")
            top[1].metric("状态", task["status"])
            top[2].metric("轮次", task["cycles"])
            top[3].metric("已入库", task["imported_total"])
            st.caption(
                f"每轮 {task['batch_size']} 篇 · 间隔 {task['interval_minutes']} 分钟 · "
                f"下次：{task.get('next_run_at') or '正在运行/等待调度'}"
            )
            b1, b2, b3, b4, _ = st.columns([1, 1, 1.3, 1.2, 4])
            if task["status"] in {"active", "running", "stopping"}:
                if b1.button("停止", key=f"stop-{task['id']}"):
                    try:
                        api("POST", f"/automations/{task['id']}/stop")
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(str(exc))
            else:
                if b1.button("重新启动", key=f"restart-{task['id']}"):
                    try:
                        api("POST", f"/automations/{task['id']}/restart")
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(str(exc))
            if b2.button("刷新", key=f"auto-refresh-{task['id']}"):
                st.rerun()
            confirmed = b3.checkbox("确认删除", key=f"auto-confirm-delete-{task['id']}")
            if b4.button(
                "删除任务",
                key=f"auto-delete-{task['id']}",
                disabled=not confirmed,
                use_container_width=True,
            ):
                try:
                    api("DELETE", f"/automations/{task['id']}")
                    st.success("持续采集任务已删除；已经入库的文献和知识继续保留。")
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
            if task.get("error_message"):
                st.warning(task["error_message"])
            if task.get("last_job_id"):
                with st.expander("最近一轮详情"):
                    render_job(task["last_job_id"], key_prefix=f"auto-{task['id']}", embedded=True)


def render_reader(conversation_id: str, knowledge_base_id: str) -> None:
    with st.expander("批量修复题名/摘要文献", expanded=False):
        cols = st.columns([2, 1, 3])
        retry_limit = cols[0].number_input(
            "本次最多重新检查",
            min_value=1,
            max_value=200,
            value=50,
            step=10,
            key=f"retry-limit-{knowledge_base_id}",
        )
        if cols[1].button("开始检查", key=f"retry-all-{knowledge_base_id}"):
            try:
                job = api(
                    "POST",
                    f"/knowledge-bases/{knowledge_base_id}/retry-metadata-only",
                    json={"max_documents": int(retry_limit)},
                )
                st.session_state.active_job = job["id"]
                st.success("已提交后台任务：将优先直接读取公开 PDF/HTML，再使用开放全文 API 作为补充。")
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
        st.caption("不会绕过登录、验证码或付费墙；成功后会在原记录上升级全文，不会重复建文献。")
    search = st.text_input("按题名筛选", placeholder="lithium / polyamide / antifouling")
    try:
        path = f"/knowledge-bases/{knowledge_base_id}/documents?limit=300"
        if search.strip():
            path += "&query=" + quote(search.strip())
        documents = api("GET", path)
    except RuntimeError as exc:
        st.error(str(exc))
        return
    if not documents:
        st.info("知识库还没有匹配文献。可先手动采集或上传合法全文。")
        return
    labels = {
        item["id"]: f"{item.get('publication_year') or '—'} | {item['title']} | {item['status']}"
        for item in documents
    }
    document_id = st.selectbox("选择文献", list(labels), format_func=lambda item: labels[item])
    try:
        detail = api("GET", f"/documents/{document_id}")
    except RuntimeError as exc:
        st.error(str(exc))
        return
    document = detail["document"]
    metadata_only = bool((document.get("metadata_json") or {}).get("metadata_only"))
    import_job_id = (document.get("metadata_json") or {}).get("import_job_id")
    processing = document.get("status") in {"selected", "downloading", "downloaded", "parsed", "extracted"}
    st.markdown(f"### {document['title']}")
    st.caption(
        f"DOI: {document.get('doi') or '—'} · 来源: {document.get('fulltext_source') or '—'} · "
        f"许可: {document.get('license') or '—'} · 模式: {'仅题名+摘要' if metadata_only else '全文/解析文本'}"
    )
    if processing and import_job_id:
        try:
            waiting_job = api("GET", f"/jobs/{import_job_id}")
            if waiting_job.get("status") == "queued":
                if document.get("object_key"):
                    st.info("全文文件已经保存，正在解析章节、表格、图注并抽取结构化事实。处理完成前原有切片仍可参与问答。")
                elif metadata_only and (document.get("metadata_json") or {}).get("upgrade_pending"):
                    st.info("正在检查 Europe PMC、Unpaywall、出版社公开入口和已有全文地址；当前仍保留题名/摘要数据。")
                else:
                    st.info("正在尝试获取合法公开全文；尚未取得全文时会保留题名和摘要，不会误报为 PDF 已保存。")
            else:
                st.info(
                    f"后台处理状态：{waiting_job.get('stage') or waiting_job.get('status')} · "
                    f"{float(waiting_job.get('progress') or 0):.0%}"
                )
            render_current_document_progress(waiting_job)
            if st.button("刷新处理状态", key=f"reader-refresh-{document_id}"):
                st.rerun()
        except RuntimeError:
            pass
    if metadata_only:
        st.warning("当前没有合法可用全文；以下内容和抽取结论仅来自题名/摘要，不能视为全文证据。")
        if processing:
            st.caption("该记录当前属于正在执行的任务，下面三个升级按钮暂时禁用，以避免两个 Worker 同时覆盖；完成或失败后会自动恢复。")
        reason = (document.get("metadata_json") or {}).get("fulltext_unavailable_reason")
        if reason:
            st.caption(f"上次失败原因：{reason}")
        if st.button(
            "自动查找并升级公开全文",
            key=f"retry-document-{document_id}",
            type="primary",
            disabled=processing,
        ):
            try:
                job = api("POST", f"/documents/{document_id}/retry-fulltext")
                st.session_state.active_job = job["id"]
                st.success("已开始检查 Europe PMC、Unpaywall、出版社公开入口和现有链接。")
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
        st.markdown("#### 立即补充全文")
        source_default = (
            document.get("fulltext_url")
            or document.get("landing_url")
            or (f"https://doi.org/{document['doi']}" if document.get("doi") else "")
        )
        public_col, upload_col = st.columns(2)
        with public_col.container(border=True):
            st.markdown("**公开网址 / DOI 导入**")
            public_source = st.text_input(
                "公开文章页、PDF 地址或 DOI",
                value=source_default,
                key=f"reader-public-source-{document_id}",
            )
            public_confirmed = st.checkbox(
                "确认该地址公开可访问",
                key=f"reader-public-confirm-{document_id}",
            )
            if st.button(
                "直接获取并升级",
                key=f"reader-public-import-{document_id}",
                disabled=processing or not public_confirmed or not public_source.strip(),
            ):
                try:
                    with st.spinner("正在验证公开地址并获取全文；完成后会进入优先解析队列……"):
                        result = api(
                            "POST",
                            f"/conversations/{conversation_id}/url-import",
                            json={
                                "source": public_source.strip(),
                                "title": document["title"],
                                "doi": document.get("doi"),
                                "authors": [item.get("name", "") for item in document.get("authors") or [] if item.get("name")],
                                "publication_year": document.get("publication_year"),
                                "public_access_confirmed": True,
                            },
                        )
                    st.session_state.active_job = result["job_id"]
                    st.success(result["message"])
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
        with upload_col.container(border=True):
            st.markdown("**合法全文上传**")
            replacement = st.file_uploader(
                "上传已有权限的 PDF/XML",
                type=["pdf", "xml"],
                key=f"reader-upload-{document_id}",
            )
            upload_confirmed = st.checkbox(
                "确认拥有处理和保存权限",
                key=f"reader-upload-confirm-{document_id}",
            )
            if st.button(
                "上传并升级原记录",
                key=f"reader-upload-submit-{document_id}",
                disabled=processing or not replacement or not upload_confirmed,
            ):
                try:
                    data = {
                        "title": document["title"],
                        "authors_json": json.dumps(
                            [item.get("name", "") for item in document.get("authors") or [] if item.get("name")],
                            ensure_ascii=False,
                        ),
                        "rights_confirmed": "true",
                    }
                    if document.get("doi"):
                        data["doi"] = document["doi"]
                    if document.get("publication_year"):
                        data["publication_year"] = str(document["publication_year"])
                    with st.spinner("正在保存合法全文并提交优先解析任务……"):
                        result = api(
                            "POST",
                            f"/conversations/{conversation_id}/upload",
                            files={"file": (replacement.name, replacement.getvalue(), replacement.type)},
                            data=data,
                        )
                    st.session_state.active_job = result["job_id"]
                    st.success(result["message"])
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
    if document.get("abstract"):
        with st.expander("摘要", expanded=True):
            st.write(document["abstract"])
    chunks, facts = detail["chunks"], detail["facts"]
    left, right = st.columns(2)
    with left:
        st.markdown(f"#### 解析切片（{len(chunks)}）")
        for chunk in chunks[:200]:
            with st.expander(f"{chunk.get('section') or chunk.get('block_kind')} · p.{chunk.get('page_start') or '—'}"):
                st.write(chunk["text"])
    with right:
        st.markdown(f"#### 结构化事实（{len(facts)}）")
        if facts:
            st.dataframe(
                [
                    {
                        "类型": item["fact_type"],
                        "主体": item["subject"],
                        "关系": item["predicate"],
                        "值": item.get("normalized_value") if item.get("normalized_value") is not None else item.get("object_text"),
                        "单位": item.get("normalized_unit") or item.get("unit"),
                        "页码": item.get("page"),
                        "置信度": item["confidence"],
                    }
                    for item in facts
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("该文献尚无结构化事实，仍可通过全文/摘要切片参与 RAG。")


def render_upload(conversation_id: str) -> None:
    st.warning("系统不会保存学校账号或密码，也不会模拟批量登录。请仅上传你有权用于本知识库的全文。")
    with st.form("upload-form"):
        file = st.file_uploader("PDF/XML", type=["pdf", "xml"])
        title = st.text_input("题名")
        doi = st.text_input("DOI（可选）")
        authors = st.text_input("作者（英文逗号分隔，可选）")
        year = st.number_input("年份（0=未知）", min_value=0, max_value=2100, value=0)
        confirmed = st.checkbox("我确认拥有处理和保存该文件的合法权限")
        submit = st.form_submit_button("上传并解析", type="primary")
    if submit:
        if not file or not title.strip() or not confirmed:
            st.error("请选择文件、填写题名并确认权限")
        else:
            try:
                data = {
                    "title": title,
                    "authors_json": json.dumps([item.strip() for item in authors.split(",") if item.strip()], ensure_ascii=False),
                    "rights_confirmed": "true",
                }
                if doi:
                    data["doi"] = doi
                if year:
                    data["publication_year"] = str(year)
                with st.spinner("正在保存文件并提交优先解析任务……"):
                    result = api(
                        "POST",
                        f"/conversations/{conversation_id}/upload",
                        files={"file": (file.name, file.getvalue(), file.type)},
                        data=data,
                    )
                st.session_state.active_job = result["job_id"]
                st.success(result["message"])
                st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))


def render_public_url_import(conversation_id: str) -> None:
    st.info("可粘贴公开文章页、直接 PDF 地址或 DOI。系统直接读取公开 PDF/XML/HTML，题录 API 只是辅助，不处理登录或付费内容。")
    with st.form("public-url-import"):
        source = st.text_input(
            "公开文章网址、PDF 地址或 DOI",
            placeholder="https://期刊官网/article/...  或  10.xxxx/xxxxx",
        )
        title = st.text_input("题名（可选，网页通常可自动识别）")
        doi = st.text_input("DOI（可选）", key="public-url-doi")
        authors = st.text_input("作者（英文逗号分隔，可选）", key="public-url-authors")
        year = st.number_input("年份（0=未知）", min_value=0, max_value=2100, value=0, key="public-url-year")
        confirmed = st.checkbox("我确认该网址无需登录即可公开访问，并允许系统按网站规则读取")
        submit = st.form_submit_button("获取并解析公开全文", type="primary")
    if submit:
        if not source.strip() or not confirmed:
            st.error("请输入公开网址/DOI并确认公开可访问")
            return
        try:
            with st.spinner("正在验证公开地址并获取正文；可能需要跟随 DOI 跳转……"):
                result = api(
                    "POST",
                    f"/conversations/{conversation_id}/url-import",
                    json={
                        "source": source.strip(),
                        "title": title.strip() or None,
                        "doi": doi.strip() or None,
                        "authors": [item.strip() for item in authors.split(",") if item.strip()],
                        "publication_year": int(year) or None,
                        "public_access_confirmed": True,
                    },
                )
            st.session_state.active_job = result["job_id"]
            st.success(result["message"])
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))


def render_training_data(conversation: dict[str, Any]) -> None:
    knowledge_base_id = conversation["knowledge_base_id"]
    conversation_id = conversation["id"]
    st.subheader("训练数据中心")
    st.caption(
        "系统持续保存 instruction、input、output、RAG 证据、Agent 工具轨迹、评分、人工修订和审核状态。"
        "这里只导出科研问答数据，不包含 API Key、学校账号或密码。"
    )
    try:
        stats = api("GET", f"/training/stats?knowledge_base_id={quote(knowledge_base_id)}")
    except RuntimeError as exc:
        st.error(str(exc))
        return
    metrics = st.columns(5)
    metrics[0].metric("全部轨迹", stats["total"])
    metrics[1].metric("已有评分", stats["rated"])
    metrics[2].metric("证据支撑", stats["evidence_backed"])
    metrics[3].metric("人工批准", stats["approved"])
    metrics[4].metric("高质量候选", stats["high_quality"])

    scope_col, approved_col, rating_col = st.columns([1.3, 1, 1])
    scope = scope_col.radio("范围", ["当前对话", "整个知识库"], horizontal=True)
    approved_only = approved_col.toggle("仅看人工批准", value=False)
    min_rating = rating_col.selectbox("最低评分", [0, 1, 2, 3, 4, 5], format_func=lambda value: "不限" if value == 0 else f"{value} 分")
    query = f"knowledge_base_id={quote(knowledge_base_id)}&approved_only={'true' if approved_only else 'false'}&min_rating={min_rating}"
    if scope == "当前对话":
        query += f"&conversation_id={quote(conversation_id)}"
    try:
        traces = api("GET", f"/training/traces?{query}&limit=300")
    except RuntimeError as exc:
        st.error(str(exc))
        return

    left, right = st.columns([2, 1])
    left.caption(f"当前筛选得到 {len(traces)} 条。质量分综合考虑独立文献数、全文证据、原文引句、工具轨迹、评分和人工修订。")
    export_query = query.replace("approved_only=false", "approved_only=true")
    try:
        export_data = api_bytes(f"/training/export?{export_query}")
        right.download_button(
            "导出已批准 JSONL",
            data=export_data,
            file_name="nf-atlas-training.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
            disabled=not export_data,
        )
    except RuntimeError as exc:
        right.error(str(exc))

    with st.expander("微调数据格式与使用说明"):
        st.code('{"instruction":"...","input":"...","output":"...","metadata":{"retrieval_evidence":[],"tool_trace":[]}}', language="json")
        st.write("建议只使用已人工批准、评分较高且有原文证据的数据。人工修订存在时，导出会自动以修订版本作为 output。")

    if not traces:
        st.info("当前筛选没有数据。完成一次 Agent 问答后会自动产生训练轨迹；在回答下方或这里评分、修订并批准。")
        return
    st.dataframe(
        [
            {
                "时间": item["created_at"],
                "问题": item["input_text"][:100],
                "评分": item.get("rating"),
                "证据": len(item.get("retrieval_evidence") or []),
                "工具": len(item.get("tool_trace") or []),
                "质量分": item["quality_score"],
                "质量等级": item["quality_label"],
                "已批准": item["approved_for_training"],
            }
            for item in traces
        ],
        use_container_width=True,
        hide_index=True,
    )
    for item in traces:
        title = f"{item['quality_label']} {item['quality_score']:.2f} | {item['input_text'][:90]}"
        with st.expander(title):
            st.markdown("**Instruction**")
            st.write(item["instruction"])
            st.markdown("**Input**")
            st.write(item["input_text"])
            st.markdown("**当前 Output**")
            st.write(item.get("human_revision") or item["output_text"])
            evidence_tab, tool_tab = st.tabs(
                [f"检索证据（{len(item.get('retrieval_evidence') or [])}）", f"工具轨迹（{len(item.get('tool_trace') or [])}）"]
            )
            with evidence_tab:
                st.json(item.get("retrieval_evidence") or [], expanded=False)
            with tool_tab:
                st.json(item.get("tool_trace") or [], expanded=False)
            with st.form(f"training-review-{item['id']}"):
                rating = st.select_slider("人工评分", options=[1, 2, 3, 4, 5], value=item.get("rating") or 4)
                revision = st.text_area(
                    "人工修订（留空则使用 Agent 原回答）",
                    value=item.get("human_revision") or "",
                    height=180,
                )
                approved = st.checkbox("批准进入后续 LoRA/蒸馏训练集", value=item["approved_for_training"])
                if st.form_submit_button("保存审核"):
                    try:
                        api(
                            "POST",
                            f"/training/traces/{item['id']}/review",
                            json={
                                "rating": rating,
                                "human_revision": revision.strip() or None,
                                "approved_for_training": approved,
                            },
                        )
                        st.success("审核已保存，质量评分和导出集合已更新。")
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(str(exc))


st.title("NF-Atlas 纳滤领域知识智能体")
st.caption("真实文献发现 → 四级查重 → 合法全文/题名摘要兜底 → MinerU/GROBID → Neo4j + Chroma/bge-m3 → 上下文 Agent")
conversation_id = render_sidebar()
render_status()
conversation = api("GET", f"/conversations/{conversation_id}")
agent_tab, literature_tab, training_tab = st.tabs(["🔬 知识发现图谱", "📚 文献采集与阅读", "🧰 训练数据中心"])
with agent_tab:
    render_chat_module(conversation)
with literature_tab:
    st.subheader("文献采集、自动化与阅读")
    render_processing_center(conversation["knowledge_base_id"])
    st.divider()
    manual_tab, automatic_tab, reader_tab, public_url_tab, upload_tab = st.tabs(
        ["手动采集 50–200 篇", "持续自动采集", "文献阅读", "公开网址/DOI导入", "合法全文上传"]
    )
    with manual_tab:
        render_manual_discovery(conversation_id)
    with automatic_tab:
        render_automations(conversation)
    with reader_tab:
        render_reader(conversation_id, conversation["knowledge_base_id"])
    with public_url_tab:
        render_public_url_import(conversation_id)
    with upload_tab:
        render_upload(conversation_id)
with training_tab:
    render_training_data(conversation)
