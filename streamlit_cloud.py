"""Streamlit Community Cloud entrypoint for the standalone OMPs project.

In production the complete Streamlit UI runs in Community Cloud while every
database, file, Agent, model and background-worker operation is executed by the
dedicated OMPs server through a host-key-pinned SSH tunnel. No NF setting,
port, database or key is imported or reused.
"""
from __future__ import annotations

import os
import runpy
from collections.abc import Mapping
from pathlib import Path


_SECRET_ENV_NAMES = (
    "SILICONFLOW_API_KEY",
    "SILICONFLOW_BASE_URL",
    "LLM_MODEL",
    "LIT_PREPROCESS_MODEL",
    "EMBED_MODEL",
    "LLM_FALLBACK_MODELS",
    "OMPS_SSH_HOST",
    "OMPS_SSH_PORT",
    "OMPS_SSH_USERNAME",
    "OMPS_SSH_HOST_KEY_SHA256",
    "OMPS_SSH_PRIVATE_KEY",
    "OMPS_REMOTE_API_HOST",
    "OMPS_REMOTE_API_PORT",
    "OMPS_API_TOKEN",
)
_SSH_REQUIRED = (
    "OMPS_SSH_HOST",
    "OMPS_SSH_PORT",
    "OMPS_SSH_USERNAME",
    "OMPS_SSH_HOST_KEY_SHA256",
    "OMPS_SSH_PRIVATE_KEY",
    "OMPS_REMOTE_API_HOST",
    "OMPS_REMOTE_API_PORT",
    "OMPS_API_TOKEN",
)


def _load_streamlit_secrets() -> None:
    try:
        import streamlit as st

        secrets: Mapping[str, object] = st.secrets
    except Exception:
        return
    for name in _SECRET_ENV_NAMES:
        if name in os.environ or name not in secrets:
            continue
        value = secrets[name]
        if isinstance(value, (str, int, float, bool)):
            os.environ[name] = str(value)


def _show_connection_error(message: str, detail: str = "") -> None:
    import streamlit as st

    st.set_page_config(
        page_title="致知 ZHIZHI · OMPs",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.error(message)
    if detail:
        st.caption(detail)
    if st.button("重新连接", use_container_width=True):
        st.rerun()


_load_streamlit_secrets()
os.environ.setdefault("STREAMLIT_CLOUD_MODE", "true")
_ROOT = Path(__file__).resolve().parent
_present = [name for name in _SSH_REQUIRED if os.environ.get(name, "").strip()]
_missing = [name for name in _SSH_REQUIRED if not os.environ.get(name, "").strip()]

if _present:
    if _missing:
        _show_connection_error(
            "OMPs SSH 连接配置不完整。",
            "缺少：" + "、".join(_missing),
        )
    else:
        os.environ["OMPS_REMOTE_MODE"] = "true"
        try:
            from zhizhi.core import remote

            remote.ensure_tunnel()
            remote.sync_model_config()
            health = remote.health()
            if not health.get("dataset") or not health.get("database"):
                raise remote.RemoteError("服务器数据集或数据库未挂载")
        except Exception as exc:  # noqa: BLE001 - show a safe connection summary
            _show_connection_error(
                "OMPs 服务器 SSH 连接失败。请检查专用 SSH Secrets 或服务器状态。",
                f"错误类型：{type(exc).__name__}；{str(exc)[:500]}",
            )
        else:
            runpy.run_path(str(_ROOT / "zhizhi" / "ui" / "app.py"), run_name="__main__")
elif (_ROOT / "dataset.xlsx").is_file():
    runpy.run_path(str(_ROOT / "zhizhi" / "ui" / "app.py"), run_name="__main__")
else:
    _show_connection_error(
        "OMPs 云端入口已启动，正在等待独立服务器 SSH 配置。",
        "请填写 OMPs 专用的 OMPS_SSH_*、OMPS_REMOTE_API_* 和 OMPS_API_TOKEN；"
        "数据集与文献库不会上传到 GitHub。",
    )
