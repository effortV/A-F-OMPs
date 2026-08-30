"""Streamlit Community Cloud entrypoint for the standalone OMPs project.

Two deployment modes are supported:

* standalone: map Streamlit Secrets to environment variables and run OMPs in
  the Community Cloud container (useful for demos without persistent data);
* server bridge: securely provision the model settings to the dedicated OMPs
  server and embed the server-hosted UI.  In this mode SQLite, PDFs, models,
  workers and all generated artifacts stay on persistent server storage.

Nothing in this module imports or connects to the separate NF project.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from collections.abc import Mapping
from urllib.parse import urlparse


_SECRET_ENV_NAMES = (
    "SILICONFLOW_API_KEY",
    "SILICONFLOW_BASE_URL",
    "LLM_MODEL",
    "LIT_PREPROCESS_MODEL",
    "EMBED_MODEL",
    "LLM_FALLBACK_MODELS",
    "OMPS_SERVER_APP_URL",
    "OMPS_BACKEND_URL",
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


_load_streamlit_secrets()
os.environ.setdefault("STREAMLIT_CLOUD_MODE", "true")

_ROOT = Path(__file__).resolve().parent


def _clean_https_url(name: str) -> str:
    """Return a normalized HTTPS URL or an empty string.

    Requiring HTTPS prevents model credentials and scientific data from being
    sent over a plaintext server link.
    """
    raw = os.environ.get(name, "").strip().rstrip("/")
    if not raw:
        return ""
    parsed = urlparse(raw)
    if parsed.scheme != "https" or not parsed.netloc:
        return ""
    return raw


def _run_server_bridge(server_url: str, backend_url: str, token: str) -> None:
    import requests
    import streamlit as st
    import streamlit.components.v1 as components

    st.set_page_config(
        page_title="致知 ZHIZHI · OMPs",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    payload = {
        name: os.environ.get(name, "")
        for name in (
            "SILICONFLOW_API_KEY",
            "SILICONFLOW_BASE_URL",
            "LLM_MODEL",
            "LIT_PREPROCESS_MODEL",
            "EMBED_MODEL",
            "LLM_FALLBACK_MODELS",
        )
        if os.environ.get(name, "")
    }

    try:
        response = requests.post(
            f"{backend_url}/config",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        health = requests.get(f"{backend_url}/health", timeout=15)
        health.raise_for_status()
    except requests.RequestException as exc:
        st.error("OMPs 服务器暂时无法连接。请稍后重试或检查服务器运行状态。")
        st.caption(f"连接目标：{urlparse(backend_url).netloc}；错误类型：{type(exc).__name__}")
        if st.button("重新连接", use_container_width=True):
            st.rerun()
        return

    # ``?embed=true`` hides the inner Streamlit toolbar while preserving every
    # OMPs page and control.  Scrolling remains enabled for long research runs.
    components.iframe(f"{server_url}/?embed=true", height=5600, scrolling=True)


_SERVER_URL = _clean_https_url("OMPS_SERVER_APP_URL")
_BACKEND_URL = _clean_https_url("OMPS_BACKEND_URL")
_API_TOKEN = os.environ.get("OMPS_API_TOKEN", "").strip()

if _SERVER_URL or _BACKEND_URL or _API_TOKEN:
    if not (_SERVER_URL and _BACKEND_URL and _API_TOKEN):
        import streamlit as st

        st.set_page_config(page_title="致知 ZHIZHI · OMPs", page_icon="🧭", layout="wide")
        st.error(
            "服务器连接配置不完整：需要同时填写 OMPS_SERVER_APP_URL、"
            "OMPS_BACKEND_URL 和 OMPS_API_TOKEN。"
        )
    else:
        _run_server_bridge(_SERVER_URL, _BACKEND_URL, _API_TOKEN)
else:
    runpy.run_path(str(_ROOT / "zhizhi" / "ui" / "app.py"), run_name="__main__")
