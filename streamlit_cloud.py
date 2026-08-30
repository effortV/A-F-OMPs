from __future__ import annotations

import os
import runpy
from collections.abc import Mapping


# Streamlit Community Cloud exposes app secrets through ``st.secrets`` rather
# than guaranteed process environment variables.  Copy only the small set of
# non-content connection settings needed by the UI into the environment before
# importing the existing frontend module.
_SECRET_ENV_NAMES = (
    "NF_BACKEND_URL",
    "API_BASE_URL",
    "NF_API_ACCESS_TOKEN",
    "CF_ACCESS_CLIENT_ID",
    "CF_ACCESS_CLIENT_SECRET",
    "STREAMLIT_REMOTE_BACKEND",
    "STREAMLIT_CLOUD_MODE",
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

# ``NF_BACKEND_URL`` is the stable public HTTPS endpoint (without ``/api``).
# Keeping the alias here lets Streamlit Cloud use a clear secret name while
# the existing UI continues to consume API_BASE_URL.
_backend_url = os.getenv("NF_BACKEND_URL", "").strip().rstrip("/")
if _backend_url:
    os.environ["API_BASE_URL"] = _backend_url
    os.environ["STREAMLIT_REMOTE_BACKEND"] = "true"
    os.environ["STREAMLIT_CLOUD_MODE"] = "true"

# Community Cloud cannot run the full Docker/worker stack.  For local smoke
# testing without a remote URL we retain the embedded fallback; production
# deployment should always set NF_BACKEND_URL so documents and jobs stay on the
# long-lived server instead of the ephemeral Streamlit filesystem.
_CLOUD_DEFAULTS = {
    "ENVIRONMENT": "production",
    "API_BASE_URL": "http://127.0.0.1:8000",
    "CORS_ORIGINS": "*",
    "DATABASE_URL": "sqlite:///./data/runtime/nf_agent.db",
    "USE_RQ": "false",
    "STORAGE_BACKEND": "local",
    "STORAGE_ROOT": "./data/runtime/objects",
    "CHROMA_PATH": "./data/runtime/chroma",
    "ALLOW_EMBEDDING_DOWNLOAD": "false",
    "ALLOW_EMBEDDING_FALLBACK": "true",
    "GROBID_URL": "",
    "STREAMLIT_CLOUD_MODE": "true",
}
for _name, _value in _CLOUD_DEFAULTS.items():
    os.environ.setdefault(_name, _value)

if not _backend_url:
    from app.cloud_runtime import ensure_embedded_api  # noqa: E402

    ensure_embedded_api()

runpy.run_module("ui.streamlit_app", run_name="__main__")
