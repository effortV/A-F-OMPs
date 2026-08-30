"""Streamlit Community Cloud entrypoint for the standalone OMPs project.

This wrapper only maps Streamlit Secrets to the OMPs environment before the
existing ``zhizhi.ui.app`` is loaded.  It does not import or connect to the
separate NF project.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from collections.abc import Mapping


_SECRET_ENV_NAMES = (
    "SILICONFLOW_API_KEY",
    "SILICONFLOW_BASE_URL",
    "LLM_MODEL",
    "LIT_PREPROCESS_MODEL",
    "EMBED_MODEL",
    "LLM_FALLBACK_MODELS",
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
runpy.run_path(str(_ROOT / "zhizhi" / "ui" / "app.py"), run_name="__main__")
