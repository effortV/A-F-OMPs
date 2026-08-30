"""Entrypoint for the persistent, server-hosted OMPs Streamlit process."""
from __future__ import annotations

import os
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SECRET_FILE = Path(os.environ.get("OMPS_SERVER_SECRET_FILE", "/run/omps-secrets/omps.env"))
ALLOWED_KEYS = {
    "SILICONFLOW_API_KEY",
    "SILICONFLOW_BASE_URL",
    "LLM_MODEL",
    "LIT_PREPROCESS_MODEL",
    "EMBED_MODEL",
    "LLM_FALLBACK_MODELS",
}


def _refresh_server_environment() -> None:
    """Reload the small allow-listed secret file on every Streamlit rerun."""
    if not SECRET_FILE.is_file():
        return
    for raw in SECRET_FILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in ALLOWED_KEYS:
            os.environ[key] = value.strip()


_refresh_server_environment()
os.environ.setdefault("OMPS_SERVER_MODE", "true")
runpy.run_path(str(ROOT / "zhizhi" / "ui" / "app.py"), run_name="__main__")
