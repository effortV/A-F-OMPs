"""Minimal authenticated configuration and health bridge for OMPs.

The bridge deliberately exposes no SQL, filesystem, shell, or NF endpoints.
Only six model-related settings can be updated. Scientific data never leaves
the dedicated OMPs server through this service.
"""
from __future__ import annotations

import hmac
import os
import sqlite3
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel


DATA_ROOT = Path(os.environ.get("OMPS_DATA_ROOT", "/data"))
SECRET_FILE = Path(os.environ.get("OMPS_SERVER_SECRET_FILE", "/run/omps-secrets/omps.env"))
API_TOKEN = os.environ.get("OMPS_API_TOKEN", "")
ALLOWED_KEYS = (
    "SILICONFLOW_API_KEY",
    "SILICONFLOW_BASE_URL",
    "LLM_MODEL",
    "LIT_PREPROCESS_MODEL",
    "EMBED_MODEL",
    "LLM_FALLBACK_MODELS",
)

app = FastAPI(title="OMPs bridge", docs_url=None, redoc_url=None, openapi_url=None)


class ModelConfig(BaseModel):
    SILICONFLOW_API_KEY: str | None = None
    SILICONFLOW_BASE_URL: str | None = None
    LLM_MODEL: str | None = None
    LIT_PREPROCESS_MODEL: str | None = None
    EMBED_MODEL: str | None = None
    LLM_FALLBACK_MODELS: str | None = None


def _authorize(authorization: str | None) -> None:
    if not API_TOKEN:
        raise HTTPException(status_code=503, detail="bridge token is not configured")
    expected = f"Bearer {API_TOKEN}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="unauthorized")


def _validate_value(key: str, value: str) -> str:
    clean = value.strip()
    if "\n" in clean or "\r" in clean or len(clean) > 4096:
        raise HTTPException(status_code=422, detail=f"invalid value for {key}")
    return clean


@app.get("/health")
def health() -> dict:
    db_path = DATA_ROOT / "store" / "zhizhi.db"
    papers = None
    if db_path.is_file():
        try:
            uri = f"file:{db_path.as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True, timeout=5) as connection:
                papers = int(connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0])
        except (sqlite3.Error, OSError):
            papers = None
    return {
        "ok": True,
        "dataset": (DATA_ROOT / "dataset.xlsx").is_file(),
        "database": db_path.is_file(),
        "papers": papers,
        "configured": SECRET_FILE.is_file(),
    }


@app.post("/config")
def update_config(config: ModelConfig, authorization: str | None = Header(default=None)) -> dict:
    _authorize(authorization)
    incoming = config.model_dump(exclude_none=True)
    current: dict[str, str] = {}
    if SECRET_FILE.is_file():
        for raw in SECRET_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in raw and not raw.lstrip().startswith("#"):
                key, value = raw.split("=", 1)
                if key in ALLOWED_KEYS:
                    current[key] = value
    for key, value in incoming.items():
        if key in ALLOWED_KEYS:
            current[key] = _validate_value(key, value)

    SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = SECRET_FILE.with_suffix(".tmp")
    temporary.write_text(
        "\n".join(f"{key}={current[key]}" for key in ALLOWED_KEYS if key in current) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, SECRET_FILE)
    return {"ok": True, "updated": sorted(incoming)}
