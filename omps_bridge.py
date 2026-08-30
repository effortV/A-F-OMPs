"""Private authenticated runtime API for the standalone OMPs deployment.

The service is bound to server loopback and is reachable only through the
OMPs-specific restricted SSH key.  It exposes an explicit allowlist of OMPs
operations; there is no shell endpoint and no NF path, port, key or database.
"""
from __future__ import annotations

import hmac
import base64
import importlib
import os
import sqlite3
import tempfile
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


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


class RPCRequest(BaseModel):
    target: str
    name: str
    args: list[Any] = Field(default_factory=list)
    kwargs: dict[str, Any] = Field(default_factory=dict)


DB_CALLS = {"q", "q1", "kv_get", "kv_set", "jdict", "jlist"}
JOB_CALLS = {
    "recover_orphaned_jobs", "submit", "cancel", "get", "list_jobs",
}
AGENT_CALLS = {
    "set_active_session", "new_session", "delete_session", "list_sessions",
    "active_session", "visible_history",
}
MODULE_CALLS = {
    "worker.bootstrap_core_corpus": ("zhizhi.lit.worker", "bootstrap_core_corpus"),
    "worker.scan_new_pdfs": ("zhizhi.lit.worker", "scan_new_pdfs"),
    "worker.control": ("zhizhi.lit.worker", "control"),
    "worker.status": ("zhizhi.lit.worker", "status"),
    "worker.needs_fulltext": ("zhizhi.lit.worker", "needs_fulltext"),
    "kg.entity_choices": ("zhizhi.lit.kg", "entity_choices"),
    "kg.stats": ("zhizhi.lit.kg", "stats"),
    "kgviz.plotly_graph": ("zhizhi.lit.kgviz", "plotly_graph"),
    "kgviz.neighborhood_figure": ("zhizhi.lit.kgviz", "neighborhood_figure"),
    "kgviz.contradiction_heatmap": ("zhizhi.lit.kgviz", "contradiction_heatmap"),
    "kgviz.export_static": ("zhizhi.lit.kgviz", "export_static"),
    "kgviz.graph_facts": ("zhizhi.lit.kgviz", "graph_facts"),
    "kgviz.narrate_overview": ("zhizhi.lit.kgviz", "narrate_overview"),
    "kgviz.narrate_entity": ("zhizhi.lit.kgviz", "narrate_entity"),
    "kgviz.narrate_conflicts": ("zhizhi.lit.kgviz", "narrate_conflicts"),
    "plots.list_figures": ("zhizhi.ml.plots", "list_figures"),
    "production.train_production": ("zhizhi.ml.production", "train_production"),
    "model.ui_feature_defaults": ("zhizhi.ml.model", "ui_feature_defaults"),
}


@app.on_event("startup")
def start_runtime() -> None:
    """Own all durable workers and Agent jobs in the server API process."""
    if SECRET_FILE.is_file():
        for raw in SECRET_FILE.read_text(encoding="utf-8").splitlines():
            if "=" in raw and not raw.lstrip().startswith("#"):
                key, value = raw.split("=", 1)
                if key in ALLOWED_KEYS:
                    os.environ[key] = value
    from zhizhi.agents.registry import all_agents
    from zhizhi.core import db, jobs
    from zhizhi.lit import worker
    from zhizhi.tools import lit_tools

    db.init()
    all_agents()
    jobs.recover_orphaned_jobs()
    if db.kv_get("lit_worker", "paused") == "running":
        worker.ensure_thread()
    lit_tools.ensure_literature_scheduler()


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


def _encode(value: Any) -> Any:
    if isinstance(value, sqlite3.Row):
        return {key: _encode(value[key]) for key in value.keys()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"__omps_type__": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_encode(item) for item in value]
    module = type(value).__module__
    if module.startswith("plotly") and hasattr(value, "to_json"):
        return {"__omps_type__": "plotly", "json": value.to_json()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "tolist"):
        try:
            return _encode(value.tolist())
        except Exception:  # noqa: BLE001
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _replace_uploads(value: Any, paths: list[Path]) -> Any:
    if isinstance(value, list):
        return [_replace_uploads(item, paths) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("__omps_type__") == "upload":
        index = int(value.get("index", -1))
        if index < 0 or index >= len(paths):
            raise ValueError("上传文件索引无效")
        return str(paths[index])
    return {key: _replace_uploads(item, paths) for key, item in value.items()}


def _dispatch(request: RPCRequest) -> Any:
    if request.target == "db" and request.name in DB_CALLS:
        from zhizhi.core import db

        return getattr(db, request.name)(*request.args, **request.kwargs)
    if request.target == "jobs" and request.name in JOB_CALLS:
        from zhizhi.core import jobs

        return getattr(jobs, request.name)(*request.args, **request.kwargs)
    if request.target == "agent" and request.name in AGENT_CALLS:
        from zhizhi.core import agent

        return getattr(agent, request.name)(*request.args, **request.kwargs)
    if request.target == "tool":
        from zhizhi.agents.registry import all_agents
        from zhizhi.core.tools import REGISTRY

        all_agents()
        if request.name not in REGISTRY.tools:
            raise ValueError(f"未知工具 {request.name}")
        if request.args:
            raise ValueError("工具 RPC 只接受命名参数")
        return REGISTRY.call(request.name, request.kwargs, agent="cloud-ui")
    if request.target == "module" and request.name in MODULE_CALLS:
        module_name, function_name = MODULE_CALLS[request.name]
        function = getattr(importlib.import_module(module_name), function_name)
        return function(*request.args, **request.kwargs)
    raise ValueError(f"不允许的远程调用：{request.target}.{request.name}")


def _rpc_response(request: RPCRequest) -> dict:
    try:
        return {"ok": True, "result": _encode(_dispatch(request))}
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=6)[-3000:],
        }


@app.get("/health")
def health() -> dict:
    db_path = DATA_ROOT / "store" / "zhizhi.db"
    papers = None
    if db_path.is_file():
        try:
            # SQLite WAL mode normally wants to create a shared-memory file.
            # The health service has a read-only data mount, so immutable mode
            # is required for this diagnostic query.
            uri = f"file:{db_path.as_posix()}?mode=ro&immutable=1"
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
            os.environ[key] = current[key]

    SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = SECRET_FILE.with_suffix(".tmp")
    temporary.write_text(
        "\n".join(f"{key}={current[key]}" for key in ALLOWED_KEYS if key in current) + "\n",
        encoding="utf-8",
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, SECRET_FILE)
    if incoming:
        from zhizhi.agents.registry import AGENTS, all_agents
        from zhizhi.core.llm import LLM

        LLM._client = None
        AGENTS.clear()
        all_agents()
    return {"ok": True, "updated": sorted(incoming)}


@app.post("/rpc")
def rpc(request: RPCRequest, authorization: str | None = Header(default=None)) -> dict:
    _authorize(authorization)
    return _rpc_response(request)


@app.post("/rpc-upload")
async def rpc_upload(
    payload: str = Form(...),
    uploads: list[UploadFile] = File(default=[]),
    authorization: str | None = Header(default=None),
) -> dict:
    _authorize(authorization)
    from zhizhi.core.config import CFG

    temporary_paths: list[Path] = []
    try:
        for uploaded in uploads:
            suffix = Path(uploaded.filename or "upload.bin").suffix
            with tempfile.NamedTemporaryFile(
                prefix="cloud_upload_",
                suffix=suffix,
                dir=CFG.new_pdf_dir,
                delete=False,
            ) as handle:
                while chunk := await uploaded.read(1024 * 1024):
                    handle.write(chunk)
                temporary_paths.append(Path(handle.name))
        raw = RPCRequest.model_validate_json(payload)
        request = RPCRequest(
            target=raw.target,
            name=raw.name,
            args=_replace_uploads(raw.args, temporary_paths),
            kwargs=_replace_uploads(raw.kwargs, temporary_paths),
        )
        return _rpc_response(request)
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


@app.get("/file")
def download_file(
    path: str,
    authorization: str | None = Header(default=None),
) -> FileResponse:
    _authorize(authorization)
    from zhizhi.core.config import CFG

    candidate = Path(path).resolve()
    root = CFG.root.resolve()
    if candidate != root and root not in candidate.parents:
        raise HTTPException(status_code=403, detail="path outside OMPs root")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(candidate)
