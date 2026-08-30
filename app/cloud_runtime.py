from __future__ import annotations

import threading
import time
from urllib.request import urlopen


_lock = threading.Lock()
_server_thread: threading.Thread | None = None


def _healthy(url: str) -> bool:
    try:
        with urlopen(url, timeout=1) as response:  # noqa: S310 - local health endpoint only
            return response.status == 200
    except Exception:
        return False


def ensure_embedded_api(host: str = "127.0.0.1", port: int = 8000, timeout: float = 30.0) -> None:
    """Start one in-process FastAPI server for Streamlit Community Cloud."""

    global _server_thread
    health_url = f"http://{host}:{port}/api/health"
    if _healthy(health_url):
        return

    with _lock:
        if not _server_thread or not _server_thread.is_alive():
            import uvicorn

            def serve() -> None:
                uvicorn.run(
                    "app.main:app",
                    host=host,
                    port=port,
                    log_level="warning",
                    access_log=False,
                )

            _server_thread = threading.Thread(target=serve, name="nf-atlas-api", daemon=True)
            _server_thread.start()

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _healthy(health_url):
            return
        if _server_thread and not _server_thread.is_alive():
            break
        time.sleep(0.2)
    raise RuntimeError("Streamlit Cloud 内嵌 API 启动失败，请查看应用日志。")
