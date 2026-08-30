"""SSH-tunnel client for the Streamlit Cloud frontend.

The public Streamlit process never owns scientific state.  It forwards a
small, authenticated RPC protocol through a host-key-pinned SSH connection to
the dedicated OMPs backend bound to 127.0.0.1 on the server.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import select
import socketserver
import threading
from pathlib import Path
from typing import Any

import requests


def enabled() -> bool:
    return os.environ.get("OMPS_REMOTE_MODE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


class RemoteError(RuntimeError):
    """Raised when the private OMPs backend cannot complete a request."""


_lock = threading.RLock()
_client = None
_forwarder = None
_forward_thread: threading.Thread | None = None
_local_port: int | None = None


def _expected_host_fingerprint() -> str:
    return os.environ.get("OMPS_SSH_HOST_KEY_SHA256", "").strip()


def _fingerprint(key) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def _load_private_key(paramiko, raw: str):
    failures: list[str] = []
    for cls in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            return cls.from_private_key(io.StringIO(raw))
        except Exception as exc:  # noqa: BLE001 - try supported key formats
            failures.append(type(exc).__name__)
    raise RemoteError("无法读取 OMPs SSH 私钥：" + ", ".join(failures))


def _new_forwarder(transport, remote_host: str, remote_port: int):
    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            channel = transport.open_channel(
                "direct-tcpip",
                (remote_host, remote_port),
                self.request.getpeername(),
            )
            if channel is None:
                return
            try:
                while True:
                    readable, _, _ = select.select([self.request, channel], [], [], 30)
                    if self.request in readable:
                        data = self.request.recv(65536)
                        if not data:
                            break
                        channel.sendall(data)
                    if channel in readable:
                        data = channel.recv(65536)
                        if not data:
                            break
                        self.request.sendall(data)
            finally:
                channel.close()

    class Forwarder(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    return Forwarder(("127.0.0.1", 0), Handler)


def _alive() -> bool:
    if not (_client and _forwarder and _forward_thread and _forward_thread.is_alive()):
        return False
    transport = _client.get_transport()
    return bool(transport and transport.is_active())


def close() -> None:
    global _client, _forwarder, _forward_thread, _local_port
    with _lock:
        if _forwarder:
            try:
                _forwarder.shutdown()
                _forwarder.server_close()
            except Exception:  # noqa: BLE001
                pass
        if _client:
            try:
                _client.close()
            except Exception:  # noqa: BLE001
                pass
        _client = _forwarder = _forward_thread = _local_port = None


def ensure_tunnel() -> int:
    """Return a healthy local forwarding port, reconnecting when necessary."""
    global _client, _forwarder, _forward_thread, _local_port
    if not enabled():
        raise RemoteError("OMPS_REMOTE_MODE 未启用")
    with _lock:
        if _alive() and _local_port:
            return _local_port
        close()
        try:
            import paramiko
        except ImportError as exc:
            raise RemoteError("缺少 paramiko，无法建立 OMPs SSH 隧道") from exc

        host = os.environ.get("OMPS_SSH_HOST", "").strip()
        port = int(os.environ.get("OMPS_SSH_PORT", "9012"))
        username = os.environ.get("OMPS_SSH_USERNAME", "").strip()
        private_key = os.environ.get("OMPS_SSH_PRIVATE_KEY", "").strip()
        expected = _expected_host_fingerprint()
        remote_host = os.environ.get("OMPS_REMOTE_API_HOST", "127.0.0.1").strip()
        remote_port = int(os.environ.get("OMPS_REMOTE_API_PORT", "8602"))
        if not all((host, username, private_key, expected)):
            raise RemoteError("OMPs SSH Secrets 不完整")

        class FingerprintPolicy(paramiko.MissingHostKeyPolicy):
            def missing_host_key(self, client, hostname, key) -> None:
                actual = _fingerprint(key)
                if actual != expected:
                    raise paramiko.SSHException(
                        f"服务器主机指纹不匹配：期望 {expected}，实际 {actual}"
                    )
                client.get_host_keys().add(hostname, key.get_name(), key)

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(FingerprintPolicy())
        try:
            client.connect(
                hostname=host,
                port=port,
                username=username,
                pkey=_load_private_key(paramiko, private_key),
                allow_agent=False,
                look_for_keys=False,
                timeout=20,
                banner_timeout=20,
                auth_timeout=20,
            )
            transport = client.get_transport()
            if transport is None:
                raise RemoteError("SSH transport 未建立")
            transport.set_keepalive(30)
            forwarder = _new_forwarder(transport, remote_host, remote_port)
            thread = threading.Thread(
                target=forwarder.serve_forever,
                name="omps-ssh-forwarder",
                daemon=True,
            )
            thread.start()
        except Exception:
            client.close()
            raise

        _client = client
        _forwarder = forwarder
        _forward_thread = thread
        _local_port = int(forwarder.server_address[1])
        return _local_port


def _base_url() -> str:
    return f"http://127.0.0.1:{ensure_tunnel()}"


def _headers() -> dict[str, str]:
    token = os.environ.get("OMPS_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def health(timeout: float = 15) -> dict:
    response = requests.get(f"{_base_url()}/health", timeout=timeout)
    response.raise_for_status()
    return response.json()


def sync_model_config() -> dict:
    payload = {
        key: os.environ.get(key, "")
        for key in (
            "SILICONFLOW_API_KEY",
            "SILICONFLOW_BASE_URL",
            "LLM_MODEL",
            "LIT_PREPROCESS_MODEL",
            "EMBED_MODEL",
            "LLM_FALLBACK_MODELS",
        )
        if os.environ.get(key, "")
    }
    response = requests.post(
        f"{_base_url()}/config",
        headers=_headers(),
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    marker = value.get("__omps_type__")
    if marker == "bytes":
        return base64.b64decode(value.get("data", ""))
    if marker == "plotly":
        import plotly.io as pio

        return pio.from_json(value["json"])
    return {key: _decode(item) for key, item in value.items()}


def _prepare(value: Any, key: str = "", uploads: list[Path] | None = None) -> Any:
    uploads = uploads if uploads is not None else []
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and (key.endswith("_path") or key in {"path", "file"}):
        candidate = Path(value)
        if candidate.is_file():
            index = len(uploads)
            uploads.append(candidate)
            return {"__omps_type__": "upload", "index": index, "name": candidate.name}
    if isinstance(value, dict):
        return {str(k): _prepare(v, str(k), uploads) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_prepare(item, key, uploads) for item in value]
    return value


def call(target: str, name: str, *args: Any, timeout: float = 1200, **kwargs: Any) -> Any:
    uploads: list[Path] = []
    payload = {
        "target": target,
        "name": name,
        "args": _prepare(list(args), uploads=uploads),
        "kwargs": _prepare(kwargs, uploads=uploads),
    }
    try:
        if uploads:
            handles = [path.open("rb") for path in uploads]
            try:
                files = [
                    ("uploads", (path.name, handle, "application/octet-stream"))
                    for path, handle in zip(uploads, handles)
                ]
                response = requests.post(
                    f"{_base_url()}/rpc-upload",
                    headers=_headers(),
                    data={"payload": json.dumps(payload, ensure_ascii=False)},
                    files=files,
                    timeout=timeout,
                )
            finally:
                for handle in handles:
                    handle.close()
        else:
            response = requests.post(
                f"{_base_url()}/rpc",
                headers=_headers(),
                json=payload,
                timeout=timeout,
            )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise RemoteError(body.get("error") or "OMPs 远程调用失败")
        return _decode(body.get("result"))
    except requests.RequestException as exc:
        close()
        raise RemoteError(f"OMPs 服务器连接失败：{type(exc).__name__}: {exc}") from exc


def fetch_file(path: str, timeout: float = 120) -> bytes:
    try:
        response = requests.get(
            f"{_base_url()}/file",
            headers=_headers(),
            params={"path": path},
            timeout=timeout,
        )
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        close()
        raise RemoteError(f"读取服务器文件失败：{type(exc).__name__}: {exc}") from exc
