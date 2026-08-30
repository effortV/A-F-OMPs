from __future__ import annotations

import io
import json
import mimetypes
import shutil
from pathlib import Path
from app.config import Settings, get_settings


class ObjectStorage:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._minio = None
        if self.settings.storage_backend == "minio":
            from minio import Minio

            if not self.settings.minio_access_key or not self.settings.minio_secret_key:
                raise RuntimeError("MinIO 后端需要 MINIO_ACCESS_KEY 和 MINIO_SECRET_KEY")
            self._minio = Minio(
                self.settings.minio_endpoint,
                access_key=self.settings.minio_access_key.get_secret_value(),
                secret_key=self.settings.minio_secret_key.get_secret_value(),
                secure=self.settings.minio_secure,
            )
            if not self._minio.bucket_exists(self.settings.minio_bucket):
                self._minio.make_bucket(self.settings.minio_bucket)

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> str:
        key = self._safe_key(key)
        if self._minio:
            self._minio.put_object(
                self.settings.minio_bucket,
                key,
                io.BytesIO(data),
                len(data),
                content_type=content_type or mimetypes.guess_type(key)[0] or "application/octet-stream",
            )
        else:
            destination = self.settings.storage_root / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        return key

    def put_file(self, key: str, source: str | Path, content_type: str | None = None) -> str:
        key = self._safe_key(key)
        source_path = Path(source)
        if self._minio:
            self._minio.fput_object(
                self.settings.minio_bucket,
                key,
                str(source_path),
                content_type=content_type or mimetypes.guess_type(key)[0],
            )
        else:
            destination = self.settings.storage_root / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
        return key

    def put_json(self, key: str, value: object) -> str:
        return self.put_bytes(key, json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8"), "application/json")

    def get_bytes(self, key: str) -> bytes:
        key = self._safe_key(key)
        if self._minio:
            response = self._minio.get_object(self.settings.minio_bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()
        return (self.settings.storage_root / key).read_bytes()

    def materialize(self, key: str, destination: str | Path) -> Path:
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if self._minio:
            self._minio.fget_object(self.settings.minio_bucket, self._safe_key(key), str(destination_path))
        else:
            shutil.copy2(self.settings.storage_root / self._safe_key(key), destination_path)
        return destination_path

    @staticmethod
    def _safe_key(key: str) -> str:
        clean = key.replace("\\", "/").lstrip("/")
        if any(part in {"", ".", ".."} for part in clean.split("/")):
            raise ValueError(f"非法对象键: {key}")
        return clean
