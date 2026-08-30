from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False
    )

    app_name: str = "NF-Atlas 纳滤文献智能体"
    environment: Literal["development", "test", "production"] = "development"
    api_base_url: str = "http://localhost:8000"
    cors_origins: str = "http://localhost:8501"

    database_url: str = "sqlite:///./data/runtime/nf_agent.db"
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "nf-literature"
    priority_queue_name: str = "nf-priority"
    use_rq: bool = False

    siliconflow_api_key: SecretStr | None = None
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    llm_model: str = "Pro/deepseek-ai/DeepSeek-V3.2"
    chat_llm_model: str = "deepseek-ai/DeepSeek-V4-Pro"
    llm_temperature: float = 0.1
    llm_timeout_seconds: int = 120

    openalex_api_key: SecretStr | None = None
    openalex_email: str | None = None
    semantic_scholar_api_key: SecretStr | None = None
    elsevier_api_key: SecretStr | None = None
    elsevier_insttoken: SecretStr | None = None
    unpaywall_email: str | None = None

    storage_backend: Literal["local", "minio"] = "local"
    storage_root: Path = Path("./data/runtime/objects")
    minio_endpoint: str = "localhost:9000"
    minio_access_key: SecretStr | None = None
    minio_secret_key: SecretStr | None = None
    minio_secure: bool = False
    minio_bucket: str = "nf-literature"

    neo4j_uri: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr | None = None
    neo4j_database: str = "neo4j"

    chroma_path: Path = Path("./data/runtime/chroma")
    chroma_host: str | None = None
    chroma_port: int = 8000
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8
    allow_embedding_download: bool = False
    allow_embedding_fallback: bool = False
    hf_endpoint: str = "https://hf-mirror.com"

    grobid_url: str | None = "http://localhost:8070"
    mineru_command: str | None = None
    parser_timeout_seconds: int = 600
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 8
    max_discovery_results: int = 500

    # Public-web full text fallback. It only reads publicly reachable HTTP(S)
    # resources and never handles logins, cookies, CAPTCHAs or paywall bypasses.
    direct_web_fetch: bool = True
    direct_web_respect_robots: bool = True
    direct_web_min_interval_seconds: float = 1.0
    direct_web_max_bytes: int = 100 * 1024 * 1024
    direct_web_max_redirects: int = 8
    direct_html_min_chars: int = 3000

    @field_validator("storage_root", "chroma_path", mode="before")
    @classmethod
    def expand_path(cls, value: str | Path) -> Path:
        return Path(value).expanduser()

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def ensure_runtime_dirs(self) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        if self.database_url.startswith("sqlite:///"):
            Path(self.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_dirs()
    return settings
