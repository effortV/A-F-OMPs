from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import Any

from app.config import Settings, get_settings
from app.models import Chunk, Document


class EmbeddingProvider:
    _model_cache: dict[tuple[str, str, bool], tuple[Any, str, str | None]] = {}
    _cache_lock = threading.Lock()

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.backend = "hash-fallback"
        self.load_error: str | None = None
        self._load_attempted = False

    def _load(self) -> None:
        if self._load_attempted:
            if self.load_error and not self.settings.allow_embedding_fallback:
                raise RuntimeError(self.load_error)
            return
        self._load_attempted = True
        cache_key = (
            self.settings.embedding_model,
            self.settings.embedding_device,
            self.settings.allow_embedding_download,
        )
        with self._cache_lock:
            cached = self._model_cache.get(cache_key)
        if cached:
            self.model, self.backend, self.load_error = cached
            if self.load_error and not self.settings.allow_embedding_fallback:
                raise RuntimeError(self.load_error)
            return
        try:
            from sentence_transformers import SentenceTransformer

            try:
                # A cached bge-m3 must never wait on optional Hugging Face HEAD requests.
                # Only contact the configured mirror if the local model is genuinely incomplete.
                self.model = SentenceTransformer(
                    self.settings.embedding_model,
                    device=self.settings.embedding_device,
                    local_files_only=True,
                )
            except Exception:
                if not self.settings.allow_embedding_download:
                    raise
                self.model = SentenceTransformer(
                    self.settings.embedding_model,
                    device=self.settings.embedding_device,
                    local_files_only=False,
                )
            self.backend = self.settings.embedding_model
            with self._cache_lock:
                self._model_cache[cache_key] = (self.model, self.backend, None)
        except Exception as exc:
            self.model = False
            self.load_error = (
                f"bge-m3 加载失败（{type(exc).__name__}: {exc}）。"
                "请检查 HF_ENDPOINT/网络或模型持久卷；未启用哈希向量降级。"
            )
            if not self.settings.allow_embedding_fallback:
                raise RuntimeError(self.load_error) from exc

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        if self.model:
            vectors = self.model.encode(
                texts,
                batch_size=self.settings.embedding_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vectors.tolist()
        return [self._hash_embedding(text) for text in texts]

    @staticmethod
    def _hash_embedding(text: str, dimensions: int = 1024) -> list[float]:
        vector = [0.0] * dimensions
        tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text.casefold())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "big")
            index = number % dimensions
            vector[index] += 1.0 if (number >> 10) & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class VectorStore:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.embedder = EmbeddingProvider(self.settings)
        self.client = None

    def _client(self):
        if self.client is None:
            import chromadb

            if self.settings.chroma_host:
                self.client = chromadb.HttpClient(host=self.settings.chroma_host, port=self.settings.chroma_port)
            else:
                self.client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        return self.client

    @staticmethod
    def _collection_name(knowledge_base_id: str, backend: str) -> str:
        backend_key = hashlib.sha256(backend.encode("utf-8")).hexdigest()[:10]
        return "kb_" + knowledge_base_id.replace("-", "_") + "_" + backend_key

    @staticmethod
    def _collection_names(client: Any) -> set[str]:
        """Support both Chroma <=0.5 Collection objects and 0.6+ name strings."""
        return {
            item if isinstance(item, str) else str(item.name)
            for item in client.list_collections()
        }

    def upsert_document(self, document: Document, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        collection = self._client().get_or_create_collection(
            self._collection_name(document.knowledge_base_id, self.embedder.backend),
            metadata={"hnsw:space": "cosine", "embedding_backend": self.embedder.backend},
        )
        ids = [chunk.vector_id or chunk.id for chunk in chunks]
        metadatas = [
            {
                "document_id": document.id,
                "evidence_mode": "metadata-only" if (document.metadata_json or {}).get("metadata_only") else "fulltext",
                "chunk_id": chunk.id,
                "doi": document.doi_normalized or "",
                "title": document.title[:1000],
                "section": chunk.section or "",
                "block_kind": chunk.block_kind or "",
                "source_label": chunk.source_label or "",
                "page_start": chunk.page_start if chunk.page_start is not None else -1,
                "page_end": chunk.page_end if chunk.page_end is not None else -1,
            }
            for chunk in chunks
        ]
        collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    def delete_document(self, document: Document) -> None:
        client = self._client()
        name = self._collection_name(document.knowledge_base_id, self.embedder.backend)
        if name not in self._collection_names(client):
            return
        client.get_collection(name).delete(where={"document_id": document.id})

    def replace_document(self, document: Document, chunks: list[Chunk]) -> None:
        # Upgrading a metadata-only record must not leave stale abstract vectors behind.
        self.embedder._load()
        self.delete_document(document)
        self.upsert_document(document, chunks)

    def search(self, knowledge_base_id: str, query: str, limit: int = 8) -> list[dict[str, Any]]:
        client = self._client()
        query_embedding = self.embedder.encode([query])
        name = self._collection_name(knowledge_base_id, self.embedder.backend)
        if name not in self._collection_names(client):
            return []
        collection = client.get_collection(name)
        collection_count = collection.count()
        if collection_count <= 0:
            return []
        result = collection.query(
            query_embeddings=query_embedding,
            n_results=min(collection_count, max(1, limit)),
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for text, metadata, distance in zip(
            (result.get("documents") or [[]])[0],
            (result.get("metadatas") or [[]])[0],
            (result.get("distances") or [[]])[0],
            strict=False,
        ):
            item = dict(metadata or {})
            item.update({"quote": text, "score": round(1.0 - float(distance), 5), "source": "vector"})
            output.append(item)
        return output
