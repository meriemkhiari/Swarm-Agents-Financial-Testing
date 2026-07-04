from __future__ import annotations

import httpx
import chromadb
from chromadb.api.types import Documents, Embeddings
from chromadb import Collection

from ai_feedback_loop.config import FeedbackLoopConfig


class OllamaEmbeddingFunction:
    def __init__(self, config: FeedbackLoopConfig) -> None:
        self._config = config

    def __call__(self, input: Documents) -> Embeddings:
        embeddings: Embeddings = []
        with httpx.Client(timeout=30.0) as client:
            for text in input:
                response = client.post(
                    f"{self._config.ollama_base_url}/api/embeddings",
                    json={"model": self._config.embedding_model, "prompt": text},
                )
                response.raise_for_status()
                embeddings.append(response.json()["embedding"])
        return embeddings


class DefectMemoryStore:
    def __init__(self, config: FeedbackLoopConfig) -> None:
        self._config = config
        self._client = chromadb.PersistentClient(path=config.vector_db_path)
        self._collection: Collection = self._client.get_or_create_collection(
            name="defect_memory",
            embedding_function=OllamaEmbeddingFunction(config),
        )

    def add_defect(self, defect_id: str, document: str, metadata: dict[str, str]) -> None:
        self._collection.upsert(ids=[defect_id], documents=[document], metadatas=[metadata])

    def query_similar(self, failure_text: str, n_results: int = 3) -> list[dict[str, str]]:
        stored_count = self._collection.count()
        if stored_count == 0:
            return []
        results = self._collection.query(
            query_texts=[failure_text],
            n_results=min(n_results, stored_count),
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        return [{"document": doc, **meta} for doc, meta in zip(documents, metadatas)]
