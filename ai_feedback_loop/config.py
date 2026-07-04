from __future__ import annotations

import os

from pydantic import BaseModel, Field


class FeedbackLoopConfig(BaseModel):
    project_root: str = Field(default_factory=lambda: os.environ.get("PROJECT_ROOT", os.getcwd()))
    tests_path: str = Field(default="tests/unit")
    ollama_base_url: str = Field(default_factory=lambda: os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = Field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "llama3"))
    embedding_model: str = Field(default_factory=lambda: os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    vector_db_path: str = Field(default=".feedback_loop/chroma")
    defects_path: str = Field(default=".feedback_loop/defects")
    max_repair_attempts: int = Field(default=3)
    auto_apply_patches: bool = Field(default=False)
