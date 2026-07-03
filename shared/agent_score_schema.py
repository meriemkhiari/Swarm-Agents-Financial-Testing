from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AgentScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    transaction_id: str
    score: float = Field(ge=0, le=1)
    flag: bool
    threshold: float
    feature_used: str
    computed_at: datetime