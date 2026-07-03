from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    transaction_id: str
    user_id: str
    amount: float = Field(gt=0)
    merchant_id: str
    device_id: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timestamp: datetime

    @field_validator("transaction_id")
    @classmethod
    def validate_transaction_id(cls, value: str) -> str:
        UUID(value, version=4)
        return value