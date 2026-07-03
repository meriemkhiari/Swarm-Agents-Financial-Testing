import pytest
import uuid
from datetime import datetime, timezone
from shared.transaction_schema import TransactionPayload
from shared.agent_score_schema import AgentScore


@pytest.fixture
def sample_transaction_payload():
    return TransactionPayload(
        transaction_id=str(uuid.uuid4()),
        user_id="user-987",
        amount=1000.00,
        merchant_id="merchant-42",
        device_id="device-xyz123",
        latitude=40.7128,
        longitude=-74.0060,
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_fraud_scores():
    tx_id = str(uuid.uuid4())
    return [
        AgentScore(
            agent_id="agent_amount",
            transaction_id=tx_id,
            score=0.85,
            flag=True,
            threshold=0.65,
            feature_used="transaction_amount",
            computed_at=datetime.now(timezone.utc),
        ),
        AgentScore(
            agent_id="agent_frequency",
            transaction_id=tx_id,
            score=0.72,
            flag=True,
            threshold=0.70,
            feature_used="tx_per_hour",
            computed_at=datetime.now(timezone.utc),
        ),
        AgentScore(
            agent_id="agent_geolocation",
            transaction_id=tx_id,
            score=0.45,
            flag=False,
            threshold=0.80,
            feature_used="velocity",
            computed_at=datetime.now(timezone.utc),
        ),
        AgentScore(
            agent_id="agent_device",
            transaction_id=tx_id,
            score=0.20,
            flag=False,
            threshold=0.75,
            feature_used="device_novelty",
            computed_at=datetime.now(timezone.utc),
        ),
    ]


@pytest.fixture
def sample_legit_scores():
    tx_id = str(uuid.uuid4())
    return [
        AgentScore(
            agent_id="agent_amount",
            transaction_id=tx_id,
            score=0.20,
            flag=False,
            threshold=0.65,
            feature_used="transaction_amount",
            computed_at=datetime.now(timezone.utc),
        ),
        AgentScore(
            agent_id="agent_frequency",
            transaction_id=tx_id,
            score=0.15,
            flag=False,
            threshold=0.70,
            feature_used="tx_per_hour",
            computed_at=datetime.now(timezone.utc),
        ),
        AgentScore(
            agent_id="agent_geolocation",
            transaction_id=tx_id,
            score=0.30,
            flag=False,
            threshold=0.80,
            feature_used="velocity",
            computed_at=datetime.now(timezone.utc),
        ),
        AgentScore(
            agent_id="agent_device",
            transaction_id=tx_id,
            score=0.25,
            flag=False,
            threshold=0.75,
            feature_used="device_novelty",
            computed_at=datetime.now(timezone.utc),
        ),
    ]
