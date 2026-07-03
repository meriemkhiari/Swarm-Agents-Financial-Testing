import pytest
from datetime import datetime, timedelta, timezone
from agents.agent_geolocation.geolocation_velocity_agent import (
    GeolocationVelocityAgent,
    haversine_distance_km,
)


def test_haversine_distance():
    # Test distance between NYC (40.7128, -74.0060) and LA (34.0522, -118.2437) should be ~3936 km
    distance = haversine_distance_km(40.7128, -74.0060, 34.0522, -118.2437)
    assert abs(distance - 3936) < 100


@pytest.mark.asyncio
async def test_geolocation_agent_normal_movement(sample_transaction_payload):
    agent = GeolocationVelocityAgent(
        redis_url="redis://test:6379",
        input_channel="test-tx",
        output_channel="test-scores",
        threshold=0.80,
        max_plausible_velocity_kmh=900,
    )
    # First transaction: should return 0.0
    score1 = await agent.compute_score(sample_transaction_payload)
    assert score1 == 0.0

    # Second transaction, slow movement
    tx2 = sample_transaction_payload.model_copy(
        update={
            "latitude": 40.7228,
            "longitude": -74.0160,
            "timestamp": sample_transaction_payload.timestamp + timedelta(hours=1),
        }
    )
    score2 = await agent.compute_score(tx2)
    assert 0.0 <= score2 <= 1.0


@pytest.mark.asyncio
async def test_geolocation_agent_impossible_movement(sample_transaction_payload):
    agent = GeolocationVelocityAgent(
        redis_url="redis://test:6379",
        input_channel="test-tx",
        output_channel="test-scores",
        threshold=0.80,
        max_plausible_velocity_kmh=900,
    )
    await agent.compute_score(sample_transaction_payload)

    # Second transaction: instant teleport to LA!
    tx2 = sample_transaction_payload.model_copy(
        update={
            "latitude": 34.0522,
            "longitude": -118.2437,
            "timestamp": sample_transaction_payload.timestamp + timedelta(minutes=1),
        }
    )
    score2 = await agent.compute_score(tx2)
    assert score2 == 1.0
