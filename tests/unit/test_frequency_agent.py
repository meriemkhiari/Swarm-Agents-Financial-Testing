import pytest
import os
import tempfile
from datetime import datetime, timedelta, timezone
from agents.agent_frequency.frequency_zscore_agent import FrequencyZScoreAgent


@pytest.mark.asyncio
async def test_frequency_agent_init_and_compute(sample_transaction_payload):
    # Create tiny paysim-like CSV
    sample_data = """step,type,amount,nameOrig,nameDest,isFraud,isFlaggedFraud
1,PAYMENT,100,user1,merchant1,0,0
1,PAYMENT,200,user1,merchant2,0,0
1,PAYMENT,150,user2,merchant3,0,0
2,PAYMENT,300,user1,merchant4,0,0
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(sample_data)
        temp_csv_path = f.name

    try:
        agent = FrequencyZScoreAgent(
            redis_url="redis://test:6379",
            input_channel="test-tx",
            output_channel="test-scores",
            threshold=0.70,
            paysim_csv_path=temp_csv_path,
            window_size_seconds=3600,
        )
        # Compute initial score
        initial_score = await agent.compute_score(sample_transaction_payload)
        assert 0.0 <= initial_score <= 1.0

        # Compute multiple transactions quickly
        now = datetime.now(timezone.utc)
        for i in range(5):
            tx = sample_transaction_payload.model_copy(
                update={"timestamp": now + timedelta(seconds=10*i)}
            )
            score = await agent.compute_score(tx)
            assert 0.0 <= score <= 1.0

    finally:
        os.remove(temp_csv_path)
