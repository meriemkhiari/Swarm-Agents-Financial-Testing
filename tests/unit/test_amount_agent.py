import pytest
import os
import tempfile
import asyncio
from agents.agent_amount.amount_anomaly_agent import AmountAnomalyAgent


@pytest.mark.asyncio
async def test_amount_agent_init_and_compute(sample_transaction_payload):
    # Create a tiny sample paysim-like CSV for testing
    sample_data = """step,type,amount,nameOrig,nameDest,isFraud,isFlaggedFraud
1,PAYMENT,100,user1,merchant1,0,0
1,PAYMENT,200,user2,merchant2,0,0
1,PAYMENT,150,user3,merchant3,0,0
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(sample_data)
        temp_csv_path = f.name

    try:
        agent = AmountAnomalyAgent(
            redis_url="redis://test:6379",
            input_channel="test-tx",
            output_channel="test-scores",
            threshold=0.65,
            paysim_csv_path=temp_csv_path,
        )

        await agent.fit()

        # Test normal amount
        score = await agent.compute_score(sample_transaction_payload)
        assert 0.0 <= score <= 1.0

    finally:
        os.remove(temp_csv_path)
