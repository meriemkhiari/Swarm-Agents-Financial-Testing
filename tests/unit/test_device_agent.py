import pytest
import os
import tempfile
import asyncio
from agents.agent_device.device_profile_agent import DeviceProfileAgent


@pytest.mark.asyncio
async def test_device_agent_init_and_compute(sample_transaction_payload):
    sample_data = """step,type,amount,nameOrig,nameDest,isFraud,isFlaggedFraud
1,PAYMENT,100,device_xyz123,merchant1,0,0
1,PAYMENT,200,device_abc456,merchant2,0,0
1,PAYMENT,150,device_xyz123,merchant3,0,0
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(sample_data)
        temp_csv_path = f.name

    try:
        agent = DeviceProfileAgent(
            redis_url="redis://test:6379",
            input_channel="test-tx",
            output_channel="test-scores",
            threshold=0.75,
            paysim_csv_path=temp_csv_path,
        )
        # Known device
        known_tx = sample_transaction_payload.model_copy(update={"device_id": "device_xyz123"})
        known_score = await agent.compute_score(known_tx)
        assert 0.0 <= known_score <= 1.0

        # Unknown device
        unknown_tx = sample_transaction_payload.model_copy(update={"device_id": "device_new123"})
        unknown_score = await agent.compute_score(unknown_tx)
        assert unknown_score == 1.0

    finally:
        os.remove(temp_csv_path)
