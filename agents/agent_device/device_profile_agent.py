from __future__ import annotations

import asyncio
import hashlib
import os

import numpy as np
import pandas as pd
import structlog
from prometheus_client import start_http_server

from agents.base_agent import BaseSwarmAgent
from shared.transaction_schema import TransactionPayload

logger = structlog.get_logger()


def hash_device_id(device_id: str) -> int:
    digest = hashlib.md5(device_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**32)


class DeviceProfileAgent(BaseSwarmAgent):
    def __init__(
        self,
        redis_url: str,
        input_channel: str,
        output_channel: str,
        threshold: float,
        paysim_csv_path: str,
    ) -> None:
        super().__init__(
            agent_id="agent_device",
            redis_url=redis_url,
            input_channel=input_channel,
            output_channel=output_channel,
            threshold=threshold,
            feature_used="device_novelty",
        )
        training_frame = pd.read_csv(paysim_csv_path, usecols=["nameOrig"])
        device_series = training_frame["nameOrig"].astype(str)
        hashed_series = device_series.map(hash_device_id)
        frequency_counts = hashed_series.value_counts()

        self._device_frequency: dict[int, int] = frequency_counts.to_dict()
        self._max_frequency: int = int(frequency_counts.max())

        logger.info(
            "model_fit_complete",
            agent_id="agent_device",
            n_unique_devices=len(self._device_frequency),
            max_frequency=self._max_frequency,
        )

    async def compute_score(self, tx: TransactionPayload) -> float:
        device_hash = hash_device_id(tx.device_id)
        frequency = self._device_frequency.get(device_hash)

        if frequency is None:
            return 1.0

        normalized_frequency = frequency / self._max_frequency
        novelty_score = 1.0 - normalized_frequency
        return float(np.clip(novelty_score, 0.0, 1.0))


def build_agent_from_env() -> DeviceProfileAgent:
    return DeviceProfileAgent(
        redis_url=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
        input_channel=os.environ.get("TX_EVENTS_CHANNEL", "tx_events"),
        output_channel=os.environ.get("AGENT_SCORES_CHANNEL", "agent_scores"),
        threshold=float(os.environ.get("THRESHOLD_AGENT_DEVICE", "0.75")),
        paysim_csv_path=os.environ.get("PAYSIM_CSV_PATH", "/data/paysim.csv"),
    )


async def main() -> None:
    start_http_server(int(os.environ.get("METRICS_PORT", "8004")))
    agent = build_agent_from_env()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())