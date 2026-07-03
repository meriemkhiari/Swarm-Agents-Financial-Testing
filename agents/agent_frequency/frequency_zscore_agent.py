from __future__ import annotations

import asyncio
import os
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import pandas as pd
import structlog
from prometheus_client import start_http_server

from agents.base_agent import BaseSwarmAgent
from shared.transaction_schema import TransactionPayload

logger = structlog.get_logger()


class FrequencyZScoreAgent(BaseSwarmAgent):
    def __init__(
        self,
        redis_url: str,
        input_channel: str,
        output_channel: str,
        threshold: float,
        paysim_csv_path: str,
        window_size_seconds: int = 3600,
    ) -> None:
        super().__init__(
            agent_id="agent_frequency",
            redis_url=redis_url,
            input_channel=input_channel,
            output_channel=output_channel,
            threshold=threshold,
            feature_used="tx_per_hour",
        )
        self._window_size_seconds = window_size_seconds
        self._windows: dict[str, deque[datetime]] = defaultdict(deque)

        training_frame = pd.read_csv(paysim_csv_path, usecols=["step", "nameOrig"])
        per_user_hourly_counts = (
            training_frame.groupby(["nameOrig", "step"]).size().to_numpy(dtype=np.float64)
        )
        self._population_mean = float(np.mean(per_user_hourly_counts))
        self._population_std = float(np.std(per_user_hourly_counts)) or 1.0

        logger.info(
            "model_fit_complete",
            agent_id="agent_frequency",
            population_mean=self._population_mean,
            population_std=self._population_std,
        )

    def _evict_expired(self, user_id: str, now: datetime) -> None:
        window = self._windows[user_id]
        while window and (now - window[0]).total_seconds() > self._window_size_seconds:
            window.popleft()
        if not window:
            del self._windows[user_id]

    async def compute_score(self, tx: TransactionPayload) -> float:
        now = tx.timestamp
        window = self._windows[tx.user_id]
        window.append(now)
        tx_count = len(window)
        self._evict_expired(tx.user_id, now)

        z_score = (tx_count - self._population_mean) / self._population_std
        sigmoid_input = z_score - 2.0
        score = 1.0 / (1.0 + np.exp(-sigmoid_input))
        return float(np.clip(score, 0.0, 1.0))


def build_agent_from_env() -> FrequencyZScoreAgent:
    return FrequencyZScoreAgent(
        redis_url=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
        input_channel=os.environ.get("TX_EVENTS_CHANNEL", "tx_events"),
        output_channel=os.environ.get("AGENT_SCORES_CHANNEL", "agent_scores"),
        threshold=float(os.environ.get("THRESHOLD_AGENT_FREQUENCY", "0.70")),
        paysim_csv_path=os.environ.get("PAYSIM_CSV_PATH", "/data/paysim.csv"),
    )


async def main() -> None:
    start_http_server(int(os.environ.get("METRICS_PORT", "8002")))
    agent = build_agent_from_env()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())