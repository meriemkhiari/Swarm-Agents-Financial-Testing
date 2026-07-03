from __future__ import annotations
import asyncio
import os
import numpy as np
import pandas as pd
import structlog
from prometheus_client import start_http_server
from sklearn.ensemble import IsolationForest
from agents.base_agent import BaseSwarmAgent
from shared.transaction_schema import TransactionPayload

logger = structlog.get_logger()

class AmountAnomalyAgent(BaseSwarmAgent):
    def __init__(
        self,
        redis_url: str,
        input_channel: str,
        output_channel: str,
        threshold: float,
        paysim_csv_path: str,
    ) -> None:
        super().__init__(
            agent_id="agent_amount",
            redis_url=redis_url,
            input_channel=input_channel,
            output_channel=output_channel,
            threshold=threshold,
            feature_used="amount",
        )
        self._paysim_csv_path = paysim_csv_path
        self._model: IsolationForest | None = None
        self._score_min: float = 0.0
        self._score_max: float = 0.0

    async def fit(self) -> None:
        training_values = await asyncio.to_thread(self._load_training_data)
        self._model = await asyncio.to_thread(self._train_model, training_values)
        raw_scores = self._model.decision_function(training_values)
        self._score_min = float(np.percentile(raw_scores, 1))
        self._score_max = float(np.percentile(raw_scores, 99))
        logger.info(
            "model_fit_complete",
            agent_id="agent_amount",
            n_samples=training_values.shape[0],
            score_min=self._score_min,
            score_max=self._score_max,
        )

    def _load_training_data(self) -> np.ndarray:
        frame = pd.read_csv(self._paysim_csv_path, usecols=["amount"])
        return frame["amount"].to_numpy(dtype=np.float64).reshape(-1, 1)

    def _train_model(self, training_values: np.ndarray) -> IsolationForest:
        model = IsolationForest(contamination=0.01, n_estimators=200, random_state=42)
        model.fit(training_values)
        return model

    async def compute_score(self, tx: TransactionPayload) -> float:
        if self._model is None:
            raise RuntimeError("AmountAnomalyAgent.fit() must be awaited before scoring")
        raw_score = self._model.decision_function(np.array([[tx.amount]], dtype=np.float64))[0]
        denominator = self._score_max - self._score_min
        if denominator == 0.0:
            return 0.0
        normalized = 1.0 - ((raw_score - self._score_min) / denominator)
        return float(np.clip(normalized, 0.0, 1.0))


def build_agent_from_env() -> AmountAnomalyAgent:
    return AmountAnomalyAgent(
        redis_url=os.environ.get("REDIS_URL", "redis://redis:6379/0"),
        input_channel=os.environ.get("TX_EVENTS_CHANNEL", "tx_events"),
        output_channel=os.environ.get("AGENT_SCORES_CHANNEL", "agent_scores"),
        threshold=float(os.environ.get("THRESHOLD_AGENT_AMOUNT", "0.65")),
        paysim_csv_path=os.environ.get("PAYSIM_CSV_PATH", "/data/paysim.csv"),
    )


async def main() -> None:
    start_http_server(int(os.environ.get("METRICS_PORT", "8001")))
    agent = build_agent_from_env()
    await agent.fit()
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())