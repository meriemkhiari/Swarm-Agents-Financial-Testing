from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from agents.agent_amount.amount_anomaly_agent import AmountAnomalyAgent
from shared.transaction_schema import TransactionPayload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def paysim_csv(tmp_path_factory) -> str:
    """A csv large enough for a non-degenerate percentile computation.

    IsolationForest with contamination=0.01 needs enough samples for the
    1st/99th percentile split to mean anything; 3 rows (as in the original
    test) collapses percentile and min/max to the same value and hides bugs
    in the normalization branch.
    """
    rng = np.random.default_rng(seed=42)
    normal_amounts = rng.normal(loc=150.0, scale=30.0, size=500).clip(min=1.0)
    outlier_amounts = rng.uniform(low=5000.0, high=20000.0, size=5)
    amounts = np.concatenate([normal_amounts, outlier_amounts])
    rng.shuffle(amounts)

    frame = pd.DataFrame({
        "step": 1,
        "type": "PAYMENT",
        "amount": amounts,
        "nameOrig": [f"user{i}" for i in range(len(amounts))],
        "nameDest": [f"merchant{i}" for i in range(len(amounts))],
        "isFraud": 0,
        "isFlaggedFraud": 0,
    })
    path = tmp_path_factory.mktemp("data") / "paysim.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
async def fitted_agent(paysim_csv: str) -> AmountAnomalyAgent:
    agent = AmountAnomalyAgent(
        redis_url="redis://test:6379",
        input_channel="test-tx",
        output_channel="test-scores",
        threshold=0.65,
        paysim_csv_path=paysim_csv,
    )
    await agent.fit()
    return agent


def make_transaction(amount: float, **overrides) -> TransactionPayload:
    base = dict(
        transaction_id="tx_1",
        user_id="user_1",
        amount=amount,
        merchant_id="merchant_1",
        device_id="device_1",
        latitude=40.7128,
        longitude=-74.0060,
        timestamp="2026-07-04T00:00:00Z",
    )
    base.update(overrides)
    return TransactionPayload(**base)


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------

class TestAmountAnomalyAgent:

    @pytest.mark.asyncio
    async def test_fit_populates_score_bounds(self, fitted_agent: AmountAnomalyAgent):
        assert fitted_agent._model is not None
        assert fitted_agent._score_min <= fitted_agent._score_max

    @pytest.mark.asyncio
    async def test_typical_amount_scores_low(self, fitted_agent: AmountAnomalyAgent):
        tx = make_transaction(amount=150.0)
        score = await fitted_agent.compute_score(tx)
        assert 0.0 <= score <= 1.0
        assert score < 0.5, "an amount matching the training population mean should not be flagged"

    @pytest.mark.asyncio
    async def test_extreme_outlier_scores_high(self, fitted_agent: AmountAnomalyAgent):
        tx = make_transaction(amount=50_000.0)
        score = await fitted_agent.compute_score(tx)
        assert score > 0.8, "an amount an order of magnitude beyond training outliers must be flagged"

    @pytest.mark.asyncio
    async def test_monotonic_with_deviation(self, fitted_agent: AmountAnomalyAgent):
        """Anomaly score should not decrease as amount moves further from the training mean."""
        amounts = [150.0, 1_000.0, 5_000.0, 20_000.0]
        scores = [await fitted_agent.compute_score(make_transaction(amount=a)) for a in amounts]
        assert scores == sorted(scores), f"expected non-decreasing scores, got {scores}"

    @pytest.mark.asyncio
    async def test_scoring_before_fit_raises(self, paysim_csv: str):
        agent = AmountAnomalyAgent(
            redis_url="redis://test:6379",
            input_channel="test-tx",
            output_channel="test-scores",
            threshold=0.65,
            paysim_csv_path=paysim_csv,
        )
        with pytest.raises(RuntimeError, match="fit"):
            await agent.compute_score(make_transaction(amount=100.0))

    @pytest.mark.asyncio
    async def test_missing_csv_raises(self):
        agent = AmountAnomalyAgent(
            redis_url="redis://test:6379",
            input_channel="test-tx",
            output_channel="test-scores",
            threshold=0.65,
            paysim_csv_path="/nonexistent/path.csv",
        )
        with pytest.raises(FileNotFoundError):
            await agent.fit()


# ---------------------------------------------------------------------------
# Degenerate-distribution edge case
# ---------------------------------------------------------------------------

class TestDegenerateTrainingData:

    @pytest.mark.asyncio
    async def test_constant_amounts_do_not_divide_by_zero(self, tmp_path):
        frame = pd.DataFrame({"amount": [100.0] * 50})
        path = tmp_path / "constant.csv"
        frame.to_csv(path, index=False)

        agent = AmountAnomalyAgent(
            redis_url="redis://test:6379",
            input_channel="test-tx",
            output_channel="test-scores",
            threshold=0.65,
            paysim_csv_path=str(path),
        )
        await agent.fit()

        score = await agent.compute_score(make_transaction(amount=100.0))
        assert score == 0.0
        assert math.isfinite(score)


# ---------------------------------------------------------------------------
# Property-based test: output contract must hold for any input
# ---------------------------------------------------------------------------

class TestScoreContract:

    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(amount=st.floats(min_value=0.01, max_value=1e9, allow_nan=False, allow_infinity=False))
    async def test_score_always_in_unit_interval(self, fitted_agent: AmountAnomalyAgent, amount: float):
        score = await fitted_agent.compute_score(make_transaction(amount=amount))
        assert 0.0 <= score <= 1.0
        assert math.isfinite(score)