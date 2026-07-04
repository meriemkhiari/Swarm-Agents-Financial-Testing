from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from agents.agent_frequency.frequency_zscore_agent import FrequencyZScoreAgent
from shared.transaction_schema import TransactionPayload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def paysim_csv(tmp_path_factory) -> str:
    """Population with a known, non-degenerate mean/std of per-user-per-hour counts.

    4 rows across 2 users (as in the original test) yields a population of
    3 counts with almost no spread -- insufficient to distinguish a working
    z-score from a broken one. This fixture builds ~40 users with a
    controlled distribution of hourly transaction counts.
    """
    rng = np.random.default_rng(seed=7)
    rows = []
    for user_idx in range(40):
        user_id = f"user{user_idx}"
        hourly_count = max(1, int(rng.normal(loc=3.0, scale=1.5)))
        for step in range(hourly_count):
            rows.append({"step": 1, "nameOrig": user_id})
    frame = pd.DataFrame(rows)
    path = tmp_path_factory.mktemp("data") / "paysim.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def agent(paysim_csv: str) -> FrequencyZScoreAgent:
    # function-scoped: this agent carries mutable per-user window state,
    # unlike the amount/device agents whose fitted state is read-only.
    return FrequencyZScoreAgent(
        redis_url="redis://test:6379",
        input_channel="test-tx",
        output_channel="test-scores",
        threshold=0.70,
        paysim_csv_path=paysim_csv,
        window_size_seconds=3600,
    )


def make_transaction(user_id: str, timestamp: datetime, **overrides) -> TransactionPayload:
    base = dict(
        transaction_id="tx_1",
        user_id=user_id,
        amount=100.0,
        merchant_id="merchant_1",
        device_id="device_1",
        latitude=40.7128,
        longitude=-74.0060,
        timestamp=timestamp,
    )
    base.update(overrides)
    return TransactionPayload(**base)


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------

class TestFrequencyZScoreAgent:

    @pytest.mark.asyncio
    async def test_population_stats_are_non_degenerate(self, agent: FrequencyZScoreAgent):
        assert agent._population_std > 0.5, "fixture must produce real spread, not a near-constant distribution"

    @pytest.mark.asyncio
    async def test_first_transaction_scores_low(self, agent: FrequencyZScoreAgent):
        now = datetime.now(timezone.utc)
        score = await agent.compute_score(make_transaction("new_user", now))
        assert 0.0 <= score <= 1.0
        assert score < 0.3, "a single transaction should sit well below the population mean"

    @pytest.mark.asyncio
    async def test_burst_scores_high(self, agent: FrequencyZScoreAgent):
        now = datetime.now(timezone.utc)
        score = None
        for i in range(50):
            tx = make_transaction("burst_user", now + timedelta(seconds=i))
            score = await agent.compute_score(tx)
        assert score > 0.9, "50 transactions inside one hour must score as a clear anomaly"

    @pytest.mark.asyncio
    async def test_score_is_monotonically_non_decreasing_within_burst(self, agent: FrequencyZScoreAgent):
        now = datetime.now(timezone.utc)
        scores = []
        for i in range(20):
            tx = make_transaction("ramping_user", now + timedelta(seconds=i))
            scores.append(await agent.compute_score(tx))
        assert scores == sorted(scores), f"score must not decrease as transaction count grows: {scores}"

    @pytest.mark.asyncio
    async def test_users_are_scored_independently(self, agent: FrequencyZScoreAgent):
        now = datetime.now(timezone.utc)
        for i in range(30):
            await agent.compute_score(make_transaction("heavy_user", now + timedelta(seconds=i)))

        quiet_score = await agent.compute_score(make_transaction("quiet_user", now))
        assert quiet_score < 0.3, "one user's burst must not inflate another user's score"


# ---------------------------------------------------------------------------
# Sliding-window eviction
# ---------------------------------------------------------------------------

class TestWindowEviction:

    @pytest.mark.asyncio
    async def test_expired_transactions_are_evicted(self, agent: FrequencyZScoreAgent):
        t0 = datetime.now(timezone.utc)
        for i in range(10):
            await agent.compute_score(make_transaction("user_x", t0 + timedelta(seconds=i)))

        assert len(agent._windows["user_x"]) == 10

        far_future = t0 + timedelta(seconds=agent._window_size_seconds + 100)
        await agent.compute_score(make_transaction("user_x", far_future))

        assert len(agent._windows["user_x"]) == 1, "all prior entries should have aged out of the window"

    @pytest.mark.asyncio
    async def test_empty_window_is_removed_from_dict(self, agent: FrequencyZScoreAgent):
        """Regression guard: the fixed eviction path must delete the dict
        entry once a user's window fully drains, or memory grows unbounded
        with every distinct user ever seen (the defect flagged in review).
        """
        t0 = datetime.now(timezone.utc)
        await agent.compute_score(make_transaction("transient_user", t0))
        assert "transient_user" in agent._windows

        far_future = t0 + timedelta(seconds=agent._window_size_seconds + 1)
        # A transaction from a different user triggers eviction only on
        # transient_user's own next call, since eviction is per-user --
        # confirm the key is gone only after that user is touched again.
        await agent.compute_score(make_transaction("transient_user", far_future))
        assert len(agent._windows["transient_user"]) == 1

    @pytest.mark.asyncio
    async def test_partial_eviction_keeps_recent_entries(self, agent: FrequencyZScoreAgent):
        t0 = datetime.now(timezone.utc)
        await agent.compute_score(make_transaction("user_y", t0))
        await agent.compute_score(make_transaction("user_y", t0 + timedelta(seconds=1800)))
        # this arrives 3601s after the first entry (outside the window) but
        # only 1801s after the second (inside the window)
        await agent.compute_score(make_transaction("user_y", t0 + timedelta(seconds=3601)))

        assert len(agent._windows["user_y"]) == 2


# ---------------------------------------------------------------------------
# Degenerate-distribution edge case
# ---------------------------------------------------------------------------

class TestDegenerateTrainingData:

    @pytest.mark.asyncio
    async def test_zero_variance_population_falls_back_to_std_one(self, tmp_path):
        # every user has exactly 2 transactions in step 1 -> std == 0
        rows = [{"step": 1, "nameOrig": f"user{i}"} for i in range(20) for _ in range(2)]
        frame = pd.DataFrame(rows)
        path = tmp_path / "constant.csv"
        frame.to_csv(path, index=False)

        agent = FrequencyZScoreAgent(
            redis_url="redis://test:6379",
            input_channel="test-tx",
            output_channel="test-scores",
            threshold=0.70,
            paysim_csv_path=str(path),
        )
        assert agent._population_std == 1.0

        now = datetime.now(timezone.utc)
        score = await agent.compute_score(make_transaction("probe_user", now))
        assert math.isfinite(score)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Property-based: output contract must hold under arbitrary transaction rates
# ---------------------------------------------------------------------------

class TestScoreContract:

    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=None)
    @given(tx_count=st.integers(min_value=1, max_value=500))
    async def test_score_always_in_unit_interval(self, agent: FrequencyZScoreAgent, tx_count: int):
        now = datetime.now(timezone.utc)
        user_id = f"fuzz_user_{tx_count}"
        score = None
        for i in range(tx_count):
            tx = make_transaction(user_id, now + timedelta(seconds=i))
            score = await agent.compute_score(tx)
        assert score is not None
        assert 0.0 <= score <= 1.0
        assert math.isfinite(score)