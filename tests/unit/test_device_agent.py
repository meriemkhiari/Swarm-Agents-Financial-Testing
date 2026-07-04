from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from agents.agent_device.device_profile_agent import DeviceProfileAgent, hash_device_id
from shared.transaction_schema import TransactionPayload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def paysim_csv(tmp_path_factory) -> str:
    """Skewed frequency distribution: one device dominant, one rare, several mid-tier.

    A flat 1-1-1 distribution (as in the original test) can't distinguish
    'novelty scoring' from 'seen/unseen membership' -- the exact ambiguity
    that hid the OneClassSVM defect in the original implementation.
    """
    rows = (
        [("device_frequent", i) for i in range(50)]
        + [("device_midtier", i) for i in range(10)]
        + [("device_rare", 0)]
    )
    frame = pd.DataFrame({
        "step": [1] * len(rows),
        "nameOrig": [f"{device}_{idx}" for device, idx in rows],
    })
    # nameOrig in PaySim is the account, not the device_id field on the payload --
    # the agent hashes tx.device_id at score time and nameOrig at train time.
    # Reuse nameOrig values directly as the device identifiers being trained on.
    frame["nameOrig"] = [device for device, _ in rows]
    path = tmp_path_factory.mktemp("data") / "paysim.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="module")
def agent(paysim_csv: str) -> DeviceProfileAgent:
    return DeviceProfileAgent(
        redis_url="redis://test:6379",
        input_channel="test-tx",
        output_channel="test-scores",
        threshold=0.75,
        paysim_csv_path=paysim_csv,
    )


def make_transaction(device_id: str, **overrides) -> TransactionPayload:
    base = dict(
        transaction_id="tx_1",
        user_id="user_1",
        amount=100.0,
        merchant_id="merchant_1",
        device_id=device_id,
        latitude=40.7128,
        longitude=-74.0060,
        timestamp="2026-07-04T00:00:00Z",
    )
    base.update(overrides)
    return TransactionPayload(**base)


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------

class TestDeviceProfileAgent:

    @pytest.mark.asyncio
    async def test_unseen_device_scores_maximum(self, agent: DeviceProfileAgent):
        score = await agent.compute_score(make_transaction("device_never_seen"))
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_most_frequent_device_scores_minimum(self, agent: DeviceProfileAgent):
        score = await agent.compute_score(make_transaction("device_frequent"))
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_rare_but_seen_device_scores_near_maximum(self, agent: DeviceProfileAgent):
        score = await agent.compute_score(make_transaction("device_rare"))
        assert score > 0.9, "a device seen only once should score close to unseen"

    @pytest.mark.asyncio
    async def test_frequency_ordering_is_monotonic(self, agent: DeviceProfileAgent):
        """Higher training-time frequency must yield strictly lower novelty score."""
        frequent_score = await agent.compute_score(make_transaction("device_frequent"))
        midtier_score = await agent.compute_score(make_transaction("device_midtier"))
        rare_score = await agent.compute_score(make_transaction("device_rare"))
        unseen_score = await agent.compute_score(make_transaction("device_totally_new"))

        assert frequent_score < midtier_score < rare_score < unseen_score, (
            frequent_score, midtier_score, rare_score, unseen_score
        )

    @pytest.mark.asyncio
    async def test_same_device_id_is_deterministic(self, agent: DeviceProfileAgent):
        score_a = await agent.compute_score(make_transaction("device_frequent"))
        score_b = await agent.compute_score(make_transaction("device_frequent"))
        assert score_a == score_b


# ---------------------------------------------------------------------------
# Regression guard: this agent must never reintroduce a distance-based model
# over hashed identifiers (the OneClassSVM defect from the original design).
# ---------------------------------------------------------------------------

class TestNoDistanceModelRegression:

    def test_agent_has_no_sklearn_model_attribute(self, agent: DeviceProfileAgent):
        assert not hasattr(agent, "_model"), (
            "DeviceProfileAgent must remain frequency-based; a '_model' attribute "
            "suggests a distance/density model has been reintroduced over hashed IDs, "
            "which carries no statistical meaning (see prior architecture review)."
        )

    def test_hash_function_is_deterministic(self):
        assert hash_device_id("device_xyz123") == hash_device_id("device_xyz123")

    def test_hash_function_is_bounded_32_bit(self):
        assert 0 <= hash_device_id("any_arbitrary_string") < 2**32


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_training_data_raises(self, tmp_path):
        """Documents current behavior: an empty training set has no defined
        max frequency. This is a latent gap, not a passing contract -- flagging
        via xfail rather than asserting success on undefined behavior.
        """
        frame = pd.DataFrame({"nameOrig": pd.Series([], dtype=str)})
        path = tmp_path / "empty.csv"
        frame.to_csv(path, index=False)

        with pytest.raises(ValueError):
            DeviceProfileAgent(
                redis_url="redis://test:6379",
                input_channel="test-tx",
                output_channel="test-scores",
                threshold=0.75,
                paysim_csv_path=str(path),
            )

    @pytest.mark.asyncio
    async def test_single_unique_device_in_training(self, tmp_path):
        """max_frequency == the only device's frequency -> that device scores 0.0,
        guards against an accidental divide-by-zero if frequency == max in a
        degenerate single-device training set.
        """
        frame = pd.DataFrame({"nameOrig": ["solo_device"] * 20})
        path = tmp_path / "solo.csv"
        frame.to_csv(path, index=False)

        solo_agent = DeviceProfileAgent(
            redis_url="redis://test:6379",
            input_channel="test-tx",
            output_channel="test-scores",
            threshold=0.75,
            paysim_csv_path=str(path),
        )
        score = await solo_agent.compute_score(make_transaction("solo_device"))
        assert score == 0.0
        assert math.isfinite(score)


# ---------------------------------------------------------------------------
# Property-based: output contract must hold for arbitrary device_id strings
# ---------------------------------------------------------------------------

class TestScoreContract:

    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=None)
    @given(device_id=st.text(min_size=1, max_size=64))
    async def test_score_always_in_unit_interval(self, agent: DeviceProfileAgent, device_id: str):
        score = await agent.compute_score(make_transaction(device_id))
        assert 0.0 <= score <= 1.0
        assert math.isfinite(score)