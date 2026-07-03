
import pytest
from aggregator.autogen_groupchat_aggregator import compute_weighted_score


def test_compute_weighted_score(sample_fraud_scores, sample_legit_scores):
    # Test fraud score calculation
    fraud_score = compute_weighted_score(sample_fraud_scores)
    expected_fraud = (
        sample_fraud_scores[0].score * 0.20
        + sample_fraud_scores[1].score * 0.25
        + sample_fraud_scores[2].score * 0.30
        + sample_fraud_scores[3].score * 0.25
    )
    assert abs(fraud_score - expected_fraud) < 0.001

    # Test legit score calculation
    legit_score = compute_weighted_score(sample_legit_scores)
    expected_legit = (
        sample_legit_scores[0].score * 0.20
        + sample_legit_scores[1].score * 0.25
        + sample_legit_scores[2].score * 0.30
        + sample_legit_scores[3].score * 0.25
    )
    assert abs(legit_score - expected_legit) < 0.001
