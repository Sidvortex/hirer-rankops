import numpy as np
import pandas as pd
import pytest

from rankops.metrics import ndcg_at_k, ranking_report, top_k_overlap
from rankops.promote import decide

RULES = {"primary_metric": "ndcg_at_10", "min_improvement": 0.005,
         "guardrail_metric": "ndcg_at_5", "max_guardrail_drop": 0.01}


def test_ndcg_perfect_and_reversed():
    y = np.array([3, 2, 1, 0])
    assert ndcg_at_k(y, np.array([4, 3, 2, 1]), 4) == pytest.approx(1.0)
    assert ndcg_at_k(y, np.array([1, 2, 3, 4]), 4) < 0.7


def test_ranking_report_averages_per_job():
    df = pd.DataFrame({"job_id": ["a"] * 3 + ["b"] * 3, "relevance": [2, 1, 0, 0, 1, 2]})
    r = ranking_report(df, np.array([3, 2, 1, 3, 2, 1]))
    assert 0 < r["ndcg_at_5"] < 1


def test_top_k_overlap():
    assert top_k_overlap(list("abcde"), list("abcde"), 5) == 1.0
    assert top_k_overlap(list("abcde"), list("vwxyz"), 5) == 0.0


def test_gate_promotes_first_model():
    assert decide(None, {"ndcg_at_10": 0.5, "ndcg_at_5": 0.5}, RULES)[0]


def test_gate_rejects_tiny_gain():
    ok, why = decide({"ndcg_at_10": 0.90, "ndcg_at_5": 0.88}, {"ndcg_at_10": 0.902, "ndcg_at_5": 0.89}, RULES)
    assert not ok and "gain" in why


def test_gate_rejects_guardrail_regression():
    ok, why = decide({"ndcg_at_10": 0.90, "ndcg_at_5": 0.88}, {"ndcg_at_10": 0.92, "ndcg_at_5": 0.85}, RULES)
    assert not ok and "dropped" in why


def test_gate_accepts_clear_win():
    assert decide({"ndcg_at_10": 0.80, "ndcg_at_5": 0.78}, {"ndcg_at_10": 0.85, "ndcg_at_5": 0.83}, RULES)[0]
