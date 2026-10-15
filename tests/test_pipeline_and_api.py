"""Integration: train -> promote -> serve -> log -> monitor, against a temp MLflow registry."""
import pytest
from fastapi.testclient import TestClient

from rankops import config
from rankops.data import write_period


@pytest.fixture(scope="module")
def trained():
    from rankops.promote import promote
    from rankops.train import train

    for p in range(3):
        write_period(p)
    result = train(2)
    decision = promote(2)
    return result, decision


def test_first_model_becomes_champion(trained):
    result, decision = trained
    assert decision["promoted"] and decision["live_version"] == result.version
    assert config.REFERENCE_PATH.exists()


def test_same_model_twice_is_not_promoted_again(trained):
    from rankops.promote import promote
    assert promote(2)["promoted"] is False


def test_api_ranks_and_logs(trained):
    from rankops.replay import payloads
    from rankops.serve import app

    config.REQUEST_LOG.unlink(missing_ok=True)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        body = next(payloads(2, limit=1))
        r = client.post("/rank", json=body)
        assert r.status_code == 200
        ranking = r.json()["ranking"]
        assert len(ranking) == 10 and ranking[0]["rank"] == 1
        assert ranking[0]["score"] >= ranking[-1]["score"]
        assert "rankops_rank_requests_total" in client.get("/metrics").text
        assert client.post("/rank", json={"job": body["job"], "candidates": []}).status_code == 422
    assert config.REQUEST_LOG.exists()


def test_drift_check_flags_shifted_traffic(trained):
    import pandas as pd

    from rankops.data import generate_period
    from rankops.features import FEATURES, build_training_frame
    from rankops.monitor import drift_check

    ref = pd.read_parquet(config.REFERENCE_PATH)
    same = build_training_frame(*generate_period(2, seed=99))[FEATURES]
    shifted = build_training_frame(*generate_period(8))[FEATURES]
    assert drift_check(current=shifted, reference=ref, tag="test_shift")["drift_detected"]
    assert not drift_check(current=same, reference=ref, tag="test_same")["drift_detected"]
