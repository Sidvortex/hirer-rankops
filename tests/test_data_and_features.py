import numpy as np
import pandas as pd
import pytest
from pandera import errors as pa_errors

from rankops.data import generate_period
from rankops.features import FEATURES, build_training_frame, pair_features
from rankops.validate import validate_period


def test_generation_is_deterministic():
    a = generate_period(1)[2]
    b = generate_period(1)[2]
    pd.testing.assert_frame_equal(a, b)


def test_generated_periods_pass_validation():
    for p in (0, 6):
        validate_period(*generate_period(p))


def test_market_shift_moves_feature_distribution():
    before = build_training_frame(*generate_period(1))
    after = build_training_frame(*generate_period(7))
    assert after.cand_genai_skills.mean() > before.cand_genai_skills.mean() + 0.5
    assert after.remote.mean() > before.remote.mean()


def test_validation_rejects_bad_grades():
    jobs, cands, apps = generate_period(0)
    apps.loc[0, "relevance"] = 9
    with pytest.raises(pa_errors.SchemaErrors):
        validate_period(jobs, cands, apps)


def test_validation_rejects_dangling_candidate():
    jobs, cands, apps = generate_period(0)
    with pytest.raises(ValueError, match="dangling"):
        validate_period(jobs, cands.iloc[1:], apps)


def test_pair_features_known_values():
    pairs = pd.DataFrame([{
        "required_skills": "python,sql,llm", "min_experience": 2, "job_location": "noida",
        "remote": False, "salary_max_lpa": 20.0, "skills": "python,llm,rag,docker",
        "years_experience": 5.0, "cand_location": "noida", "education": "masters",
        "expected_salary_lpa": 10.0,
    }])
    f = pair_features(pairs).iloc[0]
    assert list(pair_features(pairs).columns) == FEATURES
    assert f.req_coverage == pytest.approx(2 / 3)
    assert f.skill_jaccard == pytest.approx(2 / 5)
    assert f.n_extra_skills == 2
    assert f.cand_genai_skills == 2 and f.job_needs_genai == 1
    assert f.exp_gap == 3 and f.location_match == 1 and f.salary_ratio == pytest.approx(0.5)


def test_no_training_serving_skew():
    """Features built from an API payload must equal the training features for the same pair."""
    from rankops.data import load_period, write_period
    from rankops.replay import payloads
    from rankops.serve import RankRequest, _to_pairs

    write_period(0)
    frame = build_training_frame(*load_period(0))
    body = next(payloads(0, limit=1))
    served = pair_features(_to_pairs(RankRequest(**body)))
    served.index = [c["candidate_id"] for c in body["candidates"]]
    trained = frame.set_index("candidate_id").loc[served.index, FEATURES]
    np.testing.assert_allclose(served.to_numpy(), trained.to_numpy(), rtol=1e-9)
