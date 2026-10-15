"""Job-candidate pair features.

The same function builds features for training (from parquet) and for serving
(from API payloads), so there is no training/serving skew to debug later.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rankops.data import SKILL_FAMILIES

FEATURES = [
    "req_coverage",        # share of required skills the candidate has
    "skill_jaccard",
    "n_matched",
    "n_extra_skills",      # skills the candidate has that the job didn't ask for
    "cand_genai_skills",
    "cand_cloud_skills",
    "cand_ml_skills",
    "job_needs_genai",
    "years_experience",
    "exp_gap",             # candidate years - job minimum
    "location_match",
    "remote",
    "salary_ratio",        # expected / offered max
    "education_level",
]
EDU_LEVEL = {"diploma": 0, "bachelors": 1, "masters": 2, "phd": 3}


def _split(s: str) -> set[str]:
    return {x for x in str(s).split(",") if x}


def pair_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """`pairs` has one row per (job, candidate) with both sides' raw columns.

    Expected job columns: required_skills, min_experience, job_location, remote, salary_max_lpa
    Expected candidate columns: skills, years_experience, cand_location, education, expected_salary_lpa
    """
    req = pairs["required_skills"].map(_split)
    have = pairs["skills"].map(_split)
    matched = [len(r & h) for r, h in zip(req, have, strict=True)]
    union = [len(r | h) for r, h in zip(req, have, strict=True)]
    fam = {k: set(v) for k, v in SKILL_FAMILIES.items()}

    out = pd.DataFrame(index=pairs.index)
    out["req_coverage"] = [m / max(len(r), 1) for m, r in zip(matched, req, strict=True)]
    out["skill_jaccard"] = [m / max(u, 1) for m, u in zip(matched, union, strict=True)]
    out["n_matched"] = matched
    out["n_extra_skills"] = [len(h - r) for r, h in zip(req, have, strict=True)]
    out["cand_genai_skills"] = [len(h & fam["genai"]) for h in have]
    out["cand_cloud_skills"] = [len(h & fam["cloud"]) for h in have]
    out["cand_ml_skills"] = [len(h & fam["ml"]) for h in have]
    out["job_needs_genai"] = [int(bool(r & fam["genai"])) for r in req]
    out["years_experience"] = pairs["years_experience"].astype(float)
    out["exp_gap"] = pairs["years_experience"].astype(float) - pairs["min_experience"].astype(float)
    out["location_match"] = (pairs["cand_location"] == pairs["job_location"]).astype(int)
    out["remote"] = pairs["remote"].astype(int)
    out["salary_ratio"] = pairs["expected_salary_lpa"] / pairs["salary_max_lpa"].clip(lower=1.0)
    out["education_level"] = pairs["education"].map(EDU_LEVEL).fillna(1).astype(int)
    return out[FEATURES].astype(float)


def build_training_frame(jobs: pd.DataFrame, cands: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    """Join the three raw tables and return features + ids + label, sorted by job (LightGBM needs groups contiguous)."""
    df = (
        apps.merge(jobs.rename(columns={"location": "job_location", "period": "job_period"}), on="job_id")
        .merge(cands.rename(columns={"location": "cand_location"}), on="candidate_id")
        .sort_values(["job_id", "candidate_id"])
        .reset_index(drop=True)
    )
    feats = pair_features(df)
    return pd.concat([df[["job_id", "candidate_id", "period", "relevance"]], feats], axis=1)


def group_sizes(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby("job_id", sort=False).size().to_numpy()
