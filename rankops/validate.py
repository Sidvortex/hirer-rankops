"""Data contracts. A period that fails validation never reaches training."""
from __future__ import annotations

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

from rankops.data import CITIES, EDUCATION, ROLES

JOBS = DataFrameSchema(
    {
        "job_id": Column(str, unique=True),
        "period": Column(int, Check.ge(0)),
        "role": Column(str, Check.isin(list(ROLES))),
        "required_skills": Column(str, Check.str_length(min_value=1)),
        "min_experience": Column(int, Check.in_range(0, 40)),
        "location": Column(str, Check.isin(CITIES)),
        "remote": Column(bool),
        "salary_max_lpa": Column(float, Check.in_range(0, 500)),
    },
    strict=True,
)

CANDIDATES = DataFrameSchema(
    {
        "candidate_id": Column(str, unique=True),
        "skills": Column(str, nullable=False),
        "years_experience": Column(float, Check.in_range(0, 50)),
        "location": Column(str, Check.isin(CITIES)),
        "education": Column(str, Check.isin(EDUCATION)),
        "expected_salary_lpa": Column(float, Check.in_range(0, 500)),
    },
    strict=True,
)

APPLICATIONS = DataFrameSchema(
    {
        "job_id": Column(str),
        "candidate_id": Column(str, unique=True),
        "period": Column(int),
        "relevance": Column(int, Check.isin([0, 1, 2, 3, 4])),
    },
    strict=True,
)


def validate_period(jobs: pd.DataFrame, cands: pd.DataFrame, apps: pd.DataFrame) -> None:
    """Raise pandera.errors.SchemaErrors (lazy: all failures at once) or ValueError."""
    JOBS.validate(jobs, lazy=True)
    CANDIDATES.validate(cands, lazy=True)
    APPLICATIONS.validate(apps, lazy=True)

    # Cross-table checks pandera can't express per-frame.
    missing_jobs = set(apps.job_id) - set(jobs.job_id)
    missing_cands = set(apps.candidate_id) - set(cands.candidate_id)
    if missing_jobs or missing_cands:
        raise ValueError(f"dangling references: {len(missing_jobs)} jobs, {len(missing_cands)} candidates")

    # A ranking query with one grade carries no signal for LambdaRank.
    informative = apps.groupby("job_id").relevance.nunique().gt(1).mean()
    if informative < 0.8:
        raise ValueError(f"only {informative:.0%} of jobs have more than one relevance grade")
