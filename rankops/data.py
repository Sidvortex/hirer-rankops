"""Synthetic hiring data, released in time periods, with a controllable market shift.

Each period is one "batch" of new jobs, the candidates who applied, and recruiter
relevance grades (0-4). Starting at `drift_start_period` the market shifts:

* data drift    - more jobs ask for GenAI skills, more candidates list them,
                  remote roles become common, candidates skew less experienced.
* concept drift - recruiters start rewarding GenAI skills even when the posting
                  doesn't list them, and care less about raw years of experience.

A model trained before the shift keeps ranking the old way, so its NDCG falls on
new periods. That is what the retraining loop has to notice and fix.

To use real Hirer data instead, write the same three tables
(jobs / candidates / applications) into data/raw/period_XX/ and skip `generate`.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from rankops.config import RAW_DIR, ensure_dirs, params

SKILL_FAMILIES: dict[str, list[str]] = {
    "core": ["python", "sql", "git", "linux"],
    "web": ["react", "nodejs", "django", "fastapi", "typescript"],
    "cloud": ["aws", "docker", "kubernetes", "terraform"],
    "data": ["pandas", "spark", "airflow", "dbt"],
    "ml": ["pytorch", "scikit-learn", "tensorflow", "mlflow"],
    "genai": ["llm", "rag", "langchain", "vector-db", "prompt-eng"],
}
ALL_SKILLS = [s for fam in SKILL_FAMILIES.values() for s in fam]
SKILL_TO_FAMILY = {s: fam for fam, skills in SKILL_FAMILIES.items() for s in skills}

ROLES: dict[str, list[str]] = {
    "backend": ["core", "web", "cloud"],
    "frontend": ["core", "web"],
    "data_eng": ["core", "data", "cloud"],
    "ml_eng": ["core", "ml", "data"],
    "devops": ["core", "cloud"],
}
CITIES = ["bengaluru", "hyderabad", "pune", "noida", "gurugram", "chennai", "mumbai"]
EDUCATION = ["diploma", "bachelors", "masters", "phd"]


def drift_amount(period: int, start: int, strength: float) -> float:
    """0 before the shift, ramps to `strength` over two periods."""
    return float(strength * np.clip((period - start + 1) / 2, 0.0, 1.0))


def _sample_skills(rng, families: list[str], n: int, genai_p: float) -> list[str]:
    pool = [s for f in families for s in SKILL_FAMILIES[f]]
    picked = set(rng.choice(pool, size=min(n, len(pool)), replace=False))
    if rng.random() < genai_p:
        picked |= set(rng.choice(SKILL_FAMILIES["genai"], size=rng.integers(1, 4), replace=False))
    return sorted(picked)


def generate_period(period: int, cfg: dict | None = None, seed: int | None = None):
    cfg = cfg or params()["data"]
    rng = np.random.default_rng(seed if seed is not None else 1000 + period)
    d = drift_amount(period, cfg["drift_start_period"], cfg["drift_strength"])

    jobs, cands, apps = [], [], []
    for j in range(cfg["jobs_per_period"]):
        job_id = f"p{period:02d}_j{j:04d}"
        role = rng.choice(list(ROLES))
        families = ROLES[role]
        job_skills = _sample_skills(rng, families, int(rng.integers(3, 7)), genai_p=0.08 + 0.55 * d)
        min_exp = int(rng.integers(0, 9))
        remote = bool(rng.random() < 0.2 + 0.5 * d)
        job_city = rng.choice(CITIES)
        salary_max = float(6 + 2.2 * min_exp + rng.normal(0, 2))
        jobs.append(
            dict(job_id=job_id, period=period, role=role, required_skills=",".join(job_skills),
                 min_experience=min_exp, location=job_city, remote=remote, salary_max_lpa=round(salary_max, 1))
        )

        for c in range(cfg["candidates_per_job"]):
            cand_id = f"{job_id}_c{c:03d}"
            fit = rng.random()  # how close this applicant's background is to the role
            fams = families if fit > 0.35 else list(rng.choice(list(SKILL_FAMILIES), 2, replace=False))
            skills = _sample_skills(rng, fams, int(rng.integers(2, 9)), genai_p=0.10 + 0.45 * d)
            years = float(np.clip(rng.gamma(2.2 - 0.8 * d, 2.4), 0, 25))
            city = job_city if rng.random() < 0.45 else rng.choice(CITIES)
            edu = rng.choice(EDUCATION, p=[0.1, 0.6, 0.27, 0.03])
            expected = float(max(3, 5 + 2.0 * years + rng.normal(0, 3)))
            cands.append(
                dict(candidate_id=cand_id, skills=",".join(skills), years_experience=round(years, 1),
                     location=city, education=edu, expected_salary_lpa=round(expected, 1))
            )

            # --- hidden recruiter judgement (the "truth" the model must learn) ---
            req, have = set(job_skills), set(skills)
            coverage = len(req & have) / len(req)
            genai_have = len(have & set(SKILL_FAMILIES["genai"]))
            exp_term = np.tanh((years - min_exp) / 3.0)
            loc_ok = 1.0 if (remote or city == job_city) else 0.0
            salary_ok = 1.0 if expected <= salary_max * 1.1 else -0.6
            edu_bonus = {"diploma": -0.2, "bachelors": 0.0, "masters": 0.15, "phd": 0.2}[edu]
            latent = (
                3.0 * coverage
                + (0.9 - 0.7 * d) * exp_term          # experience matters less after the shift
                + 0.6 * loc_ok
                + 0.5 * salary_ok
                + edu_bonus
                + d * 0.75 * genai_have               # GenAI skills rewarded after the shift
                + rng.normal(0, 0.45)
            )
            grade = int(np.digitize(latent, [1.4, 2.4, 3.3, 4.1]))
            apps.append(dict(job_id=job_id, candidate_id=cand_id, period=period, relevance=grade))

    return pd.DataFrame(jobs), pd.DataFrame(cands), pd.DataFrame(apps)


def write_period(period: int) -> None:
    ensure_dirs()
    out = RAW_DIR / f"period_{period}"
    out.mkdir(parents=True, exist_ok=True)
    jobs, cands, apps = generate_period(period)
    jobs.to_parquet(out / "jobs.parquet", index=False)
    cands.to_parquet(out / "candidates.parquet", index=False)
    apps.to_parquet(out / "applications.parquet", index=False)
    print(f"period {period}: {len(jobs)} jobs, {len(cands)} candidates -> {out}")


def available_periods() -> list[int]:
    return sorted(int(p.name.split("_")[1]) for p in RAW_DIR.glob("period_*") if (p / "applications.parquet").exists())


def load_period(period: int):
    src = RAW_DIR / f"period_{period}"
    return (pd.read_parquet(src / "jobs.parquet"), pd.read_parquet(src / "candidates.parquet"),
            pd.read_parquet(src / "applications.parquet"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate synthetic hiring periods")
    ap.add_argument("periods", nargs="+", type=int)
    for p in ap.parse_args().periods:
        write_period(p)
