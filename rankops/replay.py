"""Turn a period's raw data into /rank requests, to replay as live traffic."""
from __future__ import annotations

import argparse

import httpx

from rankops.data import load_period


def payloads(period: int, limit: int | None = None):
    jobs, cands, apps = load_period(period)
    cands = cands.set_index("candidate_id")
    for _, job in jobs.head(limit).iterrows() if limit else jobs.iterrows():
        ids = apps.loc[apps.job_id == job.job_id, "candidate_id"]
        yield {
            "job": {
                "job_id": job.job_id, "required_skills": job.required_skills.split(","),
                "min_experience": int(job.min_experience), "location": job.location,
                "remote": bool(job.remote), "salary_max_lpa": float(job.salary_max_lpa),
            },
            "candidates": [
                {
                    "candidate_id": cid, "skills": [s for s in cands.at[cid, "skills"].split(",") if s],
                    "years_experience": float(cands.at[cid, "years_experience"]),
                    "location": cands.at[cid, "location"], "education": cands.at[cid, "education"],
                    "expected_salary_lpa": float(cands.at[cid, "expected_salary_lpa"]),
                }
                for cid in ids
            ],
            "top_k": 10,
        }


def replay(period: int, client, limit: int | None = None) -> int:
    """`client` is anything with .post(path, json=...): an httpx.Client or FastAPI TestClient."""
    n = 0
    for body in payloads(period, limit):
        client.post("/rank", json=body).raise_for_status()
        n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay a period's jobs against a running API")
    ap.add_argument("period", type=int)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    with httpx.Client(base_url=a.url, timeout=30) as c:
        print(f"sent {replay(a.period, c, a.limit)} requests")
