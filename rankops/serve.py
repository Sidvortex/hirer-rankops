"""Ranking API.

* Ranks with the `champion` model from the MLflow registry.
* If a different `challenger` exists, scores it in shadow mode: the response
  never uses it, but its agreement with the champion is logged and exported,
  so you see how a candidate model behaves on real traffic before promoting it.
* Logs every scored feature vector for drift monitoring.
* POST /reload hot-swaps models after a promotion or rollback, no restart.

Run:  uvicorn rankops.serve:app --port 8000
"""
from __future__ import annotations

import json
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from rankops import config
from rankops.features import FEATURES, pair_features
from rankops.metrics import top_k_overlap

REQUESTS = Counter("rankops_rank_requests_total", "Ranking requests", ["status"])
CANDIDATES = Counter("rankops_candidates_scored_total", "Candidates scored")
LATENCY = Histogram("rankops_rank_latency_seconds", "Ranking latency",
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0))
SHADOW_OVERLAP = Histogram("rankops_shadow_top10_overlap", "Champion vs shadow top-10 agreement",
                           buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
MODEL_VERSION = Gauge("rankops_model_version", "Loaded registry version", ["role"])


class Job(BaseModel):
    job_id: str
    required_skills: list[str] = Field(min_length=1)
    min_experience: int = Field(ge=0, le=40)
    location: str
    remote: bool = False
    salary_max_lpa: float = Field(gt=0)


class Candidate(BaseModel):
    candidate_id: str
    skills: list[str] = []
    years_experience: float = Field(ge=0, le=50)
    location: str
    education: str = "bachelors"
    expected_salary_lpa: float = Field(gt=0)


class RankRequest(BaseModel):
    job: Job
    candidates: list[Candidate] = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1)


class Ranked(BaseModel):
    candidate_id: str
    score: float
    rank: int


class RankResponse(BaseModel):
    job_id: str
    model_version: str
    ranking: list[Ranked]


class ModelStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.champion = self.shadow = None
        self.champion_version = self.shadow_version = None

    def _load(self, client: MlflowClient, name: str, alias: str):
        try:
            v = client.get_model_version_by_alias(name, alias)
        except MlflowException:
            return None, None
        return mlflow.lightgbm.load_model(f"models:/{name}/{v.version}"), str(v.version)

    def reload(self) -> dict:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        client, name = MlflowClient(), config.params()["registry"]["model_name"]
        champ, cv = self._load(client, name, "champion")
        if champ is None:
            raise RuntimeError(f"no '{name}@champion' in the registry; train and promote first")
        shadow, sv = self._load(client, name, "challenger")
        if sv == cv:
            shadow, sv = None, None
        with self.lock:
            self.champion, self.champion_version = champ, cv
            self.shadow, self.shadow_version = shadow, sv
        MODEL_VERSION.labels("champion").set(int(cv))
        MODEL_VERSION.labels("shadow").set(int(sv) if sv else 0)
        return {"champion": cv, "shadow": sv}


store = ModelStore()
_log_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.ensure_dirs()
    try:
        store.reload()
    except Exception as e:  # serve /health even before the first model exists
        print(f"[serve] no model loaded yet: {e}")
    yield


app = FastAPI(title="Hirer ranker", version="1.0", lifespan=lifespan)


def _to_pairs(req: RankRequest) -> pd.DataFrame:
    j = req.job
    return pd.DataFrame([
        {
            "required_skills": ",".join(s.lower() for s in j.required_skills),
            "min_experience": j.min_experience, "job_location": j.location.lower(),
            "remote": j.remote, "salary_max_lpa": j.salary_max_lpa,
            "skills": ",".join(s.lower() for s in c.skills),
            "years_experience": c.years_experience, "cand_location": c.location.lower(),
            "education": c.education.lower(), "expected_salary_lpa": c.expected_salary_lpa,
        }
        for c in req.candidates
    ])


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest) -> RankResponse:
    if store.champion is None:
        REQUESTS.labels("no_model").inc()
        raise HTTPException(503, "no champion model loaded")
    t0 = time.perf_counter()
    with store.lock:
        champ, cv, shadow, sv = store.champion, store.champion_version, store.shadow, store.shadow_version

    feats = pair_features(_to_pairs(req))
    ids = [c.candidate_id for c in req.candidates]
    scores = champ.predict(feats[FEATURES])
    order = sorted(range(len(ids)), key=lambda i: -scores[i])
    ranked_ids = [ids[i] for i in order]

    overlap = None
    if shadow is not None:
        s_scores = shadow.predict(feats[FEATURES])
        overlap = top_k_overlap(ranked_ids, [ids[i] for i in sorted(range(len(ids)), key=lambda i: -s_scores[i])])
        SHADOW_OVERLAP.observe(overlap)

    record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "job_id": req.job.job_id, "champion_version": cv, "shadow_version": sv,
        "shadow_overlap_at_10": overlap, "n_candidates": len(ids),
        "features": feats.round(4).values.tolist(),
    }
    with _log_lock, open(config.REQUEST_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    k = req.top_k or len(ids)
    LATENCY.observe(time.perf_counter() - t0)
    REQUESTS.labels("ok").inc()
    CANDIDATES.inc(len(ids))
    return RankResponse(
        job_id=req.job.job_id, model_version=cv,
        ranking=[Ranked(candidate_id=ids[i], score=round(float(scores[i]), 5), rank=r + 1)
                 for r, i in enumerate(order[:k])],
    )


@app.post("/reload")
def reload_models() -> dict:
    try:
        return store.reload()
    except Exception as e:
        raise HTTPException(503, str(e)) from e


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if store.champion is not None else "no_model",
            "champion": store.champion_version, "shadow": store.shadow_version}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
