"""Paths and settings shared by every stage."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

ROOT = Path(os.getenv("RANKOPS_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURE_DIR = DATA_DIR / "features"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"
REQUEST_LOG = LOG_DIR / "requests.jsonl"
REFERENCE_PATH = DATA_DIR / "reference" / "reference_features.parquet"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
EXPERIMENT_NAME = os.getenv("RANKOPS_EXPERIMENT", "hirer-ranker")
SERVING_URL = os.getenv("RANKOPS_SERVING_URL", "")  # e.g. http://localhost:8000; empty = don't notify


@lru_cache
def params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for d in (RAW_DIR, FEATURE_DIR, REPORT_DIR, LOG_DIR, REFERENCE_PATH.parent):
        d.mkdir(parents=True, exist_ok=True)
