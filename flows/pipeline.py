"""Prefect orchestration.

retrain_flow  : ingest -> validate -> train -> promote gate -> tell serving to reload
monitor_flow  : drift check on logged traffic; kicks off retrain_flow when drift is found
                or when the scheduled cadence is due

Run once:        python flows/pipeline.py retrain --period 3
                 python flows/pipeline.py monitor --period 4
Serve on cron:   python flows/pipeline.py serve      (monitor daily at 02:00 IST)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
from prefect import flow, get_run_logger, task

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rankops import config  # noqa: E402
from rankops.data import available_periods, load_period, write_period  # noqa: E402
from rankops.monitor import drift_check  # noqa: E402
from rankops.promote import promote  # noqa: E402
from rankops.train import train  # noqa: E402
from rankops.validate import validate_period  # noqa: E402


@task(retries=2, retry_delay_seconds=10)
def ingest(period: int) -> int:
    """Swap this for a pull from Hirer's database (applications + recruiter feedback)."""
    if period not in available_periods():
        write_period(period)
    return period


@task
def validate(period: int) -> None:
    validate_period(*load_period(period))


@task
def train_challenger(period: int) -> dict:
    r = train(period)
    return {"version": r.version, "metrics": r.metrics}


@task
def gate(period: int) -> dict:
    return promote(period)


@task(retries=3, retry_delay_seconds=5)
def reload_serving() -> dict | None:
    if not config.SERVING_URL:
        return None
    r = httpx.post(f"{config.SERVING_URL.rstrip('/')}/reload", timeout=30)
    r.raise_for_status()
    return r.json()


@flow(name="hirer-retrain")
def retrain_flow(period: int, reason: str = "manual") -> dict:
    log = get_run_logger()
    log.info(f"retraining through period {period} (reason: {reason})")
    window = config.params()["train"]["window_periods"]
    for p in range(max(0, period - window + 1), period + 1):
        ingest(p)
        validate(p)
    challenger = train_challenger(period)
    decision = gate(period)
    # Reload even when rejected: the new challenger becomes the shadow model.
    loaded = reload_serving()
    log.info(f"decision: {decision['reason']}; serving now {loaded}")
    return {"reason": reason, "challenger": challenger, "decision": decision}


@flow(name="hirer-monitor")
def monitor_flow(period: int | None = None) -> dict:
    """`period` = the newest labelled batch. Defaults to the latest one ingested."""
    log = get_run_logger()
    if period is None:
        period = available_periods()[-1]
    drift = drift_check(tag=f"p{period}")
    every = config.params()["monitoring"]["scheduled_retrain_every"]
    scheduled = every > 0 and period % every == 0
    log.info(f"drift share {drift.get('drift_share')} (detected={drift['drift_detected']}), scheduled={scheduled}")

    result = {"period": period, "drift": drift, "retrain": None}
    if drift["drift_detected"] or scheduled:
        result["retrain"] = retrain_flow(period, reason="drift" if drift["drift_detected"] else "schedule")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["retrain", "monitor", "serve"])
    ap.add_argument("--period", type=int, default=None)
    a = ap.parse_args()
    period = a.period if a.period is not None else (available_periods() or [0])[-1]
    if a.command == "retrain":
        retrain_flow(period)
    elif a.command == "monitor":
        monitor_flow(period)
    else:
        # Long-running process that executes the monitor on a cron schedule.
        monitor_flow.serve(name="daily-monitor", cron="0 2 * * *", timezone="Asia/Kolkata")
