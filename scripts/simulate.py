"""End-to-end demo of the closed loop over a simulated market shift.

For each new period:
  1. ingest the period's data
  2. score how well the *live* champion ranks it (what users would experience)
  3. replay the period's jobs through the API as live traffic (champion + shadow)
  4. run the monitor flow: drift check -> retrain + promotion gate when needed
  5. hot-reload the API

Writes reports/timeline.csv and reports/timeline.png.

    python scripts/simulate.py --fresh            # periods 0..9, shift at 4
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from rankops import config  # noqa: E402


def reset() -> None:
    for p in ("mlflow.db", "mlartifacts", "data", "logs", "reports"):
        path = config.ROOT / p
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)


def live_quality(period: int) -> dict:
    """Champion's ranking quality on every labelled job of this period."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from rankops.data import load_period
    from rankops.features import FEATURES, build_training_frame
    from rankops.metrics import ranking_report

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    name = config.params()["registry"]["model_name"]
    v = MlflowClient().get_model_version_by_alias(name, "champion")
    frame = build_training_frame(*load_period(period))
    model = mlflow.lightgbm.load_model(f"models:/{name}/{v.version}")
    return {"live_version": v.version, **ranking_report(frame, model.predict(frame[FEATURES]))}


def plot(df: pd.DataFrame, path: Path, drift_start: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    ax.axvspan(drift_start - 0.5, df.period.max() + 0.5, color="#f4e3c1", alpha=0.6, lw=0)
    ax.plot(df.period, df.live_ndcg_at_10, marker="o", color="#2f5d8a", lw=2,
            label="Live champion, scored on each new period")
    ax.plot(df.period, df.frozen_ndcg_at_10, marker="o", color="#b5523b", lw=1.6, ls="--",
            label="Never-retrained model")
    top = max(df.live_ndcg_at_10.max(), df.frozen_ndcg_at_10.max()) + 0.02
    ax.text(drift_start - 0.42, top, "market shift begins", fontsize=8.5, color="#8a6d2f", va="top")
    # A model promoted after period p serves period p+1: mark the hand-over between them.
    for _, r in df[df.promoted == True].iterrows():  # noqa: E712
        ax.axvline(r.period + 0.5, color="#2f5d8a", lw=0.8, ls=":", alpha=0.7)
        ax.text(r.period + 0.5, top - 0.03, f"v{r.new_version}", fontsize=8, color="#2f5d8a",
                ha="center", va="top", bbox=dict(fc="white", ec="none", pad=1))
    ax.set_ylim(top=top + 0.005)
    ax.set_xlabel("Period")
    ax.set_ylabel("NDCG@10")
    ax.set_title("Continuous retraining under market drift", loc="left", fontsize=11)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="wipe registry, data, logs and reports first")
    ap.add_argument("--last", type=int, default=9)
    a = ap.parse_args()
    if a.fresh:
        reset()
    config.ensure_dirs()
    logging.getLogger("mlflow").setLevel(logging.ERROR)

    import mlflow

    from flows.pipeline import monitor_flow, retrain_flow
    from rankops.data import load_period, write_period
    from rankops.features import FEATURES, build_training_frame
    from rankops.metrics import ranking_report
    from rankops.replay import replay
    from rankops.serve import app

    window = config.params()["train"]["window_periods"]
    first = window - 1
    for p in range(first + 1):
        write_period(p)
    boot = retrain_flow(first, reason="bootstrap")
    frozen_version = boot["decision"]["live_version"]
    frozen = mlflow.lightgbm.load_model(f"models:/{config.params()['registry']['model_name']}/{frozen_version}")

    rows = []
    with TestClient(app) as client:
        for p in range(first + 1, a.last + 1):
            write_period(p)
            q = live_quality(p)
            frame = build_training_frame(*load_period(p))
            frozen_q = ranking_report(frame, frozen.predict(frame[FEATURES]))

            config.REQUEST_LOG.unlink(missing_ok=True)  # one period of traffic per drift check
            sent = replay(p, client)
            result = monitor_flow(p)
            client.post("/reload").raise_for_status()

            d, rt = result["drift"], result["retrain"]
            dec = rt["decision"] if rt else None
            rows.append({
                "period": p,
                "requests": sent,
                "live_version": q["live_version"],
                "live_ndcg_at_10": q["ndcg_at_10"],
                "frozen_ndcg_at_10": frozen_q["ndcg_at_10"],
                "drift_share": d.get("drift_share"),
                "drift_detected": d["drift_detected"],
                "shadow_overlap": (d.get("shadow") or {}).get("mean_top10_overlap"),
                "retrain_reason": rt["reason"] if rt else None,
                "challenger_ndcg_at_10": dec["challenger_metrics"]["ndcg_at_10"] if dec else None,
                "champion_ndcg_at_10": (dec["champion_metrics"] or {}).get("ndcg_at_10") if dec else None,
                "promoted": bool(dec and dec["promoted"]),
                "new_version": dec["live_version"] if dec else q["live_version"],
                "decision": dec["reason"] if dec else "",
            })

    df = pd.DataFrame(rows)
    df.to_csv(config.REPORT_DIR / "timeline.csv", index=False)
    plot(df, config.REPORT_DIR / "timeline.png", config.params()["data"]["drift_start_period"])
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[["period", "live_version", "live_ndcg_at_10", "frozen_ndcg_at_10", "drift_share",
                  "shadow_overlap", "retrain_reason", "promoted", "new_version", "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
