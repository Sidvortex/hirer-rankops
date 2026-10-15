"""Drift monitoring on live traffic.

Serving logs the feature vector of every candidate it scores. This module
compares the most recent window of those vectors with the champion's training
features (the reference) using Evidently, and says whether to retrain.

Labels arrive late in hiring (interview outcomes take weeks), so input drift
is the earliest signal we get. The promotion gate still checks ranking quality
on labelled data before anything goes live.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from rankops import config
from rankops.features import FEATURES


def load_logged_features(last_n: int | None = None) -> pd.DataFrame:
    if not config.REQUEST_LOG.exists():
        return pd.DataFrame(columns=FEATURES)
    rows = []
    with open(config.REQUEST_LOG) as f:
        for line in f:
            rows.extend(json.loads(line)["features"])
    df = pd.DataFrame(rows, columns=FEATURES)
    return df.tail(last_n) if last_n else df


def shadow_summary() -> dict:
    if not config.REQUEST_LOG.exists():
        return {}
    overlaps = []
    with open(config.REQUEST_LOG) as f:
        for line in f:
            r = json.loads(line)
            if r.get("shadow_overlap_at_10") is not None:
                overlaps.append(r["shadow_overlap_at_10"])
    if not overlaps:
        return {}
    s = pd.Series(overlaps)
    return {"requests": len(s), "mean_top10_overlap": round(s.mean(), 3), "p10_top10_overlap": round(s.quantile(0.1), 3)}


def drift_check(current: pd.DataFrame | None = None, reference: pd.DataFrame | None = None,
                window: int = 3000, tag: str = "latest") -> dict:
    mp = config.params()["monitoring"]
    reference = reference if reference is not None else pd.read_parquet(config.REFERENCE_PATH)
    current = current if current is not None else load_logged_features(window)

    if len(current) < mp["min_requests"]:
        return {"status": "insufficient_data", "rows": len(current), "drift_detected": False}

    snap = Report([DataDriftPreset(drift_share=mp["drift_share_threshold"])]).run(
        current_data=current[FEATURES], reference_data=reference[FEATURES]
    )
    metrics = snap.dict()["metrics"]
    summary = next(m for m in metrics if m["metric_name"].startswith("DriftedColumnsCount"))
    per_column = {
        m["config"]["column"]: round(float(m["value"]), 4)
        for m in metrics if m["metric_name"].startswith("ValueDrift")
    }
    thresholds = {
        m["config"]["column"]: m["config"].get("threshold")
        for m in metrics if m["metric_name"].startswith("ValueDrift")
    }
    drifted = sorted(c for c, v in per_column.items() if thresholds[c] is not None and v >= thresholds[c])
    share = float(summary["value"]["share"])

    result = {
        "status": "ok",
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": len(current),
        "drift_share": round(share, 3),
        "threshold": mp["drift_share_threshold"],
        "drift_detected": share >= mp["drift_share_threshold"],
        "drifted_columns": drifted,
        "column_scores": per_column,
        "shadow": shadow_summary(),
    }
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    snap.save_html(str(config.REPORT_DIR / f"drift_{tag}.html"))
    (config.REPORT_DIR / f"drift_{tag}.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=3000, help="most recent logged candidates to check")
    r = drift_check(window=ap.parse_args().window)
    print(json.dumps({k: v for k, v in r.items() if k != "column_scores"}, indent=2))
