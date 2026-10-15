"""Train a LambdaRank challenger on a sliding window of periods and register it in MLflow.

The new version gets the `challenger` alias. It only becomes `champion` if
`rankops.promote` decides it beats the current one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from rankops import config
from rankops.data import available_periods, load_period
from rankops.features import FEATURES, build_training_frame, group_sizes
from rankops.metrics import ranking_report
from rankops.validate import validate_period


@dataclass
class TrainResult:
    run_id: str
    version: str
    period: int
    metrics: dict
    holdout_path: str


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        # Local runs keep artifacts in the repo; against a tracking server the server decides.
        loc = (config.ROOT / "mlartifacts").as_uri() if config.MLFLOW_TRACKING_URI.startswith("sqlite") else None
        mlflow.create_experiment(config.EXPERIMENT_NAME, artifact_location=loc)
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def load_window(period: int, window: int) -> pd.DataFrame:
    have = set(available_periods())
    wanted = [p for p in range(period - window + 1, period + 1) if p >= 0]
    missing = [p for p in wanted if p not in have]
    if missing:
        raise FileNotFoundError(f"periods {missing} not ingested yet")
    frames = []
    for p in wanted:
        jobs, cands, apps = load_period(p)
        validate_period(jobs, cands, apps)
        frames.append(build_training_frame(jobs, cands, apps))
    return pd.concat(frames, ignore_index=True)


def split_holdout(frame: pd.DataFrame, period: int, frac: float, seed: int):
    """Hold out a share of the *newest* period's jobs. Evaluating on the newest data
    is what tells us whether a model fits the market as it is now."""
    newest_jobs = frame.loc[frame.period == period, "job_id"].unique()
    rng = np.random.default_rng(seed + period)
    hold = set(rng.choice(newest_jobs, size=max(1, int(len(newest_jobs) * frac)), replace=False))
    is_hold = frame.job_id.isin(hold)
    return frame[~is_hold].reset_index(drop=True), frame[is_hold].reset_index(drop=True)


def data_fingerprint(frame: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()[:12]


def train(period: int) -> TrainResult:
    config.ensure_dirs()
    p = config.params()
    tp = p["train"]
    setup_mlflow()

    frame = load_window(period, tp["window_periods"])
    train_df, hold_df = split_holdout(frame, period, tp["holdout_job_frac"], tp["seed"])

    model = lgb.LGBMRanker(**tp["lgbm"], random_state=tp["seed"], verbose=-1)
    model.fit(train_df[FEATURES], train_df.relevance, group=group_sizes(train_df))

    metrics = ranking_report(hold_df, model.predict(hold_df[FEATURES]))
    holdout_path = config.FEATURE_DIR / f"holdout_p{period}.parquet"
    hold_df.to_parquet(holdout_path, index=False)
    # Features the model learned from: the drift baseline if this version gets promoted.
    ref_path = config.FEATURE_DIR / f"train_features_p{period}.parquet"
    train_df[FEATURES].to_parquet(ref_path, index=False)

    name = p["registry"]["model_name"]
    with mlflow.start_run(run_name=f"train-p{period}") as run:
        mlflow.log_params({f"lgbm.{k}": v for k, v in tp["lgbm"].items() if k != "label_gain"})
        mlflow.log_params({
            "period": period, "window_periods": tp["window_periods"],
            "train_rows": len(train_df), "train_jobs": train_df.job_id.nunique(),
            "holdout_jobs": hold_df.job_id.nunique(), "data_fingerprint": data_fingerprint(frame),
        })
        mlflow.log_metrics(metrics)
        imp = dict(sorted(zip(FEATURES, model.booster_.feature_importance("gain").round(1).tolist(), strict=True),
                          key=lambda kv: -kv[1]))
        mlflow.log_dict(imp, "feature_importance.json")
        mlflow.log_artifact(str(ref_path), artifact_path="reference")
        # Log the native Booster: portable, small, and loads without pickle trust issues.
        mlflow.lightgbm.log_model(
            model.booster_, name="model",
            signature=infer_signature(hold_df[FEATURES].head(50), model.predict(hold_df[FEATURES].head(50))),
            input_example=hold_df[FEATURES].head(3),
            registered_model_name=name,
        )
        run_id = run.info.run_id

    client = MlflowClient()
    version = max(client.search_model_versions(f"name='{name}' and run_id='{run_id}'"),
                  key=lambda v: int(v.version)).version
    client.set_model_version_tag(name, version, "period", str(period))
    client.set_model_version_tag(name, version, "reference_features", str(ref_path.relative_to(config.ROOT)))
    client.set_registered_model_alias(name, "challenger", version)

    print(f"trained v{version} on periods {frame.period.min()}-{period}: {json.dumps(metrics)}")
    return TrainResult(run_id, version, period, metrics, str(holdout_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=int, default=None, help="newest period to include (default: latest available)")
    args = ap.parse_args()
    train(args.period if args.period is not None else available_periods()[-1])
