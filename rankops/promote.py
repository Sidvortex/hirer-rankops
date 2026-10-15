"""Champion/challenger gate.

Both models are scored on the same held-out jobs from the newest period. The
challenger takes the `champion` alias only if it wins on the primary metric by
`min_improvement` and doesn't regress the guardrail metric by more than
`max_guardrail_drop`. Every decision is written to the registry as tags and to
reports/promotion_pXX.json so you can audit why a model is (or isn't) live.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from rankops import config
from rankops.features import FEATURES
from rankops.metrics import ranking_report
from rankops.train import setup_mlflow


def _version_for(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def load_model(name: str, version: str):
    return mlflow.lightgbm.load_model(f"models:/{name}/{version}")


def decide(champ: dict | None, chall: dict, rules: dict) -> tuple[bool, str]:
    if champ is None:
        return True, "no champion yet"
    pm, gm = rules["primary_metric"], rules["guardrail_metric"]
    gain = chall[pm] - champ[pm]
    drop = champ[gm] - chall[gm]
    if gain < rules["min_improvement"]:
        return False, f"{pm} gain {gain:+.4f} < required {rules['min_improvement']}"
    if drop > rules["max_guardrail_drop"]:
        return False, f"{gm} dropped {drop:.4f} > allowed {rules['max_guardrail_drop']}"
    return True, f"{pm} gain {gain:+.4f}, {gm} change {-drop:+.4f}"


def promote(period: int) -> dict:
    setup_mlflow()
    p = config.params()
    name, rules = p["registry"]["model_name"], p["promotion"]
    client = MlflowClient()

    chall_v = _version_for(client, name, "challenger")
    if chall_v is None:
        raise RuntimeError("no challenger registered; run training first")
    champ_v = _version_for(client, name, "champion")

    holdout = pd.read_parquet(config.FEATURE_DIR / f"holdout_p{period}.parquet")
    chall_m = ranking_report(holdout, load_model(name, chall_v.version).predict(holdout[FEATURES]))
    champ_m = None
    if champ_v is not None and champ_v.version != chall_v.version:
        champ_m = ranking_report(holdout, load_model(name, champ_v.version).predict(holdout[FEATURES]))

    promoted, reason = decide(champ_m, chall_m, rules)
    if champ_v is not None and champ_v.version == chall_v.version:
        promoted, reason = False, "challenger is already champion"

    tag = client.set_model_version_tag
    tag(name, chall_v.version, "promotion_decision", "promoted" if promoted else "rejected")
    tag(name, chall_v.version, "promotion_reason", reason)
    if promoted:
        if champ_v is not None:
            client.set_registered_model_alias(name, "previous_champion", champ_v.version)
        client.set_registered_model_alias(name, "champion", chall_v.version)
        # The new champion's training features become the drift baseline.
        ref_src = config.ROOT / chall_v.tags.get("reference_features", "")
        if ref_src.is_file():
            config.REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(ref_src, config.REFERENCE_PATH)

    decision = {
        "period": period,
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "champion_version": champ_v.version if champ_v else None,
        "challenger_version": chall_v.version,
        "champion_metrics": champ_m,
        "challenger_metrics": chall_m,
        "promoted": promoted,
        "reason": reason,
        "live_version": chall_v.version if promoted else (champ_v.version if champ_v else None),
    }
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / f"promotion_p{period}.json").write_text(json.dumps(decision, indent=2))
    print(f"{'PROMOTED' if promoted else 'kept champion'} v{decision['live_version']}: {reason}")
    return decision


def rollback() -> str:
    """Point `champion` back at `previous_champion`. Serving picks it up on /reload."""
    setup_mlflow()
    name = config.params()["registry"]["model_name"]
    client = MlflowClient()
    prev = _version_for(client, name, "previous_champion")
    if prev is None:
        raise RuntimeError("no previous champion to roll back to")
    client.set_registered_model_alias(name, "champion", prev.version)
    print(f"rolled back champion to v{prev.version}")
    return prev.version


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=int)
    ap.add_argument("--rollback", action="store_true")
    a = ap.parse_args()
    if a.rollback:
        rollback()
    else:
        from rankops.data import available_periods
        promote(a.period if a.period is not None else available_periods()[-1])
