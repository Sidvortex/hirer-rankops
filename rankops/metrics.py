"""Ranking metrics computed per job, then averaged (the standard LTR way)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _dcg(gains: np.ndarray, k: int) -> float:
    gains = gains[:k]
    return float(np.sum((2.0**gains - 1) / np.log2(np.arange(2, gains.size + 2))))


def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")
    ideal = _dcg(np.sort(labels)[::-1], k)
    return _dcg(labels[order], k) / ideal if ideal > 0 else np.nan


def ranking_report(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    df = frame[["job_id", "relevance"]].assign(score=scores)
    out: dict[str, list[float]] = {"ndcg_at_5": [], "ndcg_at_10": [], "precision_at_5": []}
    for _, g in df.groupby("job_id", sort=False):
        y, s = g.relevance.to_numpy(), g.score.to_numpy()
        out["ndcg_at_5"].append(ndcg_at_k(y, s, 5))
        out["ndcg_at_10"].append(ndcg_at_k(y, s, 10))
        top5 = y[np.argsort(-s, kind="stable")[:5]]
        out["precision_at_5"].append(float(np.mean(top5 >= 3)))  # grade 3+ = "would interview"
    return {k: round(float(np.nanmean(v)), 4) for k, v in out.items()}


def top_k_overlap(a: list[str], b: list[str], k: int = 10) -> float:
    """Share of the top-k that two rankings agree on. Used for shadow comparisons."""
    if not a or not b:
        return 1.0
    k = min(k, len(a), len(b))
    return len(set(a[:k]) & set(b[:k])) / k
