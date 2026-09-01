"""
Rule-based and aggregate metrics over TrajectoryResults + judge scores.

Behavior accuracy is fully deterministic (string match on <ask>). Quality
scores are averaged from the GPT-4o judge.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class BehaviorStats:
    n: int
    n_clear_true: int
    n_pause_true: int
    pause_precision: float
    pause_recall: float
    pause_f1: float
    clear_accuracy: float    # P(predicted=clear | expected=clear)
    pause_accuracy: float    # P(predicted=pause | expected=pause)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "n_clear_true": self.n_clear_true,
            "n_pause_true": self.n_pause_true,
            "pause_precision": round(self.pause_precision, 3),
            "pause_recall": round(self.pause_recall, 3),
            "pause_f1": round(self.pause_f1, 3),
            "clear_accuracy": round(self.clear_accuracy, 3),
            "pause_accuracy": round(self.pause_accuracy, 3),
        }


def behavior_stats(rows: Iterable[Dict[str, Any]]) -> BehaviorStats:
    rows = list(rows)
    n = len(rows)
    tp = sum(1 for r in rows if r["expected_behavior"] == "pause" and r["predicted_behavior"] == "pause")
    fp = sum(1 for r in rows if r["expected_behavior"] == "clear" and r["predicted_behavior"] == "pause")
    fn = sum(1 for r in rows if r["expected_behavior"] == "pause" and r["predicted_behavior"] == "clear")
    tn = sum(1 for r in rows if r["expected_behavior"] == "clear" and r["predicted_behavior"] == "clear")

    n_pause = tp + fn
    n_clear = tn + fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    clear_acc = tn / n_clear if n_clear else 0.0
    pause_acc = tp / n_pause if n_pause else 0.0

    return BehaviorStats(
        n=n, n_clear_true=n_clear, n_pause_true=n_pause,
        pause_precision=precision, pause_recall=recall, pause_f1=f1,
        clear_accuracy=clear_acc, pause_accuracy=pause_acc,
    )


def _avg(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(mean(vals), 3) if vals else None


def aggregate_judge(
    rows: List[Dict[str, Any]],
    field: str,
) -> Dict[str, Optional[float]]:
    """`field` is 'generation' or 'trajectory'; rows carry `judge_generation` / `judge_trajectory`."""
    key = f"judge_{field}"
    bucket: Dict[str, List[float]] = {}
    for r in rows:
        j = r.get(key) or {}
        for k, v in j.items():
            if isinstance(v, (int, float)):
                bucket.setdefault(k, []).append(float(v))
    return {k: _avg(vs) for k, vs in bucket.items()}


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    behavior = behavior_stats(rows).to_dict()
    gen = aggregate_judge(rows, "generation")
    traj = aggregate_judge([r for r in rows if r.get("expected_behavior") == "pause"], "trajectory")

    avg_tool_calls = _avg([r.get("total_tool_calls") or 0 for r in rows])
    avg_elapsed = _avg([r.get("total_elapsed_s") or 0 for r in rows])
    errored = sum(1 for r in rows if r.get("error"))

    return {
        "behavior": behavior,
        "generation_judge": gen,
        "trajectory_judge": traj,
        "avg_tool_calls": avg_tool_calls,
        "avg_elapsed_s": avg_elapsed,
        "errored": errored,
        "n": len(rows),
    }
