"""
Collapse per-cell JSONL + summary.json outputs into a single CSV table.

Looks for files matching `<model_tag>_<strategy>.summary.json` under the
results directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


FIELDS = [
    "model_tag", "strategy", "n", "errored",
    "pause_f1", "pause_precision", "pause_recall",
    "clear_accuracy", "pause_accuracy",
    "gen_overall", "gen_correctness", "gen_grounding",
    "gen_completeness", "gen_conciseness",
    "traj_overall", "traj_think_names_dimension",
    "traj_ask_minimal", "traj_post_clarification",
    "avg_tool_calls", "avg_elapsed_s",
]


def _flatten(summary: Dict[str, Any], model_tag: str, strategy: str) -> Dict[str, Any]:
    beh = summary.get("behavior", {}) or {}
    gen = summary.get("generation_judge", {}) or {}
    traj = summary.get("trajectory_judge", {}) or {}
    return {
        "model_tag": model_tag,
        "strategy": strategy,
        "n": summary.get("n"),
        "errored": summary.get("errored"),
        "pause_f1": beh.get("pause_f1"),
        "pause_precision": beh.get("pause_precision"),
        "pause_recall": beh.get("pause_recall"),
        "clear_accuracy": beh.get("clear_accuracy"),
        "pause_accuracy": beh.get("pause_accuracy"),
        "gen_overall": gen.get("overall"),
        "gen_correctness": gen.get("correctness"),
        "gen_grounding": gen.get("grounding"),
        "gen_completeness": gen.get("completeness"),
        "gen_conciseness": gen.get("conciseness"),
        "traj_overall": traj.get("overall"),
        "traj_think_names_dimension": traj.get("think_names_dimension"),
        "traj_ask_minimal": traj.get("ask_minimal"),
        "traj_post_clarification": traj.get("post_clarification"),
        "avg_tool_calls": summary.get("avg_tool_calls"),
        "avg_elapsed_s": summary.get("avg_elapsed_s"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        default="/home/soumya/reasoning/BaseAgent/Data/TestData/results",
    )
    p.add_argument(
        "--out",
        default="/home/soumya/reasoning/BaseAgent/Data/TestData/results/matrix_summary.csv",
    )
    args = p.parse_args()

    root = Path(args.results_dir)
    rows: List[Dict[str, Any]] = []
    for sp in sorted(root.glob("*.summary.json")):
        stem = sp.name.replace(".summary.json", "")
        if "_" not in stem:
            continue
        model_tag, strategy = stem.rsplit("_", 1)
        # strategy tokens contain an underscore ("zero_shot", "few_shot_cot")
        # so do a smarter split:
        for s in ("zero_shot", "few_shot_cot"):
            if stem.endswith("_" + s):
                model_tag = stem[: -(len(s) + 1)]
                strategy = s
                break
        try:
            summary = json.loads(sp.read_text())
        except json.JSONDecodeError:
            print(f"skip {sp} (bad json)", file=sys.stderr)
            continue
        rows.append(_flatten(summary, model_tag, strategy))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} cells → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
