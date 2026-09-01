"""
Per-cell eval runner: (model_tag, strategy) × test_set → JSONL rows + summary.

Usage (env-dependent — see run_eval_matrix.sh for the served-model flow):

    cd /home/soumya/reasoning/ProPauseLM-Playground/agent
    python -m evaluation.ambiguity_eval.run_eval \\
        --model-tag stage3-reasoned \\
        --strategy zero_shot \\
        --test-set /home/soumya/reasoning/BaseAgent/Data/TestData/ambiguity_eval_v1.jsonl \\
        --vllm-node n15 --mcp-node n16 \\
        --out /home/soumya/reasoning/BaseAgent/Data/TestData/results/stage3-reasoned_zero_shot.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, TextIO

import yaml


class _Tee:
    """Minimal stdout tee: write to both a buffer and the real stdout."""
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            try:
                s.write(data)
            except Exception:
                pass
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return False

_HERE = Path(__file__).resolve().parent
_AGENT_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_AGENT_ROOT))

# Patch port checks — vLLM/MCP are on remote SLURM nodes
import dr_agent.utils as _dr_utils  # noqa: E402
_dr_utils.check_port = lambda *a, **kw: True
import workflows.auto_search_sft as _asft  # noqa: E402
_asft.check_port = lambda *a, **kw: True
from workflows.auto_search_sft import AutoReasonSearchWorkflow  # noqa: E402

from .data_prep import load_test_set, TestCase  # noqa: E402
from .inference_loop import run_trajectory, make_openai_client  # noqa: E402
from .judge import JudgeCall, judge_batch  # noqa: E402
from .metrics import summarize  # noqa: E402


MODEL_TAG_TO_CONFIG = {
    "stage1":           ("workflows/auto_search_sft.yaml",        "cli"),
    "stage1-complete":  ("workflows/auto_search_sft.yaml",        "cli"),
    "stage2-pause":     ("workflows/auto_search_sft_stage2.yaml", "cli_stage2"),
    "stage3-reasoned":  ("workflows/auto_search_sft_stage3.yaml", "cli_stage3"),
    "stage4-mechanism": ("workflows/auto_search_sft_stage4.yaml", "cli_stage3"),
}

MODEL_TAG_TO_PATH = {
    "stage1":           "/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage1",
    "stage1-complete":  "/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage1-complete",
    "stage2-pause":     "/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage2-pause",
    "stage3-reasoned":  "/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage3-reasoned",
    "stage4-mechanism": "/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage4-mechanism",
}


def load_strategy(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


async def run_all(
    cases: List[TestCase],
    workflow: AutoReasonSearchWorkflow,
    openai_client: Any,
    user_agent_model: str,
    preamble: List[Dict[str, str]],
    max_turns: int,
    transcript_path: Path,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    real_stdout = sys.stdout
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    with transcript_path.open("w") as tf:
        tf.write(f"# transcript written {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        tf.flush()
        for i, case in enumerate(cases, 1):
            header = (
                "\n" + "=" * 70 + "\n"
                f"[{i}/{len(cases)}] {case.query}\n"
                + "=" * 70 + "\n"
            )
            real_stdout.write(header)
            real_stdout.flush()

            buf = io.StringIO()
            tee = _Tee(buf, real_stdout)
            start_marker = (
                f"<start id={case.id} expected_behavior={case.expected_behavior} "
                f"idx={i}/{len(cases)}>\n"
            )
            tf.write(start_marker)
            tf.write(header)
            tf.flush()

            with contextlib.redirect_stdout(tee):
                tr = await run_trajectory(
                    workflow=workflow,
                    case=case,
                    user_agent_client=openai_client,
                    user_agent_model=user_agent_model,
                    max_turns=max_turns,
                    verbose=verbose,
                    preamble_messages=preamble,
                )

            d = tr.to_dict()
            d["case"] = {
                "id": case.id,
                "reference_answer": case.reference_answer,
                "gold_clarification": case.gold_clarification,
                "category": case.category,
            }
            rows.append(d)

            summary_block = (
                "\n--- Result ---\n"
                f"  id:                 {case.id}\n"
                f"  expected_behavior:  {case.expected_behavior}\n"
                f"  predicted_behavior: {tr.predicted_behavior}\n"
                f"  n_turns:            {tr.n_turns}\n"
                f"  total_tool_calls:   {tr.total_tool_calls}\n"
                f"  total_elapsed_s:    {tr.total_elapsed_s}\n"
                f"  error:              {tr.error}\n"
                "\n--- Final answer ---\n"
                f"{tr.final_answer or ''}\n"
            )
            real_stdout.write(summary_block)
            real_stdout.flush()

            tf.write(buf.getvalue())
            tf.write(summary_block)
            tf.write(f"<end id={case.id}>\n\n")
            tf.flush()
    return rows


def build_judge_calls(rows: List[Dict[str, Any]]) -> List[JudgeCall]:
    calls: List[JudgeCall] = []
    for r in rows:
        case = r["case"]
        gen_payload = {
            "query": r["query"],
            "expected_behavior": r["expected_behavior"],
            "reference_answer": case.get("reference_answer") or "",
            "gold_clarification": case.get("gold_clarification") or "",
            "model_final_answer": r.get("final_answer") or "",
        }
        calls.append(JudgeCall("generation", r["case_id"], gen_payload))
        if r.get("expected_behavior") == "pause":
            # Trajectory rubric: first assistant turn (pause), and final answer.
            turns = r.get("turns") or []
            first_asst = next((t for t in turns if t["role"] == "assistant"), {})
            traj_payload = {
                "query": r["query"],
                "gold_clarification": case.get("gold_clarification") or "",
                "first_assistant_turn": first_asst.get("content") or "",
                "final_answer": r.get("final_answer") or "",
                "predicted_behavior": r.get("predicted_behavior"),
            }
            calls.append(JudgeCall("trajectory", r["case_id"], traj_payload))
    return calls


def attach_judge_scores(rows: List[Dict[str, Any]], calls: List[JudgeCall], results: List[Dict[str, Any]]) -> None:
    by_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for call, res in zip(calls, results):
        by_case.setdefault(call.case_id, {})[call.rubric] = res
    for r in rows:
        bundle = by_case.get(r["case_id"], {})
        r["judge_generation"] = bundle.get("generation", {})
        r["judge_trajectory"] = bundle.get("trajectory", {})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-tag", required=True, choices=list(MODEL_TAG_TO_CONFIG))
    p.add_argument("--strategy", required=True, choices=["zero_shot", "few_shot_cot"])
    p.add_argument("--test-set", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vllm-node", default="n15")
    p.add_argument("--vllm-port", type=int, default=30001)
    p.add_argument("--mcp-node", default="n16")
    p.add_argument("--mcp-port", type=int, default=8000)
    p.add_argument("--user-agent-model", default="gpt-4o")
    p.add_argument("--judge-model", default="gpt-4o")
    p.add_argument("--max-turns", type=int, default=3)
    p.add_argument("--max-cases", type=int, default=0, help="0 = all")
    p.add_argument("--use-browse-agent", action="store_true", default=True)
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--temperature", type=float, default=None,
        help="Override search_agent_temperature from the workflow YAML (default: keep YAML value).",
    )
    args = p.parse_args()

    cases = load_test_set(args.test_set)
    if args.max_cases:
        cases = cases[: args.max_cases]

    yaml_path, prompt_version = MODEL_TAG_TO_CONFIG[args.model_tag]
    model_path = MODEL_TAG_TO_PATH[args.model_tag]
    strategy_cfg = load_strategy(_HERE / "prompts" / f"{args.strategy}.yaml")
    preamble = strategy_cfg.get("preamble_messages") or []

    os.environ["MCP_TRANSPORT_HOST"] = args.mcp_node
    os.environ["MCP_TRANSPORT_PORT"] = str(args.mcp_port)
    vllm_url = f"http://{args.vllm_node}:{args.vllm_port}/v1"

    print(f"model={args.model_tag} ({model_path})")
    print(f"config={yaml_path}  prompt_version={prompt_version}")
    print(f"strategy={args.strategy}  preamble_turns={len(preamble)}")
    print(f"vLLM={vllm_url}  MCP={args.mcp_node}:{args.mcp_port}")
    print(f"cases={len(cases)}  out={args.out}")
    print()

    wf_overrides: Dict[str, Any] = dict(
        configuration=yaml_path,
        search_agent_base_url=vllm_url,
        search_agent_model_name=model_path,
        search_agent_tokenizer_name=model_path,
        browse_agent_base_url=vllm_url,
        browse_agent_model_name=model_path,
        browse_agent_tokenizer_name=model_path,
        use_browse_agent=args.use_browse_agent,
        browse_tool_name="jina",
        search_agent_max_tool_calls=6,
        browse_timeout=30,
        prompt_version=prompt_version,
    )
    if args.temperature is not None:
        wf_overrides["search_agent_temperature"] = args.temperature
        print(f"temperature override: {args.temperature}")

    workflow = AutoReasonSearchWorkflow(**wf_overrides)

    openai_client = make_openai_client()

    out_path = Path(args.out)
    transcript_path = out_path.with_suffix(".transcript.txt")

    rows = asyncio.run(run_all(
        cases=cases,
        workflow=workflow,
        openai_client=openai_client,
        user_agent_model=args.user_agent_model,
        preamble=preamble,
        max_turns=args.max_turns,
        transcript_path=transcript_path,
        verbose=True,
    ))

    if not args.skip_judge:
        calls = build_judge_calls(rows)
        print(f"\nJudge calls: {len(calls)}")
        results = judge_batch(openai_client, calls, model=args.judge_model)
        attach_judge_scores(rows, calls, results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            r["_meta"] = {"model_tag": args.model_tag, "strategy": args.strategy}
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"Wrote {len(rows)} rows → {out_path}")

    summary = summarize(rows)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary → {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
