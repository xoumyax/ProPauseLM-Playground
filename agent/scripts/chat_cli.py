#!/usr/bin/env python3
"""
Interactive CLI for the pause-aware deep-research agent.

No simulated user agent — the real human answers <ask> turns directly.

Prereqs (same as smoke_batch_sft.py):
  1. vLLM serving the SFT checkpoint (default Stage 3).
  2. MCP tool server running.

Usage:
    cd /home/soumya/reasoning/ProPauseLM-Playground/agent
    python scripts/chat_cli.py [--vllm-node n15] [--mcp-node n16] \\
           [--config workflows/auto_search_sft_stage3.yaml]

Commands (typed at the > prompt):
    /exit           quit
    /reset          clear conversation history
    /save <path>    append current transcript to <path> as JSONL
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import dr_agent.utils as _dr_utils  # noqa: E402
_dr_utils.check_port = lambda *a, **kw: True
import workflows.auto_search_sft as _asft  # noqa: E402
_asft.check_port = lambda *a, **kw: True
from workflows.auto_search_sft import AutoReasonSearchWorkflow  # noqa: E402


# ANSI colors — avoid pulling in `rich` for portability.
C_RESET = "\033[0m"
C_DIM = "\033[2m"
C_THINK = "\033[36m"     # cyan
C_AMB = "\033[33m"       # yellow
C_ASK = "\033[35m"       # magenta
C_ANSWER = "\033[32m"    # green
C_TOOL = "\033[34m"      # blue
C_CITE = "\033[90m"      # grey


TAG_COLORS = [
    (re.compile(r"(<think>.*?</think>)", re.DOTALL), C_THINK),
    (re.compile(r'(<ambiguity[^>]*>.*?</ambiguity>)', re.DOTALL), C_AMB),
    (re.compile(r"(<ask>.*?</ask>)", re.DOTALL), C_ASK),
    (re.compile(r"(<answer>.*?</answer>)", re.DOTALL), C_ANSWER),
    (re.compile(r'(<call_tool[^>]*>.*?</call_tool>)', re.DOTALL), C_TOOL),
    (re.compile(r'(<cite[^>]*>.*?</cite>)', re.DOTALL), C_CITE),
]

ASK_RE = re.compile(r"<ask>(.*?)</ask>", re.DOTALL)


def colorize(text: str) -> str:
    out = text
    for pat, color in TAG_COLORS:
        out = pat.sub(lambda m, c=color: f"{c}{m.group(1)}{C_RESET}", out)
    return out


def extract_trace_text(full_traces: Any) -> str:
    if full_traces is None:
        return ""
    if hasattr(full_traces, "generated_text"):
        return full_traces.generated_text or ""
    if isinstance(full_traces, dict):
        return full_traces.get("generated_text", "") or ""
    return ""


async def turn(workflow: AutoReasonSearchWorkflow, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    problem = messages[-1]["content"]
    t0 = time.time()
    out = await asyncio.wait_for(
        workflow(
            problem=problem,
            messages=messages,
            dataset_name="researchqa",
            verbose=False,
        ),
        timeout=420,
    )
    out["_elapsed_s"] = round(time.time() - t0, 2)
    return out


def print_agent_turn(out: Dict[str, Any]) -> str:
    full_traces = out.get("full_traces")
    trace_text = extract_trace_text(full_traces)
    final_response = out.get("final_response") or ""
    # Prefer the raw trace (contains all tags) for display; fall back to final_response.
    display = trace_text or final_response
    print()
    print(f"{C_DIM}--- agent ({out.get('_elapsed_s')}s, tools={out.get('total_tool_calls', 0)}) ---{C_RESET}")
    print(colorize(display))
    return final_response or trace_text


def human_input(prompt: str = "> ") -> str:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return "/exit"


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-node", default="n15")
    parser.add_argument("--mcp-node", default="n16")
    parser.add_argument("--vllm-port", type=int, default=30001)
    parser.add_argument("--mcp-port", type=int, default=8000)
    parser.add_argument("--config", default="workflows/auto_search_sft_stage3.yaml")
    parser.add_argument("--model-path",
                        default="/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage3-reasoned")
    parser.add_argument("--prompt-version", default="cli_stage3")
    args = parser.parse_args()

    os.environ["MCP_TRANSPORT_HOST"] = args.mcp_node
    os.environ["MCP_TRANSPORT_PORT"] = str(args.mcp_port)
    vllm_url = f"http://{args.vllm_node}:{args.vllm_port}/v1"

    print(f"{C_DIM}vLLM: {vllm_url}   MCP: {args.mcp_node}:{args.mcp_port}")
    print(f"config={args.config}  prompt_version={args.prompt_version}")
    print(f"model={args.model_path}{C_RESET}")
    print(f"{C_DIM}Commands: /exit  /reset  /save <path>{C_RESET}\n")

    workflow = AutoReasonSearchWorkflow(
        configuration=args.config,
        search_agent_base_url=vllm_url,
        search_agent_model_name=args.model_path,
        search_agent_tokenizer_name=args.model_path,
        browse_agent_base_url=vllm_url,
        browse_agent_model_name=args.model_path,
        browse_agent_tokenizer_name=args.model_path,
        use_browse_agent=True,
        browse_tool_name="jina",
        search_agent_max_tool_calls=6,
        browse_timeout=30,
        prompt_version=args.prompt_version,
    )

    messages: List[Dict[str, str]] = []
    transcript: List[Dict[str, Any]] = []

    while True:
        text = human_input("> ").strip()
        if not text:
            continue
        if text == "/exit":
            print("bye.")
            return 0
        if text == "/reset":
            messages = []
            transcript = []
            print(f"{C_DIM}[reset]{C_RESET}")
            continue
        if text.startswith("/save "):
            path = Path(text[len("/save "):].strip()).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as f:
                for t in transcript:
                    f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
            print(f"{C_DIM}[saved {len(transcript)} turns → {path}]{C_RESET}")
            continue

        messages.append({"role": "user", "content": text})
        transcript.append({"role": "user", "content": text})

        try:
            out = await turn(workflow, messages)
        except Exception as exc:  # noqa: BLE001
            print(f"\033[31mERROR: {type(exc).__name__}: {exc}{C_RESET}")
            messages.pop()  # drop the unanswered user turn
            continue

        agent_text = print_agent_turn(out)
        messages.append({"role": "assistant", "content": agent_text})
        transcript.append({
            "role": "assistant",
            "content": agent_text,
            "tool_calls": out.get("total_tool_calls", 0),
            "elapsed_s": out.get("_elapsed_s"),
        })

        m = ASK_RE.search(agent_text)
        if m:
            # Pause — user will answer at next loop iteration.
            print(f"\n{C_DIM}[agent asked — your next message will be the clarification]{C_RESET}")


if __name__ == "__main__":
    asyncio.run(main())
