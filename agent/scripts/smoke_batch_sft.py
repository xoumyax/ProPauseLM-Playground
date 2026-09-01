#!/usr/bin/env python3
"""
Smoke test for the Stage 1 SFT checkpoint on the DR-Tulu deep research workflow.

Runs 10 security queries through the full tool-calling pipeline and reports:
  - Whether the model emits <call_tool> tags (tool calls)
  - Whether it uses <think> tags (reasoning)
  - Whether it produces <answer> tags with <cite> tags
  - Tool call counts, searched/browsed links, answer quality

Prerequisites:
  1. vLLM serving the SFT checkpoint:
       sbatch scripts/serve_gemma_sft.sbatch
  2. MCP tool server running:
       sbatch scripts/serve_mcp.sbatch
  3. Update VLLM_NODE and MCP_NODE below to match allocated nodes.

Usage:
    cd /home/soumya/reasoning/ProPauseLM-Playground/agent
    python scripts/smoke_batch_sft.py [--vllm-node NODE] [--mcp-node NODE]
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Patch port checks — servers run on remote SLURM nodes
import dr_agent.utils as _dr_utils
_dr_utils.check_port = lambda *a, **kw: True
import workflows.auto_search_sft as _asft
_asft.check_port = lambda *a, **kw: True

from workflows.auto_search_sft import AutoReasonSearchWorkflow

QUERIES = [
    # Deep research — multi-step, requires searching + synthesizing
    # "Compare the root causes, exploitation techniques, and long-term industry impact of the SolarWinds Orion (2020) and XZ Utils (2024) supply-chain attacks. What lessons do they share?",
    # "Trace the evolution of ransomware-as-a-service from CryptoLocker (2013) through LockBit 3.0 (2023). How did affiliate models, encryption schemes, and law enforcement responses change over time?",
    # "What are the current best practices for detecting and mitigating living-off-the-land (LOLBin) attacks on Windows enterprise networks? Summarize recent research findings and cite specific tools or frameworks.",
    # "Analyze the security implications of LLM-powered coding assistants. What attack surfaces do they introduce, and what peer-reviewed mitigations have been proposed as of 2024-2025?",
    # "Write a technical briefing on post-quantum cryptography migration: which NIST-standardized algorithms exist, what are the known implementation pitfalls, and how are major cloud providers preparing?",
    # # Factual security — still needs search but more focused
    # "What is CVE-2021-44228 (Log4Shell), how does the JNDI lookup exploit work, and what were the cascading effects on the software supply chain?",
    # "Explain the MOVEit Transfer CVE-2023-34362 SQL injection vulnerability: root cause, exploitation timeline by Cl0p, and how organizations detected ongoing compromise.",
    #Good context
    "If we evaluate decision tree regressor and linear regression on the full training dataset, which one is more likely to appear artificially superior?",
    # Ambiguous / clarification-worthy
    "I want to know about the latest progress in disambiguating user query",
    "Please identify the current research gap in LLM personalization",
    "When to use agents in computer science?"
]


def analyze_trace(full_trace_text: str) -> dict:
    """Analyze the model's raw generation trace for DR-Tulu behaviors."""
    think_count = len(re.findall(r"<think>", full_trace_text))
    call_tool_count = len(re.findall(r"<call_tool\s", full_trace_text))
    has_answer = "<answer>" in full_trace_text and "</answer>" in full_trace_text
    cite_count = len(re.findall(r"<cite\s", full_trace_text))
    ask_count = len(re.findall(r"<ask>", full_trace_text))
    has_ask = ask_count > 0
    amb_tags = re.findall(
        r'<ambiguity\s+score="([0-9.]+)">([a-z]+)</ambiguity>', full_trace_text
    )
    amb_count = len(amb_tags)
    amb_labels = [lbl for _, lbl in amb_tags]
    amb_scores = [float(s) for s, _ in amb_tags]

    tool_types = re.findall(r'<call_tool\s+name="([^"]+)"', full_trace_text)
    tool_type_counts = {}
    for t in tool_types:
        tool_type_counts[t] = tool_type_counts.get(t, 0) + 1

    return {
        "think_count": think_count,
        "call_tool_count": call_tool_count,
        "has_answer_tags": has_answer,
        "cite_count": cite_count,
        "has_ask_tag": has_ask,
        "ask_count": ask_count,
        "ambiguity_count": amb_count,
        "ambiguity_labels": amb_labels,
        "ambiguity_scores": amb_scores,
        "tool_types_used": tool_type_counts,
    }


async def run_one(workflow, q, i):
    tool_events = []

    def cb(text, tool_calls):
        if tool_calls:
            for tc in tool_calls:
                tool_events.append(tc.tool_name)

    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            workflow(problem=q, dataset_name="researchqa", verbose=True, step_callback=cb),
            timeout=300,
        )
        dt = time.time() - t0

        ans = (result.get("final_response") or result.get("final_answer")
               or result.get("answer") or "")

        full_traces = result.get("full_traces")
        trace_text = ""
        if full_traces:
            if hasattr(full_traces, "generated_text"):
                trace_text = full_traces.generated_text
            elif isinstance(full_traces, dict):
                trace_text = full_traces.get("generated_text", "")

        trace_analysis = analyze_trace(trace_text)

        return {
            "idx": i,
            "query": q,
            "elapsed_s": round(dt, 1),
            "tool_events": tool_events,
            "n_tool_calls": result.get("total_tool_calls", len(tool_events)),
            "failed_tool_calls": result.get("total_failed_tool_calls", 0),
            "searched": len(result.get("searched_links", [])),
            "browsed": len(result.get("browsed_links", [])),
            "answer_len": len(ans),
            "answer_preview": ans[:500],
            "trace_analysis": trace_analysis,
            "trace_len": len(trace_text),
            "error": None,
        }
    except Exception as e:
        return {
            "idx": i,
            "query": q,
            "elapsed_s": round(time.time() - t0, 1),
            "tool_events": tool_events,
            "error": f"{type(e).__name__}: {e}",
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-node", default="n15",
                        help="Node running vLLM SFT server")
    parser.add_argument("--mcp-node", default="n16",
                        help="Node running MCP tool server")
    parser.add_argument("--vllm-port", type=int, default=30001)
    parser.add_argument("--mcp-port", type=int, default=8000)
    parser.add_argument("--config", default="workflows/auto_search_sft_stage3.yaml")
    parser.add_argument("--max-queries", type=int, default=10,
                        help="Run only first N queries")
    args = parser.parse_args()

    os.environ["MCP_TRANSPORT_HOST"] = args.mcp_node
    os.environ["MCP_TRANSPORT_PORT"] = str(args.mcp_port)

    vllm_url = f"http://{args.vllm_node}:{args.vllm_port}/v1"
    print(f"vLLM server: {vllm_url}")
    print(f"MCP server:  {args.mcp_node}:{args.mcp_port}")
    print(f"Config:      {args.config}")
    print()

    workflow = AutoReasonSearchWorkflow(
        configuration=args.config,
        search_agent_base_url=vllm_url,
        browse_agent_base_url=vllm_url,
        use_browse_agent=True,
        browse_tool_name="jina",
        search_agent_max_tool_calls=6,
        browse_timeout=30,
        prompt_version="cli",
    )

    queries = QUERIES[:args.max_queries] if args.max_queries else QUERIES

    out_path = Path("logs/sft_stage3_smoke.jsonl")
    out_path.parent.mkdir(exist_ok=True)
    records = []

    with out_path.open("w") as f:
        for i, q in enumerate(queries, 1):
            print(f"\n{'='*70}")
            print(f"[{i}/{len(queries)}] {q}")
            print(f"{'='*70}", flush=True)

            rec = await run_one(workflow, q, i)
            records.append(rec)
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()

            ta = rec.get("trace_analysis", {})
            print(f"\n--- Result ---")
            print(f"  Time:         {rec.get('elapsed_s')}s")
            print(f"  <think> tags: {ta.get('think_count', '?')}")
            print(f"  <call_tool>:  {ta.get('call_tool_count', '?')}  {ta.get('tool_types_used', {})}")
            print(f"  <answer> tag: {ta.get('has_answer_tags', '?')}")
            print(f"  <cite> tags:  {ta.get('cite_count', '?')}")
            print(f"  <ask> tags:   {ta.get('ask_count', '?')}")
            print(f"  <ambiguity>:  {ta.get('ambiguity_count', '?')}  "
                  f"labels={ta.get('ambiguity_labels', [])}  "
                  f"scores={ta.get('ambiguity_scores', [])}")
            print(f"  Searched:     {rec.get('searched', 0)} links")
            print(f"  Browsed:      {rec.get('browsed', 0)} links")
            print(f"  Answer len:   {rec.get('answer_len', 0)} chars")
            if rec.get("error"):
                print(f"  ERROR:        {rec['error']}")
            print(f"  Preview:      {rec.get('answer_preview', '')[:200]}...")
            print(flush=True)

    # === Summary ===
    print(f"\n{'='*70}")
    print("SUMMARY — SFT Stage 3 Deep Research Smoke Test")
    print(f"{'='*70}")
    n = len(records)
    ok = [r for r in records if not r.get("error")]

    used_think = sum(1 for r in ok if r.get("trace_analysis", {}).get("think_count", 0) > 0)
    used_tools = sum(1 for r in ok if r.get("trace_analysis", {}).get("call_tool_count", 0) > 0)
    has_answer = sum(1 for r in ok if r.get("trace_analysis", {}).get("has_answer_tags"))
    has_cite = sum(1 for r in ok if r.get("trace_analysis", {}).get("cite_count", 0) > 0)
    used_ask = sum(1 for r in ok if r.get("trace_analysis", {}).get("ask_count", 0) > 0)
    errored = n - len(ok)
    avg_tools = sum(r.get("trace_analysis", {}).get("call_tool_count", 0) for r in ok) / max(len(ok), 1)
    avg_time = sum(r.get("elapsed_s", 0) for r in ok) / max(len(ok), 1)

    print(f"  Queries run:       {n}")
    print(f"  Errored:           {errored}/{n}")
    print(f"  Used <think>:      {used_think}/{len(ok)}")
    print(f"  Used <call_tool>:  {used_tools}/{len(ok)}   (avg {avg_tools:.1f} per query)")
    print(f"  Has <answer> tags: {has_answer}/{len(ok)}")
    print(f"  Has <cite> tags:   {has_cite}/{len(ok)}")
    print(f"  Used <ask>:        {used_ask}/{len(ok)}")
    used_amb = sum(1 for r in ok
                   if r.get("trace_analysis", {}).get("ambiguity_count", 0) > 0)
    print(f"  Used <ambiguity>:  {used_amb}/{len(ok)}")
    print(f"  Avg time:          {avg_time:.1f}s")
    print()

    # Verdict
    if len(ok) == 0:
        print("VERDICT: ERROR — All queries failed, check vLLM/MCP server connectivity")
    elif used_tools >= len(ok) * 0.7 and has_answer >= len(ok) * 0.5:
        print("VERDICT: PASS — Model is doing deep research (tool calls + structured answers)")
    elif used_tools >= len(ok) * 0.3:
        print("VERDICT: PARTIAL — Model makes some tool calls but inconsistently")
    else:
        print("VERDICT: FAIL — Model is NOT making tool calls, SFT did not learn DR-Tulu format")

    print(f"\nFull log: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
