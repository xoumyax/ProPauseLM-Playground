#!/usr/bin/env python3
"""Phase A go/no-go: run 10 queries, tally tool-call and answer behavior."""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("MCP_TRANSPORT_HOST", "n2")
os.environ.setdefault("MCP_TRANSPORT_PORT", "8000")

import dr_agent.utils as _dr_utils
_dr_utils.check_port = lambda *a, **kw: True
import workflows.auto_search_sft as _asft
_asft.check_port = lambda *a, **kw: True

from workflows.auto_search_sft import AutoReasonSearchWorkflow

QUERIES = [
    "What is CVE-2021-44228 (Log4Shell) and how does the exploit work?",
    "Explain the SolarWinds Orion supply-chain attack and which backdoor was used.",
    "What mitigations exist for Spectre and Meltdown CPU vulnerabilities?",
    "How does the Kerberoasting attack against Active Directory work?",
    "Summarize the MOVEit Transfer CVE-2023-34362 SQL injection vulnerability.",
    "What is the difference between SLSA Level 3 and Level 4 for software supply chain security?",
    "Describe the 2024 XZ Utils backdoor (CVE-2024-3094) and how it was discovered.",
    "What are the main detection techniques for DLL sideloading on Windows?",
    "Explain how BGP hijacking works and list one notable real-world incident.",
    "What is the MITRE ATT&CK technique T1055 and list two sub-techniques.",
]


async def run_one(workflow, q, i):
    tool_events = []

    def cb(text, tool_calls):
        if tool_calls:
            for tc in tool_calls:
                tool_events.append(tc.tool_name)

    t0 = time.time()
    try:
        result = await asyncio.wait_for(
            workflow(problem=q, dataset_name=None, verbose=False, step_callback=cb),
            timeout=240,
        )
        dt = time.time() - t0
        ans = (result.get("final_response") or result.get("final_answer")
               or result.get("answer") or "")
        return {
            "idx": i,
            "query": q,
            "elapsed_s": round(dt, 1),
            "tool_calls": tool_events,
            "n_tool_calls": result.get("total_tool_calls", len(tool_events)),
            "failed_tool_calls": result.get("total_failed_tool_calls", 0),
            "searched": len(result.get("searched_links", [])),
            "browsed": len(result.get("browsed_links", [])),
            "answer_len": len(ans),
            "answer_preview": ans[:300],
            "answer_full": ans,
            "error": None,
        }
    except Exception as e:
        return {
            "idx": i,
            "query": q,
            "elapsed_s": round(time.time() - t0, 1),
            "tool_calls": tool_events,
            "error": f"{type(e).__name__}: {e}",
        }


async def main():
    workflow = AutoReasonSearchWorkflow(
        configuration="workflows/auto_search_sft.yaml",
        search_agent_base_url="http://n1:30001/v1",
        browse_agent_base_url="http://n1:30001/v1",
        use_browse_agent=True,
        browse_tool_name="jina",
        search_agent_max_tool_calls=6,
        browse_timeout=30,
        prompt_version="cli",
    )

    out_path = Path("logs/phaseA_gono_go.jsonl")
    md_path = Path("logs/phaseA_answers.md")
    out_path.parent.mkdir(exist_ok=True)
    records = []
    md_f = md_path.open("w")
    md_f.write("# Phase A — Gemma-4-E4B-it smoke-test answers\n\n")
    with out_path.open("w") as f:
        for i, q in enumerate(QUERIES, 1):
            print(f"\n=== [{i}/{len(QUERIES)}] {q}", flush=True)
            rec = await run_one(workflow, q, i)
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            md_f.write(f"## Q{rec['idx']}. {rec['query']}\n\n")
            md_f.write(f"- tool_calls: {rec.get('tool_calls')}\n")
            md_f.write(f"- searched: {rec.get('searched')}, browsed: {rec.get('browsed')}\n")
            md_f.write(f"- elapsed: {rec.get('elapsed_s')}s\n\n")
            md_f.write("### Answer\n\n")
            md_f.write(rec.get("answer_full") or "_(no answer)_")
            md_f.write("\n\n---\n\n")
            md_f.flush()
            print(f"    tools={rec.get('tool_calls')} "
                  f"ans_len={rec.get('answer_len')} "
                  f"err={rec.get('error')}", flush=True)

    # Tally
    print("\n=== SUMMARY ===", flush=True)
    n = len(records)
    made_call = sum(1 for r in records if r.get("n_tool_calls", 0) > 0)
    produced_ans = sum(1 for r in records if r.get("answer_len", 0) > 50)
    errored = sum(1 for r in records if r.get("error"))
    avg_calls = sum(r.get("n_tool_calls", 0) for r in records) / max(n, 1)
    print(f"N queries         : {n}")
    print(f"Issued tool call  : {made_call}/{n}")
    print(f"Produced answer   : {produced_ans}/{n}")
    print(f"Errored           : {errored}/{n}")
    print(f"Avg tool calls    : {avg_calls:.2f}")
    print(f"\nPer-query:")
    for r in records:
        print(f"  [{r['idx']}] calls={r.get('n_tool_calls',0)} "
              f"ans={r.get('answer_len',0)} err={'Y' if r.get('error') else '-'}")
    print(f"\nFull log: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
