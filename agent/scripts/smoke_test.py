#!/usr/bin/env python3
"""Phase A smoke test: send one query through the full DR-Tulu workflow."""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("MCP_TRANSPORT_HOST", "n2")
os.environ.setdefault("MCP_TRANSPORT_PORT", "8000")

# Patch check_port to always succeed — servers run on remote nodes (n1, n2),
# not localhost, so the launch-check port probe would wrongly try to boot
# local servers.
import dr_agent.utils as _dr_utils
_dr_utils.check_port = lambda *a, **kw: True
import workflows.auto_search_sft as _asft
_asft.check_port = lambda *a, **kw: True

from workflows.auto_search_sft import AutoReasonSearchWorkflow


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else (
        "What is CVE-2021-44228 (Log4Shell) and how does the exploit work?"
    )

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

    def step_cb(text, tool_calls):
        if tool_calls:
            for tc in tool_calls:
                print(f"[TOOL] {tc.tool_name}", flush=True)

    print(f"\n=== Query ===\n{query}\n", flush=True)
    result = await workflow(
        problem=query,
        dataset_name=None,
        verbose=True,
        step_callback=step_cb,
    )

    print("\n=== Final ===", flush=True)
    print(f"tool_calls={result.get('total_tool_calls')} "
          f"failed={result.get('total_failed_tool_calls')} "
          f"searched={len(result.get('searched_links', []))} "
          f"browsed={len(result.get('browsed_links', []))}", flush=True)
    ans = result.get("final_answer") or result.get("answer") or ""
    print(f"\n--- answer ---\n{ans[:2000]}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
