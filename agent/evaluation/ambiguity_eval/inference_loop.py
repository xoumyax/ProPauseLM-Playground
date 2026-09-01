"""
Multi-turn inference loop with simulated GPT-4o user-agent.

Given one TestCase, invoke AutoReasonSearchWorkflow; if the agent emits <ask>,
hand the question to a GPT-4o user-agent bound to the case's gold_clarification,
append the reply as a new user turn, and re-invoke. Cap at max_turns.

No edits to auto_search_sft.py — the workflow already accepts a `messages` list
and treats the final user turn as the current problem.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse analyze_trace without invoking the smoke batch main()
import sys
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))
from smoke_batch_sft import analyze_trace  # noqa: E402

from .data_prep import TestCase


ASK_RE = re.compile(r"<ask>(.*?)</ask>", re.DOTALL)

USER_AGENT_SYSTEM_PROMPT = (
    "You are simulating a real human user who just asked a research assistant a "
    "question. The assistant paused and asked you a clarifying question. Your job "
    "is to answer that clarifying question using ONLY the information in the "
    "provided `clarification_context`. Do NOT volunteer information beyond what "
    "the clarification context explicitly supports. Keep the reply natural, "
    "first-person, 1-3 sentences.\n\n"
    "Return JSON: {\"reply\": \"<your answer to the assistant, first person>\"}"
)


@dataclass
class Turn:
    role: str                         # "user" | "assistant" | "user_agent"
    content: str
    ask_text: Optional[str] = None    # populated for assistant turns containing <ask>
    trace_analysis: Optional[Dict[str, Any]] = None
    elapsed_s: Optional[float] = None
    n_tool_calls: Optional[int] = None


@dataclass
class TrajectoryResult:
    case_id: str
    query: str
    expected_behavior: str
    predicted_behavior: str           # "pause" | "clear" — based on first assistant turn
    turns: List[Turn] = field(default_factory=list)
    final_answer: str = ""
    total_elapsed_s: float = 0.0
    total_tool_calls: int = 0
    n_turns: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def extract_ask(text: str) -> Optional[str]:
    m = ASK_RE.search(text or "")
    if not m:
        return None
    return m.group(1).strip()


def _extract_trace_text(full_traces: Any) -> str:
    if full_traces is None:
        return ""
    if hasattr(full_traces, "generated_text"):
        return full_traces.generated_text or ""
    if isinstance(full_traces, dict):
        return full_traces.get("generated_text", "") or ""
    return ""


def _call_user_agent(
    client: Any,
    model: str,
    original_query: str,
    clarification_context: str,
    asked_question: str,
) -> str:
    """Simulate human reply using GPT-4o bound to gold_clarification."""
    import json
    messages = [
        {"role": "system", "content": USER_AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"original_query: {original_query}\n\n"
            f"clarification_context: {clarification_context}\n\n"
            f"assistant_question: {asked_question}"
        )},
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=256,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"reply": raw.strip()}
    return (data.get("reply") or "").strip()


async def run_trajectory(
    workflow: Any,
    case: TestCase,
    user_agent_client: Any,
    user_agent_model: str = "gpt-4o",
    max_turns: int = 3,
    dataset_name: Optional[str] = "researchqa",
    verbose: bool = False,
    preamble_messages: Optional[List[Dict[str, str]]] = None,
) -> TrajectoryResult:
    """
    Execute a single test case through the pause-aware workflow.

    preamble_messages: optional prior (role, content) pairs injected as history.
        Used by the few-shot CoT strategy to prepend exemplar turns.
    """
    result = TrajectoryResult(
        case_id=case.id,
        query=case.query,
        expected_behavior=case.expected_behavior,
        predicted_behavior="",
    )

    messages: List[Dict[str, str]] = list(preamble_messages or [])
    messages.append({"role": "user", "content": case.query})
    result.turns.append(Turn(role="user", content=case.query))

    t_all = time.time()
    predicted_set = False

    try:
        for turn_idx in range(max_turns):
            t0 = time.time()
            wf_out = await asyncio.wait_for(
                workflow(
                    problem=case.query,
                    messages=messages,
                    dataset_name=dataset_name,
                    verbose=verbose,
                ),
                timeout=420,
            )
            dt = time.time() - t0

            full_traces = wf_out.get("full_traces")
            trace_text = _extract_trace_text(full_traces)
            final_response = wf_out.get("final_response") or ""
            ta = analyze_trace(trace_text)
            n_tool_calls = wf_out.get("total_tool_calls", 0) or 0
            result.total_tool_calls += n_tool_calls

            ask_text = extract_ask(trace_text) or extract_ask(final_response)
            assistant_turn = Turn(
                role="assistant",
                content=final_response or trace_text,
                ask_text=ask_text,
                trace_analysis=ta,
                elapsed_s=round(dt, 2),
                n_tool_calls=n_tool_calls,
            )
            result.turns.append(assistant_turn)

            if not predicted_set:
                result.predicted_behavior = "pause" if ask_text else "clear"
                predicted_set = True

            if not ask_text:
                # No more clarification needed; this turn produced the final answer.
                result.final_answer = final_response or trace_text
                break

            # Model paused. If this is a clear-expected case we still simulate a
            # clarification so we can judge post-clarification quality.
            user_reply = _call_user_agent(
                user_agent_client,
                model=user_agent_model,
                original_query=case.query,
                clarification_context=(
                    case.gold_clarification
                    or "No additional context; provide your best reasonable assumption."
                ),
                asked_question=ask_text,
            )
            result.turns.append(Turn(role="user_agent", content=user_reply))
            messages.append({"role": "assistant", "content": final_response or trace_text})
            messages.append({"role": "user", "content": user_reply})
        else:
            # Hit max_turns without a non-ask answer; take whatever the last
            # assistant turn produced.
            result.final_answer = result.turns[-1].content if result.turns else ""
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"

    result.total_elapsed_s = round(time.time() - t_all, 2)
    result.n_turns = sum(1 for t in result.turns if t.role == "assistant")
    if not result.predicted_behavior:
        result.predicted_behavior = "clear"
    return result


def make_openai_client():
    """Instantiate openai.OpenAI; raises if OPENAI_API_KEY is missing."""
    import openai
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — export it before running the eval."
        )
    return openai.OpenAI(api_key=key)
