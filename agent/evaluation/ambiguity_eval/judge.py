"""
GPT-4o judge client for the ambiguity eval.

Two rubrics:
  * generation — scores the final answer against a reference (clear) or gold
    clarification (vague). 1-5 Likert on correctness, grounding, completeness,
    conciseness.
  * trajectory — vague only. Scores whether <think> named the missing
    dimension, <ask> was a minimal reasonable clarification, and the
    post-clarification answer addresses the clarified intent.

Responses are cached on disk keyed by (rubric, case_id, model_output_hash) so
re-running aggregation is free.
"""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_CACHE_DIR = Path(
    "/home/soumya/reasoning/BaseAgent/Data/TestData/.judge_cache"
)


GENERATION_RUBRIC = """\
You are an impartial judge scoring a deep-research assistant's final answer.

Rate the answer on four dimensions, each 1-5 (5 = excellent):
  correctness  — factual agreement with the reference
  grounding    — does the answer cite retrieved evidence (look for <cite> tags)
  completeness — does it address all parts of the (clarified) question
  conciseness  — is it appropriately brief without dropping key information

Return JSON only:
{
  "correctness": int,
  "grounding": int,
  "completeness": int,
  "conciseness": int,
  "overall": int,        // your holistic 1-5
  "rationale": "<1-2 sentences>"
}
"""


TRAJECTORY_RUBRIC = """\
You are an impartial judge scoring a deep-research assistant's pause/clarify
trajectory on an AMBIGUOUS query. Score each dimension 1-5 (5 = excellent):

  think_names_dimension — did the <think> tag explicitly name which dimension
                          is under-specified (aspect / entity / timeframe /
                          scope / format)?
  ask_minimal           — is the <ask> question a minimal, reasonable
                          clarification (not a kitchen-sink follow-up)?
  post_clarification    — does the final answer address the clarified intent
                          rather than the original vague query literally?

Return JSON only:
{
  "think_names_dimension": int,
  "ask_minimal": int,
  "post_clarification": int,
  "overall": int,
  "rationale": "<1-2 sentences>"
}
"""


@dataclass
class JudgeCall:
    rubric: str        # "generation" | "trajectory"
    case_id: str
    payload: Dict[str, Any]


def _hash_key(rubric: str, case_id: str, payload: Dict[str, Any]) -> str:
    ser = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(f"{rubric}|{case_id}|{ser}".encode()).hexdigest()[:24]
    return f"{rubric}_{case_id}_{h}.json"


def _load_cached(cache_dir: Path, key: str) -> Optional[Dict[str, Any]]:
    p = cache_dir / key
    if p.exists():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            return None
    return None


def _save_cached(cache_dir: Path, key: str, obj: Dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / key).write_text(json.dumps(obj, ensure_ascii=False))


def _call_gpt(
    client: Any,
    model: str,
    system_prompt: str,
    user_payload: Dict[str, Any],
) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.0,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"parse_error": True, "raw": raw}


def judge_one(
    client: Any,
    call: JudgeCall,
    model: str = "gpt-4o",
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Dict[str, Any]:
    key = _hash_key(call.rubric, call.case_id, call.payload)
    cached = _load_cached(cache_dir, key)
    if cached is not None:
        return cached

    rubric = GENERATION_RUBRIC if call.rubric == "generation" else TRAJECTORY_RUBRIC
    out = _call_gpt(client, model, rubric, call.payload)
    _save_cached(cache_dir, key, out)
    return out


def judge_batch(
    client: Any,
    calls: Iterable[JudgeCall],
    model: str = "gpt-4o",
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_workers: int = 8,
) -> List[Dict[str, Any]]:
    calls_list = list(calls)
    results: List[Optional[Dict[str, Any]]] = [None] * len(calls_list)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(judge_one, client, c, model, cache_dir): i
            for i, c in enumerate(calls_list)
        }
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[i] = {"error": f"{type(exc).__name__}: {exc}"}
    return [r or {} for r in results]
