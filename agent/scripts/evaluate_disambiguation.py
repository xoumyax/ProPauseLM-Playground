#!/usr/bin/env python3
"""Simple script to call the OpenAI API using the key from agent/.env."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

# Load .env from the agent/ directory regardless of where this script is run from
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def has_ask_tag_in_txt(txt_file_path: str) -> bool:
    """Return True if the text file contains an <ask>...</ask> block."""
    path = Path(txt_file_path)
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt file, got: {txt_file_path}")
    content = path.read_text(encoding="utf-8")
    return re.search(r"<ask>.*?</ask>", content, flags=re.DOTALL) is not None

def extract_content_before_answer_tag(txt_file_path: str) -> str:
    """Return content before the first <answer>...</answer> block in a text file."""
    path = Path(txt_file_path)
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt file, got: {txt_file_path}")
    content = path.read_text(encoding="utf-8")
    match = re.search(r"<answer>.*?</answer>", content, flags=re.DOTALL)
    if not match:
        return content
    return content[: match.start()]


def _clamp_unit_interval(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


def latter_contains_all_former_information(
    former: str,
    latter: str,
    *,
    model: str = "gpt-4o-mini",
    openai_client: Optional[OpenAI] = None,
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """Use an LLM to score how much of ``former``'s substantive content appears in ``latter``.

    ``latter`` may add extra detail. Score ``information_coverage`` is in ``[0.0, 1.0]``:
    0 = none of the important information from A is adequately present in B;
    1 = all substantive claims, facts, and constraints from A are present in B (rephrasing OK);
    values between reflect partial coverage. Contradictions should lower the score substantially.

    If ``return_details`` is False, returns only ``information_coverage`` (float).
    If True, returns the parsed JSON dict with keys ``information_coverage`` (float) and
    ``brief_reason`` (short string).
    """
    c = openai_client or client
    system = (
        "You compare two paragraphs. Paragraph A is the reference; paragraph B is the candidate. "
        "Estimate what fraction of A's substantive information (claims, facts, constraints, key points) "
        "is adequately reflected in B. B may add extra information; only judge coverage of A's content. "
        "Synonymous rephrasing counts as covered; clear omission or contradiction reduces the score. "
        "Reply with a single JSON object only, no markdown, with keys: "
        '"information_coverage" (number from 0.0 to 1.0 inclusive), '
        '"brief_reason" (short string explaining the score).'
    )
    user = (
        f"Paragraph A (reference):\n{former}\n\nParagraph B (candidate):\n{latter}\n\n"
        "JSON object:"
    )
    response = c.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    data: Dict[str, Any] = json.loads(raw)
    coverage = _clamp_unit_interval(data.get("information_coverage", 0))
    data["information_coverage"] = coverage
    if return_details:
        return data
    return coverage


if __name__ == "__main__":
    demo = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."},
        ],
    )
    print(demo.choices[0].message.content)
