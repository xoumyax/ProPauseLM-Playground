#!/usr/bin/env python3
import json
from pathlib import Path


SRC_DIR = Path("/home/suxinqi666/code/m1/dr-tulu/test_data")
OUT_DIR = Path("/home/suxinqi666/code/m1/evaluation_data/test_data")

QUERY_IN = SRC_DIR / "query_template.jsonl"
LLM_IN = SRC_DIR / "llm_responses.jsonl"

QUERY_OUT = OUT_DIR / "query_template_compact.jsonl"
LLM_OUT = OUT_DIR / "llm_responses_compact.jsonl"


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    query_rows = read_jsonl(QUERY_IN)
    llm_rows = read_jsonl(LLM_IN)

    summary_by_id = {row["id"]: row for row in query_rows}

    query_compact = []
    for row in query_rows:
        query_compact.append(
            {
                "id": row["id"],
                "clarified_query": row["clarified_query"],
                "final_response": row["final_response"],
                "type": row["type"],
                "number of tokens": row["final_response_tokens"],
            }
        )

    llm_compact = []
    for row in llm_rows:
        summary = summary_by_id[row["id"]]
        llm_compact.append(
            {
                "id": row["id"],
                "iteration": row["iteration"],
                "clarified_query": row["query"],
                "final_response": row["agent_response"]["response"],
                "type": summary["type"],
                "number of tokens": row["response_tokens"],
            }
        )

    write_jsonl(QUERY_OUT, query_compact)
    write_jsonl(LLM_OUT, llm_compact)

    print(
        json.dumps(
            {
                "query_template_compact": str(QUERY_OUT),
                "llm_responses_compact": str(LLM_OUT),
                "query_rows": len(query_compact),
                "llm_rows": len(llm_compact),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
