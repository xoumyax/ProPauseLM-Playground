# Runbook — Evaluation + CLI App

Practical, copy-pasteable commands. Assumes Stages 1–3 are already trained
and checkpoints live under `/home/soumya/reasoning/SFT_train/runs/`.

---

## 0. Prerequisites

| Requirement | How to satisfy |
|-------------|----------------|
| Conda env with workflow deps | `conda activate dr_agent` |
| OpenAI API key (eval only)   | `export OPENAI_API_KEY=sk-...` |
| MCP tool server running      | `sbatch agent/scripts/serve_mcp.sbatch`; note the node |
| vLLM serving an SFT checkpoint | see each section below |
| `sbatch` available           | submit from a login-node shell (not inside Claude) |

All paths below are relative to `/home/soumya/reasoning/ProPauseLM-Playground/agent`
unless stated otherwise.

```bash
cd /home/soumya/reasoning/ProPauseLM-Playground/agent
```

---

## 1. Quick single-model eval (Stage 3 only)

Use this to verify the harness end-to-end against the already-serving Stage 3
checkpoint before touching the full matrix.

### 1.1 Bring up servers (login node)
```bash
sbatch agent/scripts/serve_mcp.sbatch            # note allocated node (e.g. n16)
sbatch agent/scripts/serve_gemma_sft.sbatch      # serves Stage 3 on e.g. n15:30001
```

Wait until both sbatch jobs log `Application startup complete`.

### 1.2 Build the test set (one-time)
```bash
python -m evaluation.ambiguity_eval.data_prep --build-dummy
# → writes 40 rows to Data/TestData/ambiguity_eval_v1.jsonl
```

### 1.3 Sanity run (5 cases, no full judge spend)
```bash
export OPENAI_API_KEY=sk-...
python -m evaluation.ambiguity_eval.run_eval \
    --model-tag stage3-reasoned \
    --strategy zero_shot \
    --test-set /home/soumya/reasoning/BaseAgent/Data/TestData/ambiguity_eval_v1.jsonl \
    --out /tmp/s3_zs_sanity.jsonl \
    --vllm-node n15 --mcp-node n16 \
    --max-cases 5
```

Outputs:
- `/tmp/s3_zs_sanity.jsonl` — per-row trajectories + judge scores
- `/tmp/s3_zs_sanity.summary.json` — aggregate metrics, printed to stdout

Expected summary shape:
```json
{
  "behavior": {"pause_f1": 0.8, "clear_accuracy": 1.0, "pause_accuracy": 0.67},
  "generation_judge": {"overall": 4.1, "correctness": 4.0, ...},
  "trajectory_judge": {"overall": 3.8, "think_names_dimension": 4.0, ...},
  "avg_tool_calls": 2.1, "avg_elapsed_s": 18.3, "errored": 0, "n": 5
}
```

### 1.4 Full Stage-3 eval (40 cases, both strategies)
```bash
for STRATEGY in zero_shot few_shot_cot; do
  python -m evaluation.ambiguity_eval.run_eval \
      --model-tag stage3-reasoned --strategy $STRATEGY \
      --test-set /home/soumya/reasoning/BaseAgent/Data/TestData/ambiguity_eval_v1.jsonl \
      --out /home/soumya/reasoning/BaseAgent/Data/TestData/results/stage3-reasoned_${STRATEGY}.jsonl \
      --vllm-node n15 --mcp-node n16
done
```

---

## 2. Full matrix eval (4 models × 2 strategies)

### 2.1 Tear down the default serve
If Stage 3 from §1 is still running on `n15`, `scancel` it — the matrix
driver expects to bring its own serves up per checkpoint.

### 2.2 Run the driver
```bash
export OPENAI_API_KEY=sk-...
./scripts/run_eval_matrix.sh --vllm-node n15 --mcp-node n16
```

For each model, the driver **prints** a serve command that you copy-paste to
a login-node shell:

```
sbatch --export=ALL,MODEL_PATH=/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage1,PORT=30001 \
       scripts/serve_gemma_sft_matrix.sbatch
```

Then:
1. Wait for `Application startup complete`.
2. If the allocated node differs from `n15`, rerun the driver with
   `--vllm-node <actual>` (or Ctrl-C and re-invoke).
3. Press enter in the driver shell — it runs `zero_shot` and `few_shot_cot`
   against that checkpoint.
4. When both finish, the driver tells you to `scancel` the serve job. Do it,
   press enter, and the driver advances to the next model.

After all four models: the driver calls `aggregate.py` automatically, writing

```
/home/soumya/reasoning/BaseAgent/Data/TestData/results/
    stage1_zero_shot.jsonl            + .summary.json
    stage1_few_shot_cot.jsonl         + .summary.json
    stage1-complete_zero_shot.jsonl   + .summary.json
    …
    matrix_summary.csv                ← 8 rows, one per cell
```

### 2.3 Useful matrix flags
| Flag | Purpose |
|------|---------|
| `--max-cases N`        | run on first N rows (sanity) |
| `--skip stage1`        | skip a checkpoint (e.g. if already run) |
| `--results-dir PATH`   | override output directory |
| `--test-set PATH`      | point at a different JSONL |

### 2.4 Re-aggregation only (no eval)
If all per-cell summaries already exist and you just want a fresh CSV:
```bash
python -m evaluation.ambiguity_eval.aggregate \
    --results-dir /home/soumya/reasoning/BaseAgent/Data/TestData/results \
    --out /home/soumya/reasoning/BaseAgent/Data/TestData/results/matrix_summary.csv
```

Judge responses are cached in `Data/TestData/.judge_cache/` so re-running
`run_eval` with the same model-output-hash costs zero API calls.

---

## 3. Reading the results

### 3.1 Per-row JSONL (`stage3-reasoned_zero_shot.jsonl`)
Each line is one `TrajectoryResult` with:
- `case_id`, `query`, `expected_behavior`, `predicted_behavior`
- `turns[]` — ordered user / assistant / user_agent turns with `trace_analysis`
- `final_answer`, `total_elapsed_s`, `total_tool_calls`, `n_turns`, `error`
- `judge_generation`, `judge_trajectory` (if `--skip-judge` not passed)
- `_meta: {model_tag, strategy}`

Quick scan:
```bash
jq -c '{id:.case_id, pred:.predicted_behavior, exp:.expected_behavior, turns:.n_turns, gen:.judge_generation.overall}' \
   /home/soumya/reasoning/BaseAgent/Data/TestData/results/stage3-reasoned_zero_shot.jsonl \
 | head
```

### 3.2 Matrix CSV
```bash
column -t -s, /home/soumya/reasoning/BaseAgent/Data/TestData/results/matrix_summary.csv | less -S
```

Columns of interest:
- `pause_f1`, `pause_precision`, `pause_recall`, `clear_accuracy`
- `gen_overall` — holistic generation quality (1–5)
- `traj_overall` — trajectory quality on vague rows (1–5)
- `avg_tool_calls`, `avg_elapsed_s`

Expected ordering (hypothesis):
- **pause_f1**: stage3-reasoned > stage2-pause ≫ stage1-complete > stage1
- **clear_accuracy**: all ≥ 0.9 (no regression on clear queries)
- **traj_overall**: stage3-reasoned ≥ stage2-pause; stage1* not meaningful

---

## 4. CLI app (`chat_cli.py`)

Real human fills `<ask>` turns — no user-agent. Good for live demos.

### 4.1 Bring up servers
```bash
sbatch agent/scripts/serve_mcp.sbatch        # → n16
sbatch agent/scripts/serve_gemma_sft.sbatch  # serves Stage 3 → n15
```

### 4.2 Launch CLI
```bash
cd /home/soumya/reasoning/ProPauseLM-Playground/agent
python scripts/chat_cli.py --vllm-node n15 --mcp-node n16
```

Default target: Stage 3 checkpoint +
`workflows/auto_search_sft_stage3.yaml` + `prompt_version=cli_stage3`.

### 4.3 CLI commands
| Command | Effect |
|---------|--------|
| `/exit`             | quit |
| `/reset`            | clear conversation history (fresh context) |
| `/save <path>`      | append current transcript as JSONL to `<path>` |

### 4.4 Example session
```
> Please identify the current research gap in LLM personalization
--- agent (8.4s, tools=0) ---
<think>The user asks about a research gap but has not named which dimension …</think>
<ambiguity score="0.78">ambiguous</ambiguity>
<ask>Which aspect interests you — privacy-preserving training, on-device inference,
     evaluation benchmarks, or alignment/personality?</ask>

[agent asked — your next message will be the clarification]

> on-device inference for enterprise agents
--- agent (31.2s, tools=4) ---
<think>User clarified scope: on-device inference, enterprise. Proceed with search.</think>
<ambiguity score="0.12">clear</ambiguity>
<call_tool name="google_search">on-device LLM personalization enterprise survey 2024</call_tool>
…
<answer><cite id="S_xyz">Recent work on PEFT-based on-device personalization …</cite></answer>
```

### 4.5 Point the CLI at a different checkpoint
```bash
python scripts/chat_cli.py \
    --vllm-node n15 --mcp-node n16 \
    --config workflows/auto_search_sft_stage2.yaml \
    --model-path /home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage2-pause \
    --prompt-version cli_stage2
```

---

## 5. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `OPENAI_API_KEY is not set` | key not exported in this shell | `export OPENAI_API_KEY=sk-...` |
| `ConnectionError` to vLLM   | wrong `--vllm-node` / server not up | `squeue`; check log `logs/gemma_sft_serve_*.out` for `Application startup complete` |
| All queries return clear (no `<ask>`) on Stage 3 | wrong `prompt_version` at inference (e.g. `cli` instead of `cli_stage3`) | confirm workflow YAML has `prompt_version: "cli_stage3"` |
| `<ask>` fired but agent still answered | `auto_search_sft.py` early-return missing | should have `if "<answer>" in … or "<ask>" in …:` in `__call__` |
| Judge returns `parse_error` | non-JSON response | run again; cache will skip successful rows |
| Matrix driver stuck on "serve healthy?" | serve still warming or on different node | press enter only after log shows startup complete; re-invoke with correct `--vllm-node` |
| Empty `matrix_summary.csv`  | no `*.summary.json` in results dir | run at least one `run_eval` cell first |

---

## 6. One-screen cheat sheet

```bash
# Prereqs
cd /home/soumya/reasoning/ProPauseLM-Playground/agent
conda activate dr_agent
export OPENAI_API_KEY=sk-...

# Build test data (once)
python -m evaluation.ambiguity_eval.data_prep --build-dummy

# Sanity eval — 5 Stage 3 rows
python -m evaluation.ambiguity_eval.run_eval \
   --model-tag stage3-reasoned --strategy zero_shot \
   --test-set /home/soumya/reasoning/BaseAgent/Data/TestData/ambiguity_eval_v1.jsonl \
   --out /tmp/sanity.jsonl --vllm-node n15 --mcp-node n16 --max-cases 5

# Full 4×2 matrix
./scripts/run_eval_matrix.sh --vllm-node n15 --mcp-node n16

# Re-aggregate only
python -m evaluation.ambiguity_eval.aggregate

# Live CLI demo
python scripts/chat_cli.py --vllm-node n15 --mcp-node n16
```
