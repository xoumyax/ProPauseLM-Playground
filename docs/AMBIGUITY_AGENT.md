# Ambiguity-Aware Deep-Research Agent — System Overview

End-to-end documentation of the three-stage SFT pipeline, the DR-Tulu-style
tool-calling workflow, the evaluation harness, and the CLI application.

---

## 1. Goal

Build a deep-research assistant on **Gemma-4-E4B** that:

1. Uses tools iteratively (`<call_tool>` → snippets → `<think>` → more tools).
2. **Rates its own ambiguity** on every turn via an explicit
   `<ambiguity score="0.XX">{clear|moderate|ambiguous}</ambiguity>` tag.
3. **Pauses** on ambiguous queries with a minimal `<ask>…</ask>` clarification
   question instead of hallucinating a shallow answer.
4. Cites retrieved evidence with `<cite id="SX">…</cite>`.

The system prompt mandates a three-tag preamble on every assistant turn:

```
<think>{one-sentence rationale}</think>
<ambiguity score="0.XX">{clear|moderate|ambiguous}</ambiguity>
```

If the rating is `ambiguous` (score ≥ 0.60), the model appends
`<ask>{clarifying question}</ask>` and stops. If `clear` (< 0.30), it proceeds
with tool calls and an `<answer>`.

---

## 2. Training pipeline — three stages of SFT

All stages use the same trainer (`SFT_train/train_sft.py`), FSDP across
4 nodes × 2 H100s, `Gemma4TextDecoderLayer` wrap, prompt-masked CLM loss.

| Stage | Start from | Train data | Epochs | LR | Output dir |
|-------|-----------|-----------|--------|-----|-----------|
| **1** | `google/gemma-4-e4b-it` | `stage1_train.jsonl` (36.5k DR-Tulu traces) | 3 | 5e-6 | `runs/gemma4-e4b-sft-stage1-complete` |
| **2** | Stage 1 | `stage2_train.jsonl` (~30k mix: 5k pause + 25k stage1) | 2 | 1e-6 | `runs/gemma4-e4b-sft-stage2-pause` |
| **3** | Stage 2 | `stage3_train.jsonl` (5,161 rows: 1,161 reasoned-pause + 4k stage2) | 2 | 2e-6 | `runs/gemma4-e4b-sft-stage3-reasoned` |

Each stage writes its own system prompt to **every** training row (normalization
script inlined during Stage 3 prep) so train/inference prompts match exactly.

### 2.1 Stage 1 — tool-use foundation
- **Data**: DR-Tulu trace corpus (36,503 rows) with `<think>`/`<call_tool>`/
  `<answer>`/`<cite>` tags. Zero `<ask>` tags.
- **Outcome**: model emits the DR-Tulu format reliably (5/5 on smoke test) but
  is **confidently shallow** on ambiguous queries — declares "I have enough
  information" after a single search.

### 2.2 Stage 2 — pause-aware fine-tuning
- **Motivation**: Stage 1 never saw `<ask>`. We need matched positive/negative
  examples where the model learns *when* to pause.
- **Data generation pipeline** (`SFT_train/prep_stage2_pause_data.py`):
  1. Sample ambiguous prompts from `BaseAgent/Data/generalpurposedata/train/*/data.jsonl`.
  2. Rate each with `AmbiguityScorer` (`BaseAgent/src/ambiguity/classifier.py` —
     MiniLM-L6 + MLP + sigmoid); keep rows ≥ 0.65.
  3. Have a judge LLM generate a *disambiguated rewrite*; keep the rewrite if
     rater scores it < 0.25 (gives a matched clear/pause pair).
  4. Generate pause traces with the judge, pre-filling the rater score:
     ```
     <think>{missing-dimension analysis}</think>
     <ambiguity score="0.XX">ambiguous</ambiguity>
     <ask>{one specific clarifying question}</ask>
     ```
  5. Judge LLM verifies each pause candidate against a strict rubric; keep
     only `ambiguous=true AND ask_quality=good`.
  6. For clear examples, reuse Stage 1 traces with a fresh
     `<think>…</think>\n<ambiguity score="0.0X">clear</ambiguity>` preamble.
- **System prompt**: adds the 3-tag preamble + 1 pause example. Frozen later
  as `cli_stage2.yaml` for regression eval.
- **Outcome** (smoke_sft on Stage 2): pauses correctly on short/ambiguous
  queries ("what's the best encryption algorithm?"), but misses *scope-
  ambiguous* queries like "research gap in LLM personalization" because the
  rater under-scores those.

### 2.3 Stage 3 — ambiguity reasoning
- **Motivation**: Stage 2's rater classified "research gap in LLM
  personalization" as `clear` (score 0.27). The model needs to **reason**
  about what dimension is under-specified, not just defer to a numeric
  threshold.
- **Data generation pipeline** (`SFT_train/prep_stage3_reasoning_data.py`):
  1. Reuse 10k ambiguous prompts from `generalpurposedata/train/`.
  2. Ask **GPT-4o** (strict JSON) to classify each prompt along an explicit
     ambiguity taxonomy (`aspect` / `entity` / `timeframe` / `scope` / `format`)
     and emit `{is_ambiguous, missing_dimension, why, ask, score}`.
  3. Keep only `is_ambiguous=true` — yielded **1,161** high-quality pause rows.
  4. Format as SFT conversations:
     ```
     <think>{GPT-4o "why" — must name the missing dimension}</think>
     <ambiguity score="0.XX">ambiguous</ambiguity>
     <ask>{GPT-4o-generated minimal clarification}</ask>
     ```
- **System prompt** (`cli_stage3.yaml`): adds a mandatory line *"Your
  `<think>` MUST explicitly name which dimension is under-specified …"*
  and two pause examples + one clear example. Every one of the 5,161
  training rows carries this system prompt verbatim (normalized inline).
- **Training**: LR `2e-6`, warmup 50, 2 epochs, save/eval every 200 steps.
  Final checkpoint: `runs/gemma4-e4b-sft-stage3-reasoned` (checkpoint-318 +
  final `model.safetensors`).
- **Outcome** (smoke_sft_18926): correctly pauses on scope-ambiguous queries
  AND stays clear on specific factual/deep-research queries. Ready for eval.

---

## 3. Runtime architecture

### 3.1 System prompt versions
`agent/dr_agent/shared_prompts/unified_tool_calling.py` registers four
stage-aligned system prompts:

| `prompt_version` | File | Use |
|--------|------|-----|
| `cli` | `unified_tool_calling_cli.yaml` | Stage 1 / Stage 1-complete |
| `cli_stage2` | `unified_tool_calling_cli_stage2.yaml` | Stage 2 (frozen for eval) |
| `cli_stage3` | `unified_tool_calling_cli_stage3.yaml` | Stage 3 (current) |

### 3.2 Workflow — `agent/workflows/auto_search_sft.py`
`AutoReasonSearchWorkflow.__call__(problem, messages, …)`:

1. Accepts optional `messages` list; last user turn becomes `problem`, rest
   becomes `history`. Enables multi-turn clarification without core changes.
2. Invokes `SearchAgent` (tool loop) — emits `<call_tool>` tags, executes MCP
   tools (google_search, snippet_search, browse_webpage), feeds results back.
3. **Early-return on `<ask>`**: if the generated text contains `<ask>` OR
   `<answer>`, skip the synthesis step (`AnswerAgent`) and return immediately.
   This is what makes the agent *terminate* on a pause rather than trying to
   answer anyway.
4. Otherwise, `AnswerAgent` synthesizes a final `<answer>` with citations.

### 3.3 Inference config — `agent/workflows/auto_search_sft_stage3.yaml`
```yaml
search_agent_model_name: .../runs/gemma4-e4b-sft-stage3-reasoned
search_agent_temperature: 1.0         # matches training sampling
search_agent_max_tool_calls: 10
prompt_version: "cli_stage3"          # resolves to cli_stage3.yaml
use_browse_agent: false               # read-through browse is heavy; off by default
```

---

## 4. Evaluation harness

Lives at `agent/evaluation/ambiguity_eval/`. Designed to evaluate **four
checkpoints × two strategies = 8 cells** on a held-out test set.

### 4.1 Test data
`BaseAgent/Data/TestData/ambiguity_eval_v1.jsonl` — 40-row dummy seed:

```json
{"id": "t001",
 "query": "Please identify the current research gap in LLM personalization",
 "expected_behavior": "pause",
 "gold_clarification": "...scope/context for the simulated user agent...",
 "reference_answer": "",
 "category": "scope_ambiguous",
 "schema_version": "v1"}
```

- 20 clear rows: hand-written bank of factual + scoped deep-research queries
  with `reference_answer`.
- 20 vague rows: sampled from `generalpurposedata/train/*/data.jsonl` (all
  pre-labeled `[PAUSE]`); gold_clarification is a heuristic stub.
- **Schema-versioned**; real data can swap in by bumping `schema_version`.

### 4.2 Inference loop — `inference_loop.py`
`run_trajectory(workflow, case, user_agent_client, max_turns=3)`:

1. Invoke workflow with the query.
2. If the trace contains `<ask>`, extract the question.
3. Hand `(original_query, gold_clarification, asked_question)` to a **GPT-4o
   user-agent** with a strict system prompt: *"Answer using ONLY the
   clarification_context; do not volunteer beyond it."* JSON-mode reply.
4. Append reply as a new user turn in `messages`; re-invoke workflow.
5. Cap at `max_turns` assistant turns.

Returns a `TrajectoryResult` with ordered `Turn`s, final answer,
`predicted_behavior` (first-turn decision), tool-call count, elapsed time,
and per-turn `analyze_trace` (reused from `smoke_batch_sft.py`).

### 4.3 Judge — `judge.py`
Single GPT-4o client. Two rubrics, both JSON-mode, parallelized via
`ThreadPoolExecutor(max_workers=8)`, cached on disk at
`Data/TestData/.judge_cache/` keyed by `(rubric, case_id, payload_hash)` so
re-aggregation is free.

- **generation** (all rows): correctness / grounding / completeness /
  conciseness, each 1–5, plus overall + rationale.
- **trajectory** (vague rows only): `think_names_dimension` / `ask_minimal` /
  `post_clarification`, each 1–5.

### 4.4 Metrics — `metrics.py`
- **Behavior accuracy** (rule-based, no LLM): `predicted_behavior = "pause"`
  iff the first assistant turn contains `<ask>`. Confusion matrix →
  precision / recall / F1 on pause; separate clear- and pause-accuracy.
- **Judge aggregates**: per-dimension means across rows.
- **Runtime aggregates**: avg tool calls, avg elapsed seconds, errored count.

### 4.5 Strategies
- `zero_shot` — system prompt only (matches training).
- `few_shot_cot` — prepends 3 exemplars as prior user/assistant turns in
  `messages`: (a) clear CVE, (b) vague personalization → `<ask>`,
  (c) clear Volt Typhoon campaign.

### 4.6 Runner — `run_eval.py` + `run_eval_matrix.sh`
- `run_eval.py --model-tag <tag> --strategy <s> --test-set <path> --out <path>`
  writes per-row JSONL + `.summary.json`.
- `run_eval_matrix.sh` walks the 4×2 matrix: **prints** the sbatch serve
  command for each checkpoint (user submits from login node), runs Python
  eval inline once the user confirms the serve is up, prompts for
  `scancel` before advancing, then calls `aggregate.py` for a single CSV.
- `serve_gemma_sft_matrix.sbatch` — parameterized via
  `sbatch --export=ALL,MODEL_PATH=…,PORT=…`.

### 4.7 CSV aggregator — `aggregate.py`
Scans `*.summary.json` in the results directory; flattens behavior metrics,
generation-judge dimensions, trajectory-judge dimensions, and runtime stats
into one CSV (`matrix_summary.csv`).

---

## 5. Application — CLI app

`agent/scripts/chat_cli.py` — no simulated user-agent; the real human
answers `<ask>` turns directly.

- Loads `workflows/auto_search_sft_stage3.yaml` by default.
- ANSI-colored tag rendering (no `rich` dep): think (cyan), ambiguity
  (yellow), ask (magenta), answer (green), call_tool (blue), cite (grey).
- Maintains a `messages` list across turns; appends each new user input and
  agent response so the workflow sees full history.
- Commands: `/exit`, `/reset`, `/save <path>` (appends JSONL transcript).

Prereqs: same as smoke batch — MCP server (`serve_mcp.sbatch`) and vLLM
(`serve_gemma_sft.sbatch` pointing at Stage 3) must be healthy.

---

## 6. File map (what was created or changed)

### New training artifacts
- `SFT_train/prep_stage2_pause_data.py` — Stage 2 data pipeline
- `SFT_train/prep_stage3_reasoning_data.py` — Stage 3 data pipeline (GPT-4o)
- `SFT_train/train_sft_stage2.sbatch`, `train_sft_stage2_distributed.sh`
- `SFT_train/train_sft_stage3.sbatch`, `train_sft_stage3_distributed.sh`
- `BaseAgent/Data/sft_combined/stage2_train.jsonl` (~30k)
- `BaseAgent/Data/sft_combined/stage2_pause.jsonl`, `stage2_pause_eval.jsonl`
- `BaseAgent/Data/sft_combined/stage3_train.jsonl` (5,161, normalized prompt)
- `BaseAgent/Data/sft_combined/stage3_pause.jsonl` (1,161)

### New prompts
- `agent/dr_agent/shared_prompts/unified_tool_calling_cli_stage2.yaml`
- `agent/dr_agent/shared_prompts/unified_tool_calling_cli_stage3.yaml`
- `agent/dr_agent/shared_prompts/unified_tool_calling.py` — registers the
  `cli_stage2` and `cli_stage3` entries in `STRUCTURED_PROMPTS`

### New runtime config
- `agent/workflows/auto_search_sft_stage2.yaml`
- `agent/workflows/auto_search_sft_stage3.yaml`

### New eval package
- `agent/evaluation/ambiguity_eval/__init__.py`
- `agent/evaluation/ambiguity_eval/data_prep.py`
- `agent/evaluation/ambiguity_eval/inference_loop.py`
- `agent/evaluation/ambiguity_eval/judge.py`
- `agent/evaluation/ambiguity_eval/metrics.py`
- `agent/evaluation/ambiguity_eval/run_eval.py`
- `agent/evaluation/ambiguity_eval/aggregate.py`
- `agent/evaluation/ambiguity_eval/prompts/zero_shot.yaml`
- `agent/evaluation/ambiguity_eval/prompts/few_shot_cot.yaml`
- `BaseAgent/Data/TestData/ambiguity_eval_v1.jsonl` (40-row dummy)

### New scripts
- `agent/scripts/serve_gemma_sft_matrix.sbatch`
- `agent/scripts/run_eval_matrix.sh`
- `agent/scripts/chat_cli.py`

### Unchanged but load-bearing
- `agent/workflows/auto_search_sft.py` — early-return on `<ask>` (single
  patch line in `__call__`; see §3.2).
- `agent/scripts/smoke_batch_sft.py` — `analyze_trace` is imported by
  the eval inference loop.
- `SFT_train/train_sft.py` — trainer, untouched across all three stages.
- `BaseAgent/src/ambiguity/classifier.py` — `AmbiguityScorer` reused by
  Stage 2 data prep and the dummy test-set builder.

---

## 7. Running the full pipeline

### 7.1 Train (user submits; login node lacks `sbatch`)
```bash
sbatch SFT_train/train_sft.sbatch         # Stage 1
sbatch SFT_train/train_sft_stage2.sbatch  # Stage 2 (needs Stage 1)
sbatch SFT_train/train_sft_stage3.sbatch  # Stage 3 (needs Stage 2)
```

### 7.2 Smoke test a checkpoint
```bash
# serve on one node, MCP on another
sbatch agent/scripts/serve_gemma_sft.sbatch
sbatch agent/scripts/serve_mcp.sbatch
# then
cd agent
python scripts/smoke_batch_sft.py --vllm-node <N> --mcp-node <N> \
       --config workflows/auto_search_sft_stage3.yaml
```

### 7.3 Run the full eval matrix
```bash
cd agent
export OPENAI_API_KEY=sk-…
./scripts/run_eval_matrix.sh --vllm-node <N> --mcp-node <N>
# → 8 per-cell JSONLs + summaries, and matrix_summary.csv
```

### 7.4 Interactive demo
```bash
# Serve Stage 3 + MCP first.
cd agent
python scripts/chat_cli.py --vllm-node <N> --mcp-node <N>
# Try: "Please identify the current research gap in LLM personalization"
# Agent should <ask>. Answer in free text; agent resumes with <answer>.
```

---

## 8. Open items / future work

- **Replace dummy test data**: when real held-out set is available, swap the
  contents of `ambiguity_eval_v1.jsonl`; loader is schema-versioned.
- **Ministral-3-8B replication** (stretch): copy `train_sft_stage{1,2,3}.sbatch`
  to `mistral_sft_stage{1,2,3}.sbatch`, verify `chat_template.jinja`
  compatibility, re-run eval matrix. 8 new cells for cross-model comparison.
- **Judge bias monitoring**: spot-check GPT-4o generation scores against a
  second judge (e.g. Gemini) on a 20-row slice to catch systematic bias.
- **Few-shot context budget**: 3 exemplars + long Stage 3 system prompt may
  approach 8k tokens; truncate retrieved-snippet budget for that strategy if
  we see context overflow errors.
