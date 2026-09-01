# Stage 4 SFT — Mechanism-Unspecified Ambiguity

## Context

Stage 3 eval on the 58-row held-out test set revealed a gap:

| | zero-shot | few-shot CoT (3 exemplars) |
|---|---|---|
| `pause_recall` | 0.111 | 0.889 |
| `clear_accuracy` | 0.959 | 0.551 |
| `pause_f1` | 0.167 | 0.410 |
| `gen_overall` | 4.19 | 4.17 |

Temperature tuning (T=0.3 few-shot) broke the tool loop (52/58 errored), so
prompt-only ceiling is around `pause_f1 ≈ 0.41` with an undesirable
precision/recall trade-off.

### What the test data actually exercises

All 58 queries are hardware-architecture / systems research. Surface features
are identical across clear and vague:

| | Clear (49) | Vague (9) |
|---|---|---|
| First word | "How"×39, "What"×8 | "How"×7, "What"×2 |
| Median length | 17 words | 16 words |

The distinguishing axis is **mechanism-unspecified**:

- **Clear** rows name a specific mechanism:
  *"How can a multi-bank, **tag-less branch predictor** be designed using different hashing functions to eliminate conflict aliasing…?"*
- **Vague** rows name only the problem category; the gold clarification
  commits to a mechanism the base query avoids:
  *"How reliable is modern memory technologies?"* → gold: RowHammer in DRAM.
  *"How can shared hardware resources be efficiently scheduled…?"* → gold:
  NISQ quantum throughput.

Stage 3 was trained on GPT-4o ambiguity labels along the taxonomy
`aspect / entity / timeframe / scope / format`. **`mechanism`** was not a
category the model was taught to flag.

## Goal

Extend the model's ambiguity taxonomy to include a sixth dimension,
**mechanism-unspecified**, without regressing on the Stage 3 behavior (clear
accuracy, tool use, generation quality) — i.e., **no catastrophic forgetting**.

## Approach

### 1. Data — `SFT_train/prep_stage4_mechanism_data.py`

**Seed corpus** (~500 queries): existing test clears + a broader sample of
hardware/systems-research queries. Sources:
- Test file clears (49) — known-good clear pattern.
- Generate additional clears + vagues from scratch via GPT-4o, seeded with
  ISCA / MICRO / HPCA / ASPLOS paper abstracts (local if available; otherwise
  GPT-4o is asked to emit queries in that domain).

**Pair generator (GPT-4o, strict JSON)**: for each seed topic, emit a
**matched pair**:
- `vague_query`: problem category only, no committed mechanism.
- `clarified_query`: specific mechanism named.
- `sibling_clear_query`: different mechanism in the same problem space, fully
  specified — gives us a broad-looking-but-clear negative.
- `reference_answer` (optional; used only if we want to mirror test format).

**Judge filter (GPT-4o)**: binary rubric per row — *"Does the query commit to
a specific mechanism, technique, or named method (not just a topic)?"* →
`{commits: bool, mechanism: str}`. Keep only rows where:
- vague rows have `commits=false`
- clear rows have `commits=true`

**Target mix**: 3,000 rows — **65% clear (1,950) / 35% vague (1,050)**.
Deliberately over-weighted toward clears to counteract the over-pause
regression observed under few-shot CoT.

### 2. Assistant-trace templates

**Vague (mechanism-unspecified)**:
```
<think>The user asks about {problem category} but has not committed to a
specific mechanism. Candidates differ materially: {mech1}, {mech2}, {mech3}.
Mechanism-unspecified — pause.</think>
<ambiguity score="0.{70-80}">ambiguous</ambiguity>
<ask>Which direction are you considering — {mech1}, {mech2}, or {mech3}?</ask>
```

**Clear (broad topic, specific mechanism)**:
```
<think>The user names {specific mechanism} in the {problem category} space.
Asked dimension is {how-it-works / tradeoff / design-rationale}. No missing
dimension.</think>
<ambiguity score="0.{10-20}">clear</ambiguity>
<think>I need authoritative sources on {mechanism}.</think>
<call_tool name="snippet_search">{mechanism} {domain}</call_tool>
...
<answer>...<cite id="...">...</cite>...</answer>
```

Clear traces reuse **real Stage 3 trajectories** from `stage3_train.jsonl`
where available; only the opening `<think>`/`<ambiguity>` preamble is rewritten
to match the mechanism-specific template.

### 3. System prompt — `cli_stage4.yaml`

Extends Stage 3 prompt with one additional taxonomy item and two
hardware-domain examples. Key edit:

> `<think>` MUST explicitly name which dimension is under-specified (missing
> **aspect, entity, timeframe, scope, output format, or mechanism**) …

Plus a new paragraph:

> **Mechanism-unspecified**: if the query names a problem category (e.g.,
> "memory reliability", "scheduling shared hardware") without committing to
> a specific mechanism or technique, and multiple mechanisms would give
> materially different answers, that is ambiguous. Conversely, a *broad topic
> with a specific mechanism named* (e.g., "multi-bank tag-less branch
> predictor") is **clear** — broadness of topic is not ambiguity.

Registered as `STRUCTURED_PROMPTS["cli_stage4"]`. `cli_stage3` stays for
comparison eval.

### 4. Training — catastrophic-forgetting guard

Continue from Stage 3 checkpoint
(`runs/gemma4-e4b-sft-stage3-reasoned`).

| Lever | Value | Why |
|-------|-------|-----|
| Mix | 3,000 new + 5,161 full `stage3_train.jsonl` = **8,161 total (37% new)** | Dominant old-data share anchors prior knowledge |
| Learning rate | `1e-6` | Half of Stage 3's 2e-6; minimal drift |
| Epochs | 2 | Stage 3 used 2; no signal for more |
| Warmup | 50 steps | Match Stage 3 |
| Eval / save | every 150 steps | Early checkpoint for rollback if regression |
| Early stopping | patience 3 on eval loss | Automatic rollback |
| Gradient accum | 4 (match Stage 3) | |
| System prompt | `cli_stage4` normalized across **every** row (including mixed-in Stage 3 rows) | Train/inference parity |

`save_total_limit=3` keeps checkpoint-150, -300, -450 etc. If the final
checkpoint regresses on clear_accuracy, we fall back to an earlier one.

Output: `runs/gemma4-e4b-sft-stage4-mechanism`.

### 5. Verification gate

Re-run the full eval harness on the 58-row test set, zero-shot:

| Metric | Stage 3 (baseline) | Stage 4 target |
|--------|---------|----------------|
| `pause_f1` | 0.167 | **> 0.55** |
| `clear_accuracy` | 0.959 | **≥ 0.85** |
| `pause_recall` | 0.111 | **> 0.65** |
| `gen_overall` | 4.19 | **≥ 4.0** |

**Catastrophic-forgetting gate** (all must hold):
- Stage 1 smoke queries from `smoke_batch_sft.py` still produce well-formed
  DR-Tulu traces (tool calls, cites, answer tags).
- `gen_overall` on clear rows ≥ 4.0.
- `avg_tool_calls` on clear rows ≥ 2.0 (not collapsed to near-zero).

If any gate fails → rollback to earlier Stage 4 checkpoint; if every
checkpoint fails → revert operating model to Stage 3 and iterate on data.

## File deliverables

| Path | Action |
|------|--------|
| `SFT_train/prep_stage4_mechanism_data.py` | **new** — data pipeline |
| `BaseAgent/Data/sft_combined/stage4_pause.jsonl` | **new** — ~1050 vague rows |
| `BaseAgent/Data/sft_combined/stage4_clear.jsonl` | **new** — ~1950 clear rows |
| `BaseAgent/Data/sft_combined/stage4_train.jsonl` | **new** — ~8161-row mix |
| `agent/dr_agent/shared_prompts/unified_tool_calling_cli_stage4.yaml` | **new** |
| `agent/dr_agent/shared_prompts/unified_tool_calling.py` | register `cli_stage4` |
| `agent/workflows/auto_search_sft_stage4.yaml` | **new** — inference config |
| `SFT_train/train_sft_stage4.sbatch`, `train_sft_stage4_distributed.sh` | **new** |
| `agent/evaluation/ambiguity_eval/run_eval.py` | add `stage4-mechanism` tag |

## Runbook

```bash
# 1. Generate data (login node; needs OPENAI_API_KEY)
cd /home/soumya/reasoning/SFT_train
python prep_stage4_mechanism_data.py \
    --out-dir /home/soumya/reasoning/BaseAgent/Data/sft_combined \
    --n-pairs 1500 --judge-model gpt-4o

# 2. Train (user submits)
sbatch SFT_train/train_sft_stage4.sbatch

# 3. Serve stage4 checkpoint
sbatch --export=ALL,MODEL_PATH=/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage4-mechanism,PORT=30001 \
       scripts/serve_gemma_sft_matrix.sbatch

# 4. Evaluate
sbatch --export=ALL,VLLM_NODE=n15,MCP_NODE=n16,OPENAI_API_KEY=$OPENAI_API_KEY,MODEL_TAG=stage4-mechanism,STRATEGY=zero_shot \
       scripts/run_eval_sft.sbatch
```

## Risks

- **GPT-4o mis-labels mechanism specificity**: 20% judge disagreement
  expected; filter aggressively (reject borderline cases).
- **Over-correction toward clear**: we over-weighted clears to fix over-pause.
  Watch `pause_recall` — if it drops below 0.5, rebalance to 55/45.
- **Stage 3 knowledge drift**: 37% new-data share is conservative but still a
  risk. Early-stopping + checkpoint-keep mitigates.
- **Domain concentration**: seed corpus is hardware-systems-heavy. If later
  test distributions add biomedical or social-science queries, we may need
  another narrow stage.
