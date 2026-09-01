#!/bin/bash
#SBATCH --job-name=SFT-Eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --output=logs/eval_sft_%j.out
#SBATCH --error=logs/eval_sft_%j.err

# ============================================================
# Run the ambiguity-aware eval harness against already-running
# vLLM (stage-3 SFT checkpoint) + MCP servers.
#
# Usage:
#   sbatch --export=ALL,VLLM_NODE=n15,MCP_NODE=n16 \
#          scripts/run_eval_sft.sbatch
#
# Optional overrides:
#   MODEL_TAG   (default: stage3-reasoned)
#   STRATEGY    (default: zero_shot)
#   TEST_SET    (default: Data/TestData/ambiguity_eval_v1.jsonl)
#   MAX_CASES   (default: 0 = all rows)
#   SKIP_JUDGE  (default: 0; set to 1 to skip GPT-4o judge calls)
#
# Requires OPENAI_API_KEY in the submitting shell's env.
# ============================================================

SLURM_BIN_DIR="$(dirname "$(command -v scontrol 2>/dev/null || echo /usr/bin/scontrol)")"

module purge
module load anaconda3/2023.09-0-none-none
export PATH="${SLURM_BIN_DIR}:${PATH}"

source /mnt/beegfs/optLocal/spack/opt/spack/linux-zen4/anaconda3-2023.09-0-w2d4vg6kvki7tlrqcnobbh5nixrvyg6n/etc/profile.d/conda.sh
conda activate dr_agent

export PYTHONUNBUFFERED=1

cd /home/soumya/reasoning/ProPauseLM-Playground/agent
mkdir -p logs

VLLM_NODE=${VLLM_NODE:-n15}
VLLM_PORT=${VLLM_PORT:-30001}
MCP_NODE=${MCP_NODE:-n16}
MCP_PORT=${MCP_PORT:-8000}
MODEL_TAG=${MODEL_TAG:-stage3-reasoned}
STRATEGY=${STRATEGY:-zero_shot}
TEST_SET=${TEST_SET:-/home/soumya/reasoning/BaseAgent/Data/TestData/ambiguity_eval_v1.jsonl}
MAX_CASES=${MAX_CASES:-0}
SKIP_JUDGE=${SKIP_JUDGE:-0}
TEMPERATURE=${TEMPERATURE:-}

RESULTS_DIR=/home/soumya/reasoning/BaseAgent/Data/TestData/results
mkdir -p "$RESULTS_DIR"
OUT="${RESULTS_DIR}/${MODEL_TAG}_${STRATEGY}_${SLURM_JOB_ID}.jsonl"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "[warn] OPENAI_API_KEY is not set in the job env."
  if [ "$SKIP_JUDGE" != "1" ]; then
    echo "       Judge calls will fail. Re-submit with --export=ALL,OPENAI_API_KEY=..."
    echo "       or set SKIP_JUDGE=1 to skip judge scoring."
    exit 2
  fi
fi

echo "=========================================="
echo "Ambiguity SFT Eval"
echo "  model_tag:  $MODEL_TAG"
echo "  strategy:   $STRATEGY"
echo "  test_set:   $TEST_SET"
echo "  vLLM:       $VLLM_NODE:$VLLM_PORT"
echo "  MCP:        $MCP_NODE:$MCP_PORT"
echo "  max_cases:  $MAX_CASES (0 = all)"
echo "  skip_judge: $SKIP_JUDGE"
echo "  output:     $OUT"
echo "=========================================="

EXTRA_ARGS=()
if [ "$MAX_CASES" -gt 0 ]; then
  EXTRA_ARGS+=(--max-cases "$MAX_CASES")
fi
if [ "$SKIP_JUDGE" = "1" ]; then
  EXTRA_ARGS+=(--skip-judge)
fi
if [ -n "$TEMPERATURE" ]; then
  EXTRA_ARGS+=(--temperature "$TEMPERATURE")
fi

python -m evaluation.ambiguity_eval.run_eval \
    --model-tag "$MODEL_TAG" \
    --strategy "$STRATEGY" \
    --test-set "$TEST_SET" \
    --out "$OUT" \
    --vllm-node "$VLLM_NODE" --vllm-port "$VLLM_PORT" \
    --mcp-node "$MCP_NODE" --mcp-port "$MCP_PORT" \
    "${EXTRA_ARGS[@]}"

exit_code=$?
echo "run_eval exited with code: $exit_code"
exit $exit_code
