#!/usr/bin/env bash
# Matrix eval driver: 4 models × 2 strategies = 8 cells.
#
# The login node cannot submit sbatch jobs directly (user submits them).
# This script PRINTS the serve-submit command for each model, then waits for
# the user to confirm the serve is healthy before running the Python eval
# calls for that model across both strategies. After both strategies finish,
# it prints the scancel cue, waits, then advances to the next model.
#
# Finally, it aggregates all per-cell summaries into one CSV.
#
# Usage:
#   cd /home/soumya/reasoning/ProPauseLM-Playground/agent
#   export OPENAI_API_KEY=sk-...
#   ./scripts/run_eval_matrix.sh \
#       [--vllm-node n15] [--mcp-node n16] [--vllm-port 30001] \
#       [--test-set /path/to/ambiguity_eval_v1.jsonl] \
#       [--results-dir /path/to/results/] [--max-cases 0] [--skip <tag>]

set -euo pipefail

VLLM_NODE="n15"
MCP_NODE="n16"
VLLM_PORT=30001
TEST_SET="/home/soumya/reasoning/BaseAgent/Data/TestData/ambiguity_eval_v1.jsonl"
RESULTS_DIR="/home/soumya/reasoning/BaseAgent/Data/TestData/results"
MAX_CASES=0
SKIP_TAGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm-node)    VLLM_NODE="$2"; shift 2 ;;
        --mcp-node)     MCP_NODE="$2"; shift 2 ;;
        --vllm-port)    VLLM_PORT="$2"; shift 2 ;;
        --test-set)     TEST_SET="$2"; shift 2 ;;
        --results-dir)  RESULTS_DIR="$2"; shift 2 ;;
        --max-cases)    MAX_CASES="$2"; shift 2 ;;
        --skip)         SKIP_TAGS+=("$2"); shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY is not set" >&2
    exit 1
fi

MODELS=(
    "stage1:/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage1"
    "stage1-complete:/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage1-complete"
    "stage2-pause:/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage2-pause"
    "stage3-reasoned:/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage3-reasoned"
)
STRATEGIES=(zero_shot few_shot_cot)

mkdir -p "$RESULTS_DIR"

skip_contains() {
    local needle="$1"
    for t in "${SKIP_TAGS[@]:-}"; do
        [[ "$t" == "$needle" ]] && return 0
    done
    return 1
}

for entry in "${MODELS[@]}"; do
    TAG="${entry%%:*}"
    MPATH="${entry#*:}"

    if skip_contains "$TAG"; then
        echo "=== skipping $TAG (per --skip) ==="
        continue
    fi

    echo ""
    echo "================================================================"
    echo "MODEL: $TAG"
    echo "================================================================"
    echo "1) Submit serve (run this on the login node):"
    echo ""
    echo "   sbatch --export=ALL,MODEL_PATH=$MPATH,PORT=$VLLM_PORT \\"
    echo "          scripts/serve_gemma_sft_matrix.sbatch"
    echo ""
    echo "2) Note the allocated node (update --vllm-node if not $VLLM_NODE)."
    echo ""
    read -r -p "Press enter once the vLLM serve is healthy on node [$VLLM_NODE]… "

    for STRATEGY in "${STRATEGIES[@]}"; do
        OUT="$RESULTS_DIR/${TAG}_${STRATEGY}.jsonl"
        echo ""
        echo "--- run $TAG / $STRATEGY → $OUT ---"
        EXTRA=()
        [[ "$MAX_CASES" != "0" ]] && EXTRA+=(--max-cases "$MAX_CASES")
        python -m evaluation.ambiguity_eval.run_eval \
            --model-tag "$TAG" \
            --strategy "$STRATEGY" \
            --test-set "$TEST_SET" \
            --out "$OUT" \
            --vllm-node "$VLLM_NODE" --vllm-port "$VLLM_PORT" \
            --mcp-node "$MCP_NODE" \
            "${EXTRA[@]}"
    done

    echo ""
    echo "$TAG complete. scancel the serve job now."
    read -r -p "Press enter once scancel confirmed… "
done

echo ""
echo "=== aggregate ==="
python -m evaluation.ambiguity_eval.aggregate --results-dir "$RESULTS_DIR"
