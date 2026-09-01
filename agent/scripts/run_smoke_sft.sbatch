#!/bin/bash
#SBATCH --job-name=SFT-Smoke
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/smoke_sft_%j.out
#SBATCH --error=logs/smoke_sft_%j.err

# ============================================================
# Run smoke_batch_sft.py against already-running vLLM + MCP servers
#
# Usage:
#   sbatch --export=VLLM_NODE=n15,MCP_NODE=n16 scripts/run_smoke_sft.sbatch
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
MCP_NODE=${MCP_NODE:-n16}
MAX_QUERIES=${MAX_QUERIES:-2}

echo "=========================================="
echo "SFT Smoke Test"
echo "  vLLM node: $VLLM_NODE"
echo "  MCP node:  $MCP_NODE"
echo "  Queries:   $MAX_QUERIES"
echo "=========================================="

python scripts/smoke_batch_sft.py \
    --vllm-node "$VLLM_NODE" \
    --mcp-node "$MCP_NODE" \
    --max-queries "$MAX_QUERIES"
