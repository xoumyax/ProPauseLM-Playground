#!/bin/bash
#SBATCH --job-name=MCPServe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mcp_serve_%j.out
#SBATCH --error=logs/mcp_serve_%j.err

# === Preserve SLURM paths before module purge ===
SLURM_BIN_DIR="$(dirname "$(command -v scontrol 2>/dev/null || echo /usr/bin/scontrol)")"

# === Load Modules ===
module purge
module load anaconda3/2023.09-0-none-none
export PATH="${SLURM_BIN_DIR}:${PATH}"

# === Conda Environment ===
source /mnt/beegfs/optLocal/spack/opt/spack/linux-zen4/anaconda3-2023.09-0-w2d4vg6kvki7tlrqcnobbh5nixrvyg6n/etc/profile.d/conda.sh
conda activate dr_agent

export PYTHONUNBUFFERED=1

# === Config ===
PORT=8000
HOST=0.0.0.0

cd /home/soumya/reasoning/ProPauseLM-Playground/agent
mkdir -p logs

# Load API keys from .env (SERPER, S2, JINA)
set -a
[ -f .env ] && source .env
set +a

NODE=$(hostname -s)
echo "=========================================="
echo "MCP backend on node: $NODE"
echo "Port: $PORT"
echo "=========================================="
echo "To use from another node, set:"
echo "  export MCP_TRANSPORT_HOST=$NODE"
echo "  export MCP_TRANSPORT_PORT=$PORT"
echo ""

python -m dr_agent.mcp_backend.main \
    --transport http \
    --host $HOST \
    --port $PORT \
    --log-level info

echo "MCP server exited (exit=$?)."
