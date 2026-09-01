#!/bin/bash
#SBATCH --job-name=GemSFTServe
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gemma_sft_serve_%j.out
#SBATCH --error=logs/gemma_sft_serve_%j.err

# === Preserve SLURM paths before module purge ===
SLURM_BIN_DIR="$(dirname "$(command -v scontrol 2>/dev/null || echo /usr/bin/scontrol)")"

# === Load Modules ===
module purge
module load cuda/12.8 \
            anaconda3/2023.09-0-none-none
export PATH="${SLURM_BIN_DIR}:${PATH}"

# === Conda Environment ===
source /mnt/beegfs/optLocal/spack/opt/spack/linux-zen4/anaconda3-2023.09-0-w2d4vg6kvki7tlrqcnobbh5nixrvyg6n/etc/profile.d/conda.sh
conda activate GRPO

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# === Config ===
MODEL_PATH=/home/soumya/reasoning/SFT_train/runs/gemma4-e4b-sft-stage3-reasoned
PORT=30001
MAX_MODEL_LEN=32768

cd /home/soumya/reasoning/ProPauseLM-Playground/agent
mkdir -p logs

NODE=$(hostname -s)
echo "=========================================="
echo "Serving Gemma-4-E4B SFT Stage3 on node: $NODE"
echo "Port: $PORT"
echo "Model path: $MODEL_PATH"
echo "=========================================="
echo ""
echo "To test from a login node or other shell:"
echo "  ssh -L $PORT:$NODE:$PORT <cluster>   # forward the port"
echo "  curl http://localhost:$PORT/v1/models"
echo "  curl -X POST http://localhost:$PORT/v1/chat/completions \\"
echo "       -H 'Content-Type: application/json' \\"
echo "       -d '{\"model\":\"'$MODEL_PATH'\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'"
echo ""

# === Launch vLLM server ===
vllm serve "$MODEL_PATH" \
    --port $PORT \
    --dtype auto \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

echo "vLLM server exited (exit=$?)."
