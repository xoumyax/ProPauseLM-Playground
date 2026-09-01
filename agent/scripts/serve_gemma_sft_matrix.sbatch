#!/bin/bash
#SBATCH --job-name=GemSFTMatrix
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gemma_sft_matrix_%j.out
#SBATCH --error=logs/gemma_sft_matrix_%j.err

# Parameterized serve for the eval matrix.
# Pass MODEL_PATH and PORT via sbatch --export=, e.g.:
#   sbatch --export=ALL,MODEL_PATH=/path/to/ckpt,PORT=30001 \
#          scripts/serve_gemma_sft_matrix.sbatch

SLURM_BIN_DIR="$(dirname "$(command -v scontrol 2>/dev/null || echo /usr/bin/scontrol)")"

module purge
module load cuda/12.8 \
            anaconda3/2023.09-0-none-none
export PATH="${SLURM_BIN_DIR}:${PATH}"

source /mnt/beegfs/optLocal/spack/opt/spack/linux-zen4/anaconda3-2023.09-0-w2d4vg6kvki7tlrqcnobbh5nixrvyg6n/etc/profile.d/conda.sh
conda activate GRPO

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

: "${MODEL_PATH:?MODEL_PATH must be set via --export}"
: "${PORT:=30001}"
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

cd /home/soumya/reasoning/ProPauseLM-Playground/agent
mkdir -p logs

NODE=$(hostname -s)
echo "=========================================="
echo "Serving SFT checkpoint for eval matrix"
echo "Node:       $NODE"
echo "Port:       $PORT"
echo "Model:      $MODEL_PATH"
echo "Max tokens: $MAX_MODEL_LEN"
echo "=========================================="

vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code

echo "vLLM server exited (exit=$?)."
