#!/bin/bash
#SBATCH --job-name=JudgeLlama
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/judge_llama_serve_%j.out
#SBATCH --error=logs/judge_llama_serve_%j.err

# Serves Llama-3.1-70B-Instruct as a vLLM OpenAI-compatible endpoint
# for Stage 2 pause-data generation + verification.

SLURM_BIN_DIR="$(dirname "$(command -v scontrol 2>/dev/null || echo /usr/bin/scontrol)")"

module purge
module load cuda/12.8 \
            anaconda3/2023.09-0-none-none
export PATH="${SLURM_BIN_DIR}:${PATH}"

source /mnt/beegfs/optLocal/spack/opt/spack/linux-zen4/anaconda3-2023.09-0-w2d4vg6kvki7tlrqcnobbh5nixrvyg6n/etc/profile.d/conda.sh
conda activate GRPO

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

MODEL_PATH=/home/soumya/FoundationModels/Llama-3.1-70B-Instruct
PORT=30001
MAX_MODEL_LEN=8192

cd /home/soumya/reasoning/ProPauseLM-Playground/agent
mkdir -p logs

NODE=$(hostname -s)
echo "=========================================="
echo "Judge LLM (Llama-3.1-70B) on node: $NODE"
echo "Port: $PORT"
echo "Model: $MODEL_PATH"
echo "=========================================="

vllm serve "$MODEL_PATH" \
    --port $PORT \
    --dtype auto \
    --tensor-parallel-size 2 \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code

echo "Judge vLLM server exited (exit=$?)."
