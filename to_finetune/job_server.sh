#!/bin/bash
# =====================================================================================================
# Slurm Job: Legal LLM Inference Server (OpenAI-compatible REST API on Plaksha HPC)
# Partition: rtx (1x 48 GB RTX PRO 6000 MIG slice)
# Time: 12:00:00 (adjustable up to your QOS limit)
#
# How to run:
#   1. Submit this job:           sbatch job_server.sh
#   2. Check which node it is on: squeue -u $USER
#   3. View connection command:   cat logs/server-<jobid>.out
#   4. Open tunnel on laptop:     ssh -N -L 8000:<NODE>:8000 ishan.jawale.pg26@hpc.plaksha.edu.in
# =====================================================================================================
#SBATCH --job-name=legal-server
#SBATCH --partition=rtx
#SBATCH --qos=general
#SBATCH --gres=gpu:2g.48gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/server-%j.out
#SBATCH --error=logs/server-%j.err

set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR"
mkdir -p logs

NODE_NAME=$(hostname -s)
PORT="${SERVER_PORT:-8000}"
MODEL_TAG="${MODEL_TAG:-4b}"
ADAPTER_TAG="${ADAPTER_TAG:-4b-1ep}"

echo "=================================================================="
echo "Starting Legal LLM Inference Server on $NODE_NAME at $(date)"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Model Tag:     $MODEL_TAG"
echo "Adapter Tag:   $ADAPTER_TAG"
echo "Listening:     0.0.0.0:$PORT"
echo "=================================================================="

# Environment activation
module load python
if [ -d "$HOME/envs/legal-ft" ]; then
    source "$HOME/envs/legal-ft/bin/activate"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate qwen-vl 2>/dev/null || true
fi

# Ensure FastAPI & Uvicorn are present
python3 -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "Installing lightweight API dependencies (fastapi, uvicorn)..."
    pip install fastapi uvicorn
}

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

echo "=================================================================="
echo ">>> COPY AND PASTE THIS COMMAND INTO YOUR LAPTOP TERMINAL: <<<"
echo ""
echo "    ssh -N -L 8000:${NODE_NAME}:${PORT} ishan.jawale.pg26@hpc.plaksha.edu.in"
echo ""
echo "Once running, your laptop can access the model directly at:"
echo "    http://localhost:8000/health"
echo "    http://localhost:8000/v1/chat/completions"
echo "=================================================================="

# Launch the server
python3 server.py \
    --host 0.0.0.0 \
    --port "$PORT" \
    --model-tag "$MODEL_TAG" \
    --adapter "$ADAPTER_TAG"
