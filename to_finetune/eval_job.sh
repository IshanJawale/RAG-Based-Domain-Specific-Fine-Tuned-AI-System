#!/bin/bash
# =====================================================================================================
# Plaksha HPC - Master Indian Legal Evaluation Benchmark on ONE 48 GB RTX PRO 6000 GPU Slice
#
# Usage (from ~/to_finetune):
#   sbatch eval_job.sh
#
# Monitor progress:
#   squeue -u $USER
#   tail -f logs/eval-<jobid>.out
# =====================================================================================================
#SBATCH --job-name=legal-eval
#SBATCH --partition=rtx
#SBATCH --qos=general
#SBATCH --gres=gpu:2g.48gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR"
mkdir -p logs evaluation_report

# Load Python environment
module load python
source "$HOME/envs/legal-ft/bin/activate"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=================================================================="
echo "Master Evaluation Job started on $(hostname) at $(date)"
echo "CUDA Device:"
nvidia-smi -L
echo "Python environment:"
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "=================================================================="

# Run 100-sample unified benchmark
python3 evaluate_cluster.py --num-samples 100 --quant auto "$@"

echo "=================================================================="
echo "Benchmark completed successfully at $(date)"
echo "Reports generated in: $SUBMIT_DIR/evaluation_report/"
echo "=================================================================="
