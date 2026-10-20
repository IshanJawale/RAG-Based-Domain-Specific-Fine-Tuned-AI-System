#!/bin/bash
# =====================================================================================================
# Slurm Job: Comparative Evaluation (Base vs Fine-Tuned for 4B & 8B)
# Partition: rtx (1x 48 GB RTX PRO 6000 MIG slice)
# Time: ~45 minutes for 100 comprehensive validation samples across all models
#
# Submit:   sbatch job_eval.sh
# Monitor:  tail -f logs/eval-models-<jobid>.out
# Report:   cat eval_results/evaluation_report.md
# =====================================================================================================
#SBATCH --job-name=eval-models
#SBATCH --partition=rtx
#SBATCH --qos=general
#SBATCH --gres=gpu:2g.48gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR"
mkdir -p logs eval_results

echo "=================================================================="
echo "Starting Comparative Evaluation on $(hostname) at $(date)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "=================================================================="

module load python
if [ -d "$HOME/envs/legal-ft" ]; then
    source "$HOME/envs/legal-ft/bin/activate"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate qwen-vl 2>/dev/null || true
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

nvidia-smi -L

# Run evaluation on 100 strictly unseen legal validation samples
python3 evaluate.py \
  --val-data data/val.jsonl \
  --num-samples 100 \
  --max-gen-tokens 200 \
  --out-dir evaluation_report

echo "=================================================================="
echo "Evaluation Finished at $(date)"
echo "Markdown report generated at: evaluation_report/evaluation_report.md"
echo "JSON metrics generated at: evaluation_report/evaluation_report.json"
echo "=================================================================="
