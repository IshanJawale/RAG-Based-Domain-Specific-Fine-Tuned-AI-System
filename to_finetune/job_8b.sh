#!/bin/bash
# =====================================================================================================
# Plaksha HPC - LoRA fine-tuning for Qwen3-VL-8B-Instruct
# Hardware: ONE 48 GB RTX PRO 6000 Blackwell slice (partition rtx, gpu:2g.48gb:1)
#
# Submit (from this folder):   mkdir -p logs && sbatch job_8b.sh
# Monitor:                     squeue -u $USER ; python3 status.py ; tail -f logs/legal-lora-vl-8b-<jobid>.out
# Cancel:                      scancel <jobid>          (a checkpoint is saved first)
# =====================================================================================================
#SBATCH --job-name=legal-8b
#SBATCH --partition=rtx
#SBATCH --qos=general
#SBATCH --gres=gpu:2g.48gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail

# ------------------------------ settings (override with --export) ------------------------------------
RUN_NAME="${RUN_NAME:-legal-lora-vl-8b}"
MODEL="${MODEL:-${SCRATCH:-/scratch/$USER}/models/Qwen3-VL-8B-Instruct}"     # base 8B-VL model folder
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-2}"                                   # micro-batch 2 for 8B on 48GB MIG slice
ACCUM="${ACCUM:-8}"                                   # effective batch = BATCH * ACCUM = 16
MAXLEN="${MAXLEN:-4096}"
CKPT_EVERY="${CKPT_EVERY:-200}"                       # optimizer steps between checkpoint + eval
LORA_R="${LORA_R:-32}"
MARGIN_MIN="${MARGIN_MIN:-15}"                        # stop this many minutes before wall-time
MAX_CHAIN="${MAX_CHAIN:-6}"                           # max automatic re-submissions
CHAIN_COUNT="${CHAIN_COUNT:-0}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR"
OUT="/scratch/$USER/legal_lora/$RUN_NAME"             # checkpoints (big, purgeable)
SYNC="$HOME/legal_lora_out/$RUN_NAME"                 # adapter + progress.json (backed up in /home)
mkdir -p "$OUT" "$SYNC" logs

echo "=================================================================="
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) | chain #$CHAIN_COUNT | started $(date)"
echo "MODEL=$MODEL"
echo "OUT=$OUT  SYNC=$SYNC"
echo "=================================================================="

# ------------------------------ environment ----------------------------------------------------------
module load python
# Activate whichever environment is set up (conda qwen-vl or venv legal-ft)
if [ -d "$HOME/envs/legal-ft" ]; then
    source "$HOME/envs/legal-ft/bin/activate"
elif command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate qwen-vl 2>/dev/null || true
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi -L
python3 -c "import torch,transformers,peft; print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| transformers',transformers.__version__,'| peft',peft.__version__)"
if [ ! -f "$MODEL/config.json" ]; then
  echo "ERROR: base model not found at $MODEL (no config.json)." >&2
  exit 2
fi

# ------------------------------ time budget (read the real limit from Slurm) -------------------------
to_min() {  # D-HH:MM:SS | HH:MM:SS | MM:SS | MM  ->  minutes
  local t="$1" d=0 a b c
  if [[ "$t" == *-* ]]; then d="${t%%-*}"; t="${t#*-}"; fi
  IFS=: read -r a b c <<< "$t"
  if [ -n "${c:-}" ]; then echo $(( d*1440 + 10#$a*60 + 10#$b ))
  elif [ -n "${b:-}" ]; then echo $(( d*1440 + 10#$a ))
  else echo $(( d*1440 + 10#$a )); fi
}
LIMIT_STR="$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %l 2>/dev/null | head -n1 || true)"
if [ -n "$LIMIT_STR" ] && [ "$LIMIT_STR" != "UNLIMITED" ] && [ "$LIMIT_STR" != "INVALID" ]; then
  WALL_MIN="$(to_min "$LIMIT_STR")"
else
  WALL_MIN="${WALL_MIN:-720}"
fi
BUDGET_MIN=$(( WALL_MIN - MARGIN_MIN ))
[ "$BUDGET_MIN" -lt 10 ] && BUDGET_MIN=10
echo "wall-time ${WALL_MIN} min -> training budget ${BUDGET_MIN} min"

# ------------------------------ train (in background so we can forward Slurm's USR1/TERM) ------------
python3 train.py \
  --model "$MODEL" \
  --train data/train.jsonl --val data/val.jsonl \
  --out "$OUT" --sync-dir "$SYNC" \
  --epochs "$EPOCHS" --lr "$LR" --batch-size "$BATCH" --grad-accum "$ACCUM" \
  --max-len "$MAXLEN" --ckpt-every "$CKPT_EVERY" --lora-r "$LORA_R" --lora-alpha $(( LORA_R * 2 )) \
  --time-budget-min "$BUDGET_MIN" &
PY=$!
trap 'echo "[job] USR1 from Slurm (wall-time almost up) -> asking trainer to checkpoint"; kill -USR1 $PY 2>/dev/null' USR1
trap 'echo "[job] TERM -> asking trainer to checkpoint"; kill -USR1 $PY 2>/dev/null' TERM

RC=0
wait "$PY"; RC=$?
while kill -0 "$PY" 2>/dev/null; do wait "$PY"; RC=$?; done
echo "trainer exited with code $RC at $(date)"

# ------------------------------ decide: done, paused (-> resubmit) or failed -------------------------
STATUS="$(python3 - "$OUT/progress.json" <<'EOF'
import json, sys
try:
    print(json.load(open(sys.argv[1]))["status"])
except Exception:
    print("unknown")
EOF
)"
echo "progress.json status: $STATUS"
cp -f "$OUT/progress.json" "$SYNC/progress.json" 2>/dev/null || true

if [ "$RC" -eq 0 ] && [[ "$STATUS" == paused_* ]]; then
  if [ "$CHAIN_COUNT" -lt "$MAX_CHAIN" ]; then
    NEXT=$(( CHAIN_COUNT + 1 ))
    echo "Not finished -> submitting continuation job (chain #$NEXT of $MAX_CHAIN)"
    sbatch --export=ALL,CHAIN_COUNT="$NEXT",RUN_NAME="$RUN_NAME",MODEL="$MODEL",EPOCHS="$EPOCHS",LR="$LR",BATCH="$BATCH",ACCUM="$ACCUM",MAXLEN="$MAXLEN",CKPT_EVERY="$CKPT_EVERY",LORA_R="$LORA_R" \
           "$SUBMIT_DIR/job_8b.sh"
  else
    echo "Reached MAX_CHAIN=$MAX_CHAIN. Run 'sbatch job_8b.sh' again to continue (it resumes automatically)."
  fi
elif [ "$STATUS" == "completed" ]; then
  echo "TRAINING COMPLETE. Final adapter: $SYNC/final_adapter"
else
  echo "Trainer did not finish cleanly (rc=$RC, status=$STATUS). Check logs/*.err . Latest checkpoint is kept in $OUT."
fi
echo "Finished $(date)"
exit "$RC"
