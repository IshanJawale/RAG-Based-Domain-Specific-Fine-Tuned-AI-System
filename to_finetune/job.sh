#!/bin/bash
# =====================================================================================================
# Plaksha HPC - LoRA fine-tuning on ONE 48 GB RTX PRO 6000 slice  (partition rtx, gpu:2g.48gb:1)
#
# Submit (from this folder):   mkdir -p logs && sbatch job.sh
# Monitor:                     squeue -u $USER ; python3 status.py ; tail -f logs/legal-lora-<jobid>.out
# Cancel:                      scancel <jobid>          (a checkpoint is saved first)
#
# The job is RESUMABLE and SELF-CHAINING:
#   * training stops cleanly ~12 min before the wall-time (and on Slurm's 10-min USR1 warning / scancel),
#   * a checkpoint is written, progress.json says "paused_*",
#   * this script then submits itself again (up to MAX_CHAIN times) and the next job continues from the checkpoint.
# When progress.json says "completed" the chain stops.
#
# Override anything without editing:  sbatch --export=ALL,MODEL=/scratch/$USER/models/Qwen3-8B,EPOCHS=1 job.sh
# =====================================================================================================
#SBATCH --job-name=legal-lora
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
# NOTE: --time must be <= your QOS limit (course = 4 h, rfphd = 5 d, faculty = 3 d, paid = 7 d). The script reads
#       the real limit from Slurm, so just change the line above - nothing else needs to match.

set -uo pipefail

# ------------------------------ settings (override with --export) ------------------------------------
RUN_NAME="${RUN_NAME:-legal-lora-vl-4b}"
MODEL="${MODEL:-${SCRATCH:-/scratch/$USER}/models/Qwen3-VL-4B-Instruct}"     # base model folder
EPOCHS="${EPOCHS:-2}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-4}"                                   # micro batch (4 is rock-solid on 48GB MIG slice)
ACCUM="${ACCUM:-4}"                                   # effective batch = BATCH*ACCUM = 16
MAXLEN="${MAXLEN:-4096}"
CKPT_EVERY="${CKPT_EVERY:-200}"                       # optimizer steps between checkpoint + eval
LORA_R="${LORA_R:-32}"
MARGIN_MIN="${MARGIN_MIN:-12}"                        # stop this many minutes before wall-time
MAX_CHAIN="${MAX_CHAIN:-6}"                           # max automatic re-submissions
CHAIN_COUNT="${CHAIN_COUNT:-0}"

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$SUBMIT_DIR"
OUT="/scratch/$USER/legal_lora/$RUN_NAME"             # checkpoints (big, purgeable)   - SAME path on every chained job
SYNC="$HOME/legal_lora_out/$RUN_NAME"                 # adapter + progress.json (small, backed up)
mkdir -p "$OUT" "$SYNC" logs

echo "=================================================================="
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) | chain #$CHAIN_COUNT | started $(date)"
echo "MODEL=$MODEL"
echo "OUT=$OUT  SYNC=$SYNC"
echo "=================================================================="

# ------------------------------ environment ----------------------------------------------------------
module load python
# module load cuda            # <- only if `module avail` shows you need it for this cluster's torch build
source "$HOME/envs/legal-ft/bin/activate"                   # created by setup_env.sh
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi -L
python3 -c "import torch,transformers,peft; print('torch',torch.__version__,'| cuda',torch.cuda.is_available(),'| transformers',transformers.__version__,'| peft',peft.__version__)"
if [ ! -f "$MODEL/config.json" ]; then
  echo "ERROR: base model not found at $MODEL (no config.json). See README step 3." >&2
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
  WALL_MIN="${WALL_MIN:-480}"
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
while kill -0 "$PY" 2>/dev/null; do wait "$PY"; RC=$?; done   # `wait` returns early when a trapped signal arrives
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
           "$SUBMIT_DIR/job.sh"
  else
    echo "Reached MAX_CHAIN=$MAX_CHAIN. Run 'sbatch job.sh' again to continue (it resumes automatically)."
  fi
elif [ "$STATUS" == "completed" ]; then
  echo "TRAINING COMPLETE. Final adapter: $SYNC/final_adapter"
else
  echo "Trainer did not finish cleanly (rc=$RC, status=$STATUS). Check logs/*.err . Latest checkpoint is kept in $OUT."
fi
echo "Finished $(date)"
exit "$RC"
