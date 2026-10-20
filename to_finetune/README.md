# Fine-tuning on Plaksha HPC (48 GB RTX PRO 6000 slice)

LoRA fine-tune of **Qwen3-VL-4B-Instruct** (bf16) on `data/train.jsonl` (47.5k legal instruction pairs) with checkpoints,
auto-resume and a `progress.json` status file, so short Slurm wall-times are not a problem.

```
to_finetune/
  train.py          trainer (checkpoints, time-budget stop, signal-safe, progress.json, resume)
  job.sh            Slurm script: partition rtx, gpu:2g.48gb:1, 12 CPU, 64 GB RAM, self-resubmits until done
  setup_env.sh      one-time venv (uses the cluster's torch module)
  requirements.txt  extra pip packages (no torch)
  modeling.py       model loader used by train.py / merge_adapter.py
  status.py         pretty-print progress.json
  merge_adapter.py  merge the finished LoRA into the base model (run on your laptop or the HPC)
  data/train.jsonl  47,500 examples (147 MB)      data/val.jsonl  2,500 examples (8 MB)
  logs/             Slurm stdout/stderr
```

## Model: Qwen3-VL-4B-Instruct
`Qwen3-VL-4B-Instruct` takes ~8 GB in bf16, fitting comfortably on the 48 GB RTX 6000 slice with batch 8 x 4096 tokens.
On the cluster it is located at: `$SCRATCH/models/Qwen3-VL-4B-Instruct`.

## 1. Copy to the cluster (from your laptop)
```powershell
scp -r "D:\Drive E\Ishan\Plaksha\IDEA for hackathon\RAG\to_finetune" USER@10.1.45.201:~/
# if the connection drops, re-run with rsync (resumable):
# rsync -avP to_finetune/ USER@10.1.45.201:~/to_finetune/
```

## 2. One-time setup (login node)
```bash
cd ~/to_finetune
sed -i 's/\r$//' *.sh          # Windows -> Linux line endings (needed if bash complains about '\r')
bash setup_env.sh
```

## 3. Put the base model on /scratch
Either download it on the login node (if it has internet):
```bash
source ~/envs/legal-ft/bin/activate
mkdir -p /scratch/$USER/models
huggingface-cli download Qwen/Qwen3-8B --local-dir /scratch/$USER/models/Qwen3-8B
```
or upload your local copy (`RAG\Models\Qwen3-8B`, ~16 GB):
```powershell
scp -r "D:\Drive E\Ishan\Plaksha\IDEA for hackathon\RAG\Models\Qwen3-8B" USER@10.1.45.201:/scratch/USER/models/
```
/scratch is purged: if the model disappears, copy it again.

## 4. Quick test (optional, ~10 min), then the real run
```bash
mkdir -p logs
srun -p rtx --qos=general --gres=gpu:2g.48gb:1 --cpus-per-task=12 --mem=64G --time=00:30:00 --pty bash
module load python; source ~/envs/legal-ft/bin/activate
python3 train.py --model /scratch/$USER/models/Qwen3-8B --out /scratch/$USER/test_run --limit 400 --max-steps 20 --ckpt-every 10
exit
```
Real run:
```bash
cd ~/to_finetune && mkdir -p logs
sbatch job.sh
```
Change the wall-time by editing `#SBATCH --time=` (must be <= your QOS limit; `course` = 4 h). Everything else adapts.

## 5. Monitor
```bash
squeue -u $USER
python3 status.py                       # reads ~/legal_lora_out/*/progress.json
tail -f logs/legal-lora-<jobid>.out
```
`progress.json` has: status (`running` / `paused_time_budget` / `paused_signal_*` / `completed` / `failed`), step,
epoch, % done, train/eval loss, eval history, best + last checkpoint, ETA, total time across all sessions.

## 6. How the time limit is handled
* Checkpoint (weights, optimizer, scheduler, RNG, data position) every 200 steps, newest 3 kept in
  `/scratch/$USER/legal_lora/legal-lora-8b/`; the small LoRA adapter + `progress.json` are mirrored to
  `~/legal_lora_out/legal-lora-8b/` after every checkpoint (home is backed up).
* ~12 min before the wall-time (and on Slurm's 10-min warning / `scancel`) training saves a checkpoint and stops.
* `job.sh` then submits itself again (up to `MAX_CHAIN=6`) and the next job resumes exactly where it stopped.
  When `status` is `completed` the chain stops. To continue manually: `sbatch job.sh`.
* Credits: a 48 GB slice costs 1.0 credit/hour (monthly: paid 250 h, faculty 150 h, rfphd 75 h, course 30 h).
  The run is roughly 4-8 GPU-hours; check `ETA` in `status.py` after the first few hundred steps and lower
  `EPOCHS` if needed: `sbatch --export=ALL,EPOCHS=1 job.sh`.

## 7. Bring the result back
```powershell
scp -r USER@10.1.45.201:~/legal_lora_out/legal-lora-8b/final_adapter "D:\Drive E\Ishan\Plaksha\IDEA for hackathon\RAG\legal_lora_out\"
python to_finetune/merge_adapter.py --model Models/Qwen3-8B --adapter legal_lora_out/final_adapter --out Models/Qwen3-8B-legal --val to_finetune/data/val.jsonl --sample 3
```
Then add `"Qwen3-8B-legal": MODELS_DIR / "Qwen3-8B-legal"` to `AVAILABLE_MODELS` in `src/llm.py`.

## Tuning knobs (`sbatch --export=ALL,NAME=value job.sh`)
`MODEL RUN_NAME EPOCHS LR BATCH ACCUM MAXLEN CKPT_EVERY LORA_R MARGIN_MIN MAX_CHAIN`.
If you hit `CUDA out of memory`: `BATCH=4 ACCUM=4` (same effective batch).
