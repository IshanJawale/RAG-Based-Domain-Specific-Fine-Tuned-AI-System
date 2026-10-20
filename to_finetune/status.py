"""
status.py - human-readable view of progress.json (works on the HPC and after you copy it home).

    python3 status.py                         # looks in ~/legal_lora_out/*/progress.json
    python3 status.py path/to/progress.json
"""
import glob
import json
import os
import sys


def fmt(sec):
    if sec is None:
        return "?"
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m"


paths = sys.argv[1:] or sorted(glob.glob(os.path.expanduser("~/legal_lora_out/*/progress.json")))
if not paths:
    sys.exit("No progress.json found. Pass a path.")
for p in paths:
    d = json.load(open(p, encoding="utf-8"))
    print(f"\n=== {p}")
    print(f"status        : {d['status']}   ({d.get('note', '')})")
    print(f"progress      : step {d['global_step']}/{d['max_steps']}  ({d['percent_done']}%)   epoch {d['epoch']}/{d['num_train_epochs']}")
    print(f"train loss    : {d.get('latest_train_loss')}    eval loss: {d.get('latest_eval_loss')}    best eval: {d.get('best_eval_loss')}")
    print(f"checkpoints   : last={d.get('last_checkpoint')}  best={os.path.basename(d.get('best_checkpoint') or '') or None}")
    print(f"time          : this session {fmt(d.get('session_elapsed_sec'))} | total {fmt(d.get('total_elapsed_sec'))} "
          f"| ETA (this rate) {fmt(d.get('eta_this_session_sec'))} | sessions {d.get('sessions')}")
    print(f"slurm job     : {d.get('slurm_job_id')} on {d.get('host')}   updated {d.get('updated_at')}")
    ev = d.get("eval_history", [])
    if ev:
        print("eval history  : " + "  ".join(f"{e['step']}:{e['eval_loss']:.4f}" for e in ev[-8:]))
    if d["status"] == "completed":
        print(f"DONE -> {d.get('final_adapter')}   final_eval={d.get('final_eval')}")
    elif d["status"].startswith("paused"):
        print("PAUSED -> a continuation job is auto-submitted; or run: sbatch job.sh")
