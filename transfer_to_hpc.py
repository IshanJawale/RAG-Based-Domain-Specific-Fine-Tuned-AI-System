"""
transfer_to_hpc.py - Transfer to_finetune package to Plaksha HPC

Uploads the to_finetune folder to ishan.jawale.pg26@hpc.plaksha.edu.in using scp.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

REMOTE_USER = "ishan.jawale.pg26"
REMOTE_HOST = "hpc.plaksha.edu.in"
REMOTE_DEST = f"{REMOTE_USER}@{REMOTE_HOST}:~/"

ROOT = Path(__file__).resolve().parent
LOCAL_DIR = ROOT / "to_finetune"

def get_dir_size(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 2)

def main():
    print("=" * 70)
    print("  Plaksha HPC - Fine-Tuning Data & Script Transfer Tool")
    print("=" * 70)

    if not LOCAL_DIR.exists():
        sys.exit(f"Error: {LOCAL_DIR} not found. Run from the project root directory.")

    scp_path = shutil.which("scp")
    if not scp_path:
        sys.exit("Error: 'scp' not found in system PATH. Ensure Windows OpenSSH Client is installed.")

    print(f"\nLocal source directory: {LOCAL_DIR}")
    print(f"Total size to upload  : {get_dir_size(LOCAL_DIR)} MB")
    print("\nFiles in package:")
    for f in sorted(LOCAL_DIR.rglob("*")):
        if f.is_file():
            rel = f.relative_to(LOCAL_DIR)
            sz_mb = round(f.stat().st_size / (1024 * 1024), 2)
            print(f"  - {str(rel):<30} ({sz_mb} MB)")

    print(f"\nRemote Destination: {REMOTE_DEST}")
    print("-" * 70)
    print("Starting scp transfer... (Enter your HPC password if prompted)\n")

    cmd = [scp_path, "-r", str(LOCAL_DIR), REMOTE_DEST]

    try:
        ret = subprocess.run(cmd)
        if ret.returncode == 0:
            print("\n" + "=" * 70)
            print("  TRANSFER SUCCESSFUL!")
            print("=" * 70)
            print(f"\nNext Steps on Plaksha HPC:")
            print(f"1. SSH into the cluster:")
            print(f"   ssh {REMOTE_USER}@{REMOTE_HOST}")
            print("\n2. Go to the project folder and set up environment (first time only):")
            print("   cd ~/to_finetune")
            print("   sed -i 's/\\r$//' *.sh")
            print("   bash setup_env.sh")
            print("\n3. Submit the Slurm training job (RTX 6000 48GB slice):")
            print("   mkdir -p logs")
            print("   sbatch job.sh")
            print("\n4. Monitor progress:")
            print("   squeue -u $USER")
            print("   python3 status.py")
            print("   tail -f logs/legal-lora-*.out")
            print("=" * 70)
        else:
            print(f"\n[Error] scp exited with code {ret.returncode}")
            sys.exit(ret.returncode)
    except KeyboardInterrupt:
        print("\n[Transfer cancelled by user]")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error executing scp]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
