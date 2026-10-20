import os
import subprocess
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

DATA_FILE = 'curated_for_rag.jsonl'
OUTPUT_DIR = 'high_court'
S3_BUCKET_BASE = 's3://indian-high-court-judgments/'

# Configure how many parallel downloads to run
MAX_WORKERS = 50
# Configure how many PDFs to download (set to None for all 779k)
LIMIT = None

# Global flag for Ctrl+C
stop_requested = False

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Create a highly concurrent S3 client with unsigned (anonymous) access
s3_client = boto3.client(
    's3', 
    config=Config(signature_version=UNSIGNED, max_pool_connections=50)
)

def download_pdf(args):
    s3_key, local_path = args
    if stop_requested:
        return False, "Aborted"
        
    if os.path.exists(local_path):
        return True, "Already exists"
        
    try:
        # Direct boto3 download is 50x faster than spawning the AWS CLI via subprocess
        s3_client.download_file('indian-high-court-judgments', s3_key, local_path)
        return True, "Downloaded"
    except Exception as e:
        return False, str(e)

def main():
    global stop_requested
    
    print(f"Reading {DATA_FILE}...")
    import json
    
    s3_downloads = {}
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                link = data.get('pdf_link')
                if not link: continue
                
                year = data.get('partition_year')
                court = data.get('partition_court')
                bench = data.get('partition_bench')
                filename = link.split('/')[-1]
                
                if year and court and bench:
                    s3_key = f"data/pdf/year={year}/court={court}/bench={bench}/{filename}"
                    s3_downloads[filename] = s3_key
                else:
                    s3_key = link # link is already a relative path like court/cnrorders/...
                    s3_downloads[filename] = s3_key
            except json.JSONDecodeError:
                pass
                
    items = list(s3_downloads.items())
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if LIMIT:
        items = items[:LIMIT]
        print(f"Limiting to first {LIMIT} records for testing...")
    
    print(f"Starting download of {len(items)} PDFs...")
    
    tasks = [(s3_path, os.path.join(OUTPUT_DIR, filename)) for filename, s3_path in items]
    
    # Handle Ctrl+C properly to shutdown pool gracefully
    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    
    def signal_handler(sig, frame):
        global stop_requested
        print("\n[Ctrl+C] Stopping downloads... Please wait for active workers to finish.")
        stop_requested = True
        pool.shutdown(wait=False, cancel_futures=True)
        
    signal.signal(signal.SIGINT, signal_handler)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("[cyan]Downloading PDFs...", total=len(tasks))
            
            futures = {pool.submit(download_pdf, args): args for args in tasks}
            
            for future in as_completed(futures):
                if stop_requested:
                    break
                    
                args = futures[future]
                filename = os.path.basename(args[1])
                try:
                    success, msg = future.result()
                    if success:
                        if msg == "Already exists":
                            skip_count += 1
                        else:
                            success_count += 1
                            progress.console.print(f"[green][OK][/green] {filename}")
                    else:
                        fail_count += 1
                        progress.console.print(f"[red][FAIL][/red] {filename} - {msg}")
                except Exception as e:
                    fail_count += 1
                    progress.console.print(f"[red][ERROR][/red] {filename} - {e}")
                    
                progress.update(task_id, advance=1)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        
    print(f"\nDone! Downloaded: {success_count} | Already existed: {skip_count} | Failed: {fail_count}")

if __name__ == '__main__':
    main()
