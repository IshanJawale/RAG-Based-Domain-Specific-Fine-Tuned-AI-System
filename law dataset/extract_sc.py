"""
extract_sc.py

Extracts all Supreme Court judgment PDFs from year-wise tar archives (1950 - 2025).
Safe, resumable, and organized neatly into year-wise folders.
"""

import os
import glob
import tarfile
from pathlib import Path

# Base directory for the Supreme Court data
BASE_DIR = Path(__file__).resolve().parent / "supreme_court"

# Target subfolder for extracted PDFs inside each year
# e.g., supreme_court/1950/pdfs/1950_1_459_519_EN.pdf
EXTRACT_SUBDIR = "pdfs"

def extract_archive(tar_path: Path):
    year_dir = tar_path.parent
    dest_dir = year_dir / EXTRACT_SUBDIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with tarfile.open(tar_path, "r:*") as tar:
            members = tar.getmembers()
            total_members = len(members)
            
            # Check how many already exist to allow fast skipping / resuming
            existing = sum(1 for m in members if (dest_dir / Path(m.name).name).exists())
            if existing == total_members and total_members > 0:
                return "skipped", total_members
            
            # Extract only files (prevent directory traversal safety issues)
            for member in members:
                target_file = dest_dir / Path(member.name).name
                if not target_file.exists():
                    # Extract file directly to dest_dir
                    f_obj = tar.extractfile(member)
                    if f_obj is not None:
                        with open(target_file, "wb") as out_f:
                            out_f.write(f_obj.read())
                            
            return "extracted", total_members
            
    except Exception as e:
        return f"error: {e}", 0

def main():
    print(f"Scanning for Supreme Court tar files in: {BASE_DIR}")
    tar_files = sorted(BASE_DIR.glob("*/english.tar"))
    
    if not tar_files:
        print("No 'english.tar' files found in the supreme_court subdirectories.")
        return

    print(f"Found {len(tar_files)} year archives to process.\n")
    
    # Try rich for progress tracking, fallback to plain prints
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
        use_rich = True
    except ImportError:
        use_rich = False

    total_extracted_files = 0
    total_skipped_years = 0
    
    if use_rich:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total} years)"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Extracting Supreme Court archives...", total=len(tar_files))
            
            for tar_path in tar_files:
                year = tar_path.parent.name
                progress.update(task, description=f"[cyan]Year {year}...")
                status, count = extract_archive(tar_path)
                
                if status == "skipped":
                    total_skipped_years += 1
                elif status == "extracted":
                    total_extracted_files += count
                else:
                    progress.console.print(f"[red][FAIL] Year {year}: {status}[/red]")
                    
                progress.update(task, advance=1)
    else:
        for i, tar_path in enumerate(tar_files, start=1):
            year = tar_path.parent.name
            print(f"[{i}/{len(tar_files)}] Processing Year {year}...", end=" ", flush=True)
            status, count = extract_archive(tar_path)
            if status == "skipped":
                total_skipped_years += 1
                print(f"Already extracted ({count} files)")
            elif status == "extracted":
                total_extracted_files += count
                print(f"Extracted {count} files")
            else:
                print(f"FAILED: {status}")

    print("\n" + "=" * 60)
    print("Supreme Court extraction complete!")
    print(f"Years processed : {len(tar_files)}")
    print(f"Years skipped   : {total_skipped_years} (already extracted)")
    print(f"PDFs organized in: supreme_court/<YEAR>/pdfs/")
    print("=" * 60)

if __name__ == "__main__":
    main()
