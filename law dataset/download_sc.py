import subprocess
import os

OUTPUT = "supreme_court"

os.makedirs(OUTPUT, exist_ok=True)

for year in range(1950, 2026):

    year_dir = os.path.join(OUTPUT, str(year))
    os.makedirs(year_dir, exist_ok=True)

    archive = os.path.join(
        year_dir,
        "english.tar"
    )

    # Skip if already downloaded
    if os.path.exists(archive):
        print(f"Skipping {year} - already downloaded")
        continue

    s3_path = (
        f"s3://indian-supreme-court-judgments/"
        f"data/tar/year={year}/english/english.tar"
    )

    print(f"\nDownloading {year}...")

    result = subprocess.run([
        "aws",
        "s3",
        "cp",
        s3_path,
        archive,
        "--no-sign-request"
    ])

    if result.returncode != 0:
        print(f"FAILED: {year}")

print("\nFinished.")