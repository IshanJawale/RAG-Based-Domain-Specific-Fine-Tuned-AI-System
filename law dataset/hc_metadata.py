import subprocess
import os

OUTPUT = "high_court_metadata"

os.makedirs(OUTPUT, exist_ok=True)

result = subprocess.run([
    "aws",
    "s3",
    "sync",
    "s3://indian-high-court-judgments/metadata/parquet/",
    OUTPUT,
    "--no-sign-request"
])

if result.returncode == 0:
    print("\nMetadata download complete.")
else:
    print("\nDownload failed.")