"""
test_client.py - Test your HPC-hosted Legal Model Server from your Laptop
Make sure your SSH tunnel is active before running:
    ssh -N -L 8000:cn02:8000 ishan.jawale.pg26@hpc.plaksha.edu.in
"""

import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"

def test_health():
    print(f"Checking server health at {BASE_URL}/health ...")
    try:
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            print("Server is ONLINE!")
            print(json.dumps(data, indent=2))
            return True
    except (urllib.error.URLError, ConnectionResetError, ConnectionRefusedError) as e:
        print(f"\n[!] Failed to connect: {e}")
        print("\nCommon reasons for this error:")
        print("1. The model is STILL LOADING into GPU memory on the HPC (takes ~45-60s). Check with: cat logs/server-*.out")
        print("2. The SSH tunnel used 'localhost:8000' instead of the compute node name (e.g. 'cn02:8000').")
        print("3. The Slurm job has not started or finished/errored. Check with: squeue -u ishan.jawale.pg26")
        return False
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
        return False

def test_chat():
    print("\nSending test legal query via OpenAI-compatible endpoint (/v1/chat/completions)...")
    payload = {
        "model": "qwen3-vl-legal",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert Indian Legal AI Assistant. Provide accurate statutory citations."
            },
            {
                "role": "user",
                "content": "What are the essential statutory requirements to establish an offence under Section 138 of the Negotiable Instruments Act?"
            }
        ],
        "max_tokens": 300,
        "temperature": 0.2,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            res = json.loads(resp.read().decode())
            elapsed = time.time() - t0
            print("=" * 70)
            print("GENERATED LEGAL ANSWER:")
            print("=" * 70)
            print(res["choices"][0]["message"]["content"])
            print("=" * 70)
            print(f"Total time:       {elapsed:.2f} seconds")
            if "performance" in res:
                print(f"Inference latency: {res['performance']['latency_sec']}s")
                print(f"Throughput:        {res['performance']['tokens_per_sec']} tokens/sec")
            print(f"Tokens generated: {res['usage']['completion_tokens']}")
            print("=" * 70)
    except Exception as e:
        print(f"Error querying model: {e}")

if __name__ == "__main__":
    if test_health():
        test_chat()
