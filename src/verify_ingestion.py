import os
import sqlite3
import lancedb
from pathlib import Path
import pyarrow.compute as pc

BASE_DIR = Path(r"d:\Drive E\Ishan\Plaksha\IDEA for hackathon\RAG")
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "lancedb"

print("=" * 60)
print("1. CHECKING SQLITE PROGRESS DATABASES")
print("=" * 60)

for db_file in DATA_DIR.glob("*.db"):
    print(f"\nDatabase: {db_file.name} ({db_file.stat().st_size / (1024*1024):.2f} MB)")
    try:
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        tables = [t[0] for t in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        print(f"  Tables: {tables}")
        for t in tables:
            cnt = c.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            print(f"  - {t}: {cnt:,} rows")
            
            # If completed_files, show chunk statistics
            if t == "completed_files":
                c.execute(f"SELECT sum(chunk_count), count(CASE WHEN chunk_count == 0 THEN 1 END) FROM {t}")
                total_chunks, zero_chunks = c.execute(f"SELECT sum(chunk_count), count(CASE WHEN chunk_count == 0 THEN 1 END) FROM {t}").fetchone()
                print(f"    Total chunks recorded: {total_chunks:,}")
                print(f"    Files with 0 chunks (corrupted/skipped/too short): {zero_chunks:,}")
                
                # Sample few rows
                print("    Sample rows:")
                for row in c.execute(f"SELECT filepath, chunk_count FROM {t} LIMIT 3").fetchall():
                    print(f"      {row}")
            elif t == "file_progress":
                cols = [col[1] for col in c.execute("PRAGMA table_info(file_progress)").fetchall()]
                print(f"    Columns: {cols}")
                for row in c.execute("SELECT * FROM file_progress LIMIT 3").fetchall():
                    print(f"      Sample row: {row}")
                try:
                    status_counts = c.execute("SELECT status, count(*) FROM file_progress GROUP BY status").fetchall()
                    print(f"    Status breakdown: {status_counts}")
                except Exception:
                    pass
        conn.close()
    except Exception as e:
        print(f"  Error reading {db_file.name}: {e}")

print("\n" + "=" * 60)
print("2. CHECKING LANCEDB TABLES & RECORD COUNTS")
print("=" * 60)

db = lancedb.connect(DB_PATH)
tables = ['acts', 'high_court', 'supreme_court']
print(f"Checking tables: {tables}\n")

for tbl_name in tables:
    print(f"--- Table: {tbl_name} ---")
    tbl = db.open_table(tbl_name)
    num_rows = tbl.count_rows()
    print(f"Total Rows: {num_rows:,}")
    
    # Schema check
    schema = tbl.schema
    print(f"Schema columns ({len(schema.names)}): {schema.names}")
    
    # Inspect a small sample
    sample = tbl.head(3).to_pylist()
    if sample:
        print(f"Sample row keys: {list(sample[0].keys())}")
        print(f"Sample source_type: {sample[0].get('source_type')}")
        print(f"Sample text preview: {repr(sample[0].get('text', '')[:120])}")
        vec = sample[0].get("vector")
        if vec is not None:
            print(f"Sample vector length: {len(vec)}, sample values: {vec[:3]}...")
        else:
            print("WARNING: Sample vector is None!")
    print()

print("=" * 60)
print("3. VERIFYING VECTOR RETRIEVAL TEST")
print("=" * 60)

try:
    from transformers import AutoTokenizer, AutoModel
    import torch

    print("Loading embedding model to test vector search...")
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5', local_files_only=True)
    model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5', local_files_only=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    def get_query_vector(text):
        encoded = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**encoded)
            emb = out[0][:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().tolist()[0]

    test_query = "Right to equality and freedom of speech under the Constitution"
    q_vec = get_query_vector(test_query)
    print(f"Test query: '{test_query}'")

    for tbl_name in tables:
        tbl = db.open_table(tbl_name)
        res = tbl.search(q_vec).limit(2).to_list()
        print(f"\nSearch results from '{tbl_name}':")
        for idx, r in enumerate(res, 1):
            title = r.get("act_title") or r.get("case_id") or "N/A"
            dist = r.get("_distance", 0.0)
            preview = (r.get("text") or "")[:150].replace("\n", " ")
            print(f"  [{idx}] Dist: {dist:.4f} | Title: {title} | Preview: {preview}...")
except Exception as e:
    print(f"Vector search test error: {e}")

print("\nVerification script finished successfully.")
