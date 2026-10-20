import os
import lancedb
import pyarrow as pa
import json
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

DATA_DIR = Path(r"d:\Drive E\Ishan\Plaksha\IDEA for hackathon\RAG\data")
DB_PATH = DATA_DIR / "lancedb"

print("Loading embedding model (pure transformers)...")
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5', local_files_only=True)
model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5', local_files_only=True)
model.eval()

# Check for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def get_embeddings(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # Normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().numpy().tolist()

schema = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("source_type", pa.string()),
    pa.field("act_id", pa.string()),
    pa.field("act_title", pa.string()),
    pa.field("jurisdiction", pa.string()),
    pa.field("section_number", pa.string()),
    pa.field("section_heading", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 384))
])

def build_db():
    db = lancedb.connect(DB_PATH)
    
    # Create or overwrite table
    table = db.create_table("acts", schema=schema, mode="overwrite")
    
    chunks_file = DATA_DIR / "acts_chunks.jsonl"
    print(f"Reading chunks from {chunks_file}...")
    
    data = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            chunk = json.loads(line)
            data.append(chunk)
            
    print(f"Loaded {len(data)} chunks.")
    
    batch_size = 512
    for i in tqdm(range(0, len(data), batch_size), desc="Embedding & Ingesting"):
        batch = data[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = get_embeddings(texts)
        
        for j, chunk in enumerate(batch):
            chunk["vector"] = embeddings[j]
            
        table.add(batch)
        
    print("Done building LanceDB index!")

if __name__ == "__main__":
    build_db()
