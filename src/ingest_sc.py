import os
import json
import uuid
import lancedb
import pyarrow as pa
from pathlib import Path
from tqdm import tqdm
import pymupdf
# Suppress C-level MuPDF errors (like the HiddenHorzOCR font warnings)
pymupdf.TOOLS.mupdf_display_errors(False)

import torch
from transformers import AutoTokenizer, AutoModel
import concurrent.futures

# ------------------------------------------------------------------
# CONFIG & PATHS
# ------------------------------------------------------------------
BASE_DIR = Path(r"d:\Drive E\Ishan\Plaksha\IDEA for hackathon\RAG")
DATA_DIR = BASE_DIR / "data"
SC_PDF_DIR = BASE_DIR / "law dataset" / "supreme_court"
DB_PATH = DATA_DIR / "lancedb"

CHUNK_WORD_LIMIT = 500  # Split long PDFs into ~500-word chunks

# We will initialize these ONLY in the main process to avoid OOM
tokenizer = None
model = None
device = None

def init_model():
    global tokenizer, model, device
    print("Loading embedding model (BGE-small)...")
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5', local_files_only=True)
    model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5', local_files_only=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

def get_embeddings(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().numpy().tolist()

schema = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("source_type", pa.string()),
    pa.field("case_id", pa.string()),
    pa.field("court", pa.string()),
    pa.field("year", pa.string()),
    pa.field("chunk_index", pa.int32()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 384))
])

def extract_chunks_from_pdf(pdf_path):
    """CPU-bound task to extract and chunk text using PyMuPDF."""
    try:
        doc = pymupdf.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
        
        words = full_text.split()
        if not words:
            return []
            
        case_id = pdf_path.stem
        year = pdf_path.parent.parent.name if pdf_path.parent.name == "pdfs" else "unknown"
        
        chunks = []
        for j in range(0, len(words), CHUNK_WORD_LIMIT):
            chunk_words = words[j:j + CHUNK_WORD_LIMIT]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "source_type": "supreme_court",
                "case_id": case_id,
                "court": "Supreme Court of India",
                "year": year,
                "chunk_index": len(chunks),
                "text": f"[Supreme Court of India - {year}] {case_id}\n\n{chunk_text}",
                "vector": None
            })
        return chunks
    except Exception:
        return []

# ------------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------------
def process_sc_cases():
    db = lancedb.connect(DB_PATH)
    
    print("Creating/Overwriting Supreme Court table in LanceDB...")
    table = db.create_table("supreme_court", schema=schema, mode="overwrite")
        
    print(f"Scanning for PDFs in {SC_PDF_DIR} ...")
    pdf_files = list(SC_PDF_DIR.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} Supreme Court PDFs.")
    
    total_chunks_processed = 0
    batch_records = []
    
    # We MUST use ProcessPoolExecutor! PyMuPDF text extraction + Python string
    # manipulation is CPU-heavy and thrashes the GIL if we use threads.
    print("Extracting and embedding cases in parallel (multi-processing)...")
    
    # Process in chunks to avoid submitting 43,000 futures at once (which eats RAM)
    batch_records = []
    total_chunks_processed = 0
    
    # Safe limits for laptop RAM/VRAM to prevent paging file exhaustion
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        # We use executor.map which is memory efficient
        # It yields results in order, which is perfectly fine.
        results_iterator = executor.map(extract_chunks_from_pdf, pdf_files, chunksize=50)
        
        for chunks in tqdm(results_iterator, total=len(pdf_files), desc="Processing PDFs"):
            if not chunks:
                continue
                
            # Now pass to GPU for embedding (main process handles GPU to avoid CUDA IPC errors)
            texts = [c["text"] for c in chunks]
            
            # Sub-batch embedding to avoid OOM
            for k in range(0, len(texts), 128):
                sub_texts = texts[k:k+128]
                sub_embeds = get_embeddings(sub_texts)
                for idx, emb in enumerate(sub_embeds):
                    chunks[k+idx]["vector"] = emb
                    
            batch_records.extend(chunks)
            total_chunks_processed += len(chunks)
            
            if len(batch_records) >= 2000:
                table.add(batch_records)
                batch_records = []
                
    # Flush remaining
    if batch_records:
        table.add(batch_records)
        
    print(f"Done! Extracted and embedded {total_chunks_processed} chunks from Supreme Court.")

if __name__ == "__main__":
    init_model()
    process_sc_cases()
