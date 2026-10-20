import os
import json
import uuid
from pathlib import Path

# Paths
BASE_DIR = Path(r"d:\Drive E\Ishan\Plaksha\IDEA for hackathon\RAG")
DATA_DIR = BASE_DIR / "data"
LAW_DATASET = BASE_DIR / "law dataset" / "indian_law_dataset"
ACTS_DIR = LAW_DATASET / "acts"
ACTS_INDEX = LAW_DATASET / "acts_index.json"
OUTPUT_FILE = DATA_DIR / "acts_chunks.jsonl"

def build_acts():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Reading acts index from {ACTS_INDEX}...")
    with open(ACTS_INDEX, "r", encoding="utf-8") as f:
        acts = json.load(f)
        
    print(f"Total acts found: {len(acts)}")
    
    total_sections = 0
    total_acts_processed = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for act in acts:
            act_id = act.get("id")
            if not act_id:
                continue
                
            act_file = ACTS_DIR / f"{act_id}.json"
            if not act_file.exists():
                continue
                
            try:
                with open(act_file, "r", encoding="utf-8") as f:
                    act_data = json.load(f)
                    
                sections = act_data.get("sections", [])
                
                for sec in sections:
                    sec_text = sec.get("text")
                    if sec_text is None:
                        sec_text = ""
                    sec_text = sec_text.strip()
                    if not sec_text:
                        continue
                        
                    act_title = act.get("short_title")
                    if act_title is None:
                        act_title = ""
                    act_title = act_title.strip('" ')
                    
                    sec_num = sec.get("number")
                    if sec_num is None:
                        sec_num = ""
                        
                    sec_heading = sec.get("heading")
                    if sec_heading is None:
                        sec_heading = ""
                    
                    # Build chunk record
                    chunk = {
                        "chunk_id": str(uuid.uuid4()),
                        "source_type": "act",
                        "act_id": act_id,
                        "act_title": act_title,
                        "jurisdiction": act.get("jurisdiction"),
                        "section_number": sec_num,
                        "section_heading": sec_heading,
                        "text": f"[{act_title}] Section {sec_num}: {sec_heading}\n\n{sec_text}"
                    }
                    out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    total_sections += 1
                total_acts_processed += 1
                
                if total_acts_processed % 1000 == 0:
                    print(f"Processed {total_acts_processed}/{len(acts)} acts...")
                    
            except Exception as e:
                print(f"Error processing {act_id}: {e}")
                
    print(f"\nDone! Processed {total_acts_processed} acts.")
    print(f"Extracted {total_sections} section chunks.")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_acts()
