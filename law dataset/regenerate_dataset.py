import os
import glob
import pandas as pd

base = 'd:\\Drive E\\Ishan\\Plaksha\\IDEA for hackathon\\RAG\\law dataset\\high_court_metadata'
files = []
for court in ['court=27_1', 'court=22_18']:
    pattern = os.path.join(base, 'year=*', court, 'bench=*', '*.parquet')
    files.extend(glob.glob(pattern))

substantive_disposals = ['allowed', 'dismissed', 'partly allowed', 'rule made absolute', 'granted']
out_file = 'curated_for_rag.jsonl'
valid_records = 0

print(f"Generating {out_file} (JSONL is much better for LLM fine-tuning than CSV)...")

with open(out_file, 'w', encoding='utf-8') as f:
    for file in files:
        try:
            # Extract partition keys from the filepath
            # e.g. ...\year=2020\court=27_1\bench=newos\metadata.parquet
            parts = file.replace('\\', '/').split('/')
            
            # Find the parts containing year, court, bench
            year_val = None
            court_val = None
            bench_val = None
            for p in parts:
                if p.startswith('year='): year_val = p.split('=')[1]
                elif p.startswith('court='): court_val = p.split('=')[1]
                elif p.startswith('bench='): bench_val = p.split('=')[1]
                
            df = pd.read_parquet(file)
            
            # Inject them
            if year_val: df['partition_year'] = year_val
            if court_val: df['partition_court'] = court_val
            if bench_val: df['partition_bench'] = bench_val
            
            if 'disposal_nature' in df.columns:
                df['disposal_nature_lower'] = df['disposal_nature'].fillna('').str.lower()
                df = df[df['disposal_nature_lower'].isin(substantive_disposals)]
                df = df.drop(columns=['disposal_nature_lower'])
            
            if 'raw_html' in df.columns:
                df = df.drop(columns=['raw_html'])
                
            df = df[df['title'].notna() & (df['title'] != '')]
            df = df[df['pdf_link'].notna() & (df['pdf_link'] != '')]
            
            if not df.empty:
                valid_records += len(df)
                # Write to JSONL
                df.to_json(f, orient='records', lines=True)
                f.write('\n')
        except Exception as e:
            pass

print(f'Done! Found {valid_records} highly useful cases.')
print(f'Saved to {out_file}. You can delete curated_for_rag.csv now.')
