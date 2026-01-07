#!/usr/bin/env python3
"""
Download Qasper dataset from Hugging Face Parquet files.
"""

import json
import urllib.request
from pathlib import Path

print("Downloading Qasper dataset from Hugging Face (Parquet format)...")
print("This may take a few minutes...\n")

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Hugging Face converted the dataset to Parquet format
# We can download these directly
parquet_urls = {
    "train": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/train/0000.parquet",
    "validation": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/validation/0000.parquet",
    "test": "https://huggingface.co/datasets/allenai/qasper/resolve/refs%2Fconvert%2Fparquet/qasper/test/0000.parquet",
}

try:
    import pandas as pd
    import pyarrow.parquet as pq
    import numpy as np
    
    def convert_to_python_types(obj):
        """Recursively convert numpy/pandas types to native Python types."""
        if isinstance(obj, (np.ndarray, pd.Series)):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_python_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    for split, url in parquet_urls.items():
        print(f"Downloading {split} split...")
        
        # Download parquet file
        parquet_file = data_dir / f"temp_{split}.parquet"
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')
        
        with urllib.request.urlopen(req, timeout=60) as response:
            with open(parquet_file, 'wb') as f:
                f.write(response.read())
        
        # Read parquet and convert to original JSON format
        df = pd.read_parquet(parquet_file)
        
        output = {}
        for _, row in df.iterrows():
            paper_id = str(row['id'])
            # Convert all values to native Python types
            output[paper_id] = convert_to_python_types({
                'title': row['title'],
                'abstract': row['abstract'],
                'full_text': row['full_text'],
                'qas': row['qas']
            })
        
        # Save as JSON in original format
        split_name = "dev" if split == "validation" else split
        output_file = data_dir / f"qasper-{split_name}-v0.3.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # Clean up parquet file
        parquet_file.unlink()
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"✓ Saved {len(output)} papers to {output_file.name} ({file_size:.1f} MB)")
    
    print("\n✅ Download complete!")
    print("\nFiles created:")
    for f in sorted(data_dir.glob("*.json")):
        size = f.stat().st_size / (1024 * 1024)
        with open(f) as file:
            data = json.load(file)
            num_papers = len(data)
        print(f"  - {f.name}: {num_papers} papers ({size:.1f} MB)")

except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("\nPlease install: pip install pandas pyarrow")
    exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    print("\nPlease manually download the files from:")
    print("  https://allenai.org/data/qasper")
    print("\nAnd place them in the 'data/' directory.")
    exit(1)
