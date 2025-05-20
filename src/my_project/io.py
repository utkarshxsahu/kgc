import os
import json
import pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def discover_csvs(input_dir):
    return [os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith('.csv') and 'templates' not in f]

def load_csv(path):
    return pd.read_csv(path)

def save_csv(df, path):
    df.to_csv(path, index=False)

def write_metadata(output_dir, info: dict):
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as fp:
        json.dump(info, fp, indent=2)
