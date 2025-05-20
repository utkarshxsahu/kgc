#!/usr/bin/env python3
import argparse, os
from my_project.gnn import run_gnn

if __name__=="__main__":
    p = argparse.ArgumentParser(
        description="Step 4: Train GraphSAGE & predict entity kg_values"
    )
    p.add_argument('--input-dir', required=True,
                   help="Processed dataset folder (with metadata.json, entity_kg_values.csv, etc.)")
    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"No such folder: {args.input_dir}")
    run_gnn(args.input_dir)
