#!/usr/bin/env python3
import argparse, os
from my_project.query import run_queries

if __name__=="__main__":
    p = argparse.ArgumentParser(
        description="Step 2: Query LLM over generated triplet_prompts"
    )
    p.add_argument('--input-dir', required=True,
                   help="Path to processed dataset folder (with metadata.json)")
    p.add_argument('--model-dir', default=None,
                   help="Override MODEL_DIR from config if needed")
    args = p.parse_args()
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"No such folder: {args.input_dir}")
    run_queries(args.input_dir, args.model_dir)
