#!/usr/bin/env python3
import argparse, os
from my_project.knowledge import run_knowledge

if __name__=="__main__":
    p = argparse.ArgumentParser(
        description="Step 3: Compute entity knowledge scores from LLM outputs"
    )
    p.add_argument('--input-dir', required=True,
                   help="Processed dataset folder (with metadata.json, llm_results.csv, etc.)")
    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"No such folder: {args.input_dir}")
    run_knowledge(args.input_dir)
