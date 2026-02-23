#!/usr/bin/env python3
"""
Step: Homophily knowledge checking + GraphML export  (Part 1)

Examples
--------
# Sparse mode (default) with open-source LLM:
python scripts/run_homophily.py \
    --input-dir  data/processed/trex \
    --output-dir data/processed/trex/homophily

# Full mode with OpenAI:
python scripts/run_homophily.py \
    --input-dir  data/processed/trex \
    --output-dir data/processed/trex/homophily \
    --mode full \
    --llm-backend openai \
    --openai-model gpt-3.5-turbo

# Sparse mode, custom fraction:
python scripts/run_homophily.py \
    --input-dir  data/processed/trex \
    --output-dir data/processed/trex/homophily \
    --mode sparse \
    --sparse-fraction 0.10
"""

import argparse
import os
import sys

# allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from my_project.homophily import run_homophily


def main():
    p = argparse.ArgumentParser(
        description="Part 1: Compute entity knowledge scores and export GraphML."
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Processed dataset folder (metadata.json, triplet_prompts.csv)",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Where to write outputs (graphml, scores, checkpoints)",
    )
    p.add_argument(
        "--mode", choices=["full", "sparse"], default="sparse",
        help=(
            "full  = query every triplet against LLM; "
            "sparse = query sparse_fraction of triplets, train GNN for the rest "
            "(default: sparse)"
        ),
    )
    p.add_argument(
        "--llm-backend", choices=["opensource", "openai"], default="opensource",
        help="LLM backend to use (default: opensource / unsloth)",
    )
    p.add_argument(
        "--sparse-fraction", type=float, default=0.10,
        help="Fraction of triplets to query in sparse mode (default: 0.10)",
    )
    p.add_argument(
        "--openai-model", default="gpt-3.5-turbo",
        help="OpenAI model name (ignored for opensource backend)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    run_homophily(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        llm_backend=args.llm_backend,
        sparse_fraction=args.sparse_fraction,
        openai_model=args.openai_model,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
