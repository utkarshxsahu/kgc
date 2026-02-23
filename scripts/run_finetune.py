#!/usr/bin/env python3
"""
Step: Ignorance-guided knowledge injection + fine-tuning  (Part 2)

Examples
--------
# Default (open-source LLM, budget from config):
python scripts/run_finetune.py \
    --input-dir  data/processed/trex \
    --output-dir data/processed/trex/finetune

# With OpenAI for base-model querying:
python scripts/run_finetune.py \
    --input-dir  data/processed/trex \
    --output-dir data/processed/trex/finetune \
    --llm-backend openai \
    --openai-model gpt-3.5-turbo

Pipeline
--------
  1. Select anchor triplets (20% of budget, degree-stratified, capped per entity)
  2. Query anchors against base LLM  → ground-truth knowledge scores
  3. Train GNN on anchor entity scores → predict all entity scores
  4. Ignorance-guided selection of remaining 80% budget triplets
  5. Fine-tune LLM on anchor + selected (combined)
  6. Evaluate on held-out 2% test set  → evaluation_summary.json
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from my_project.finetune import run_finetune


def main():
    p = argparse.ArgumentParser(
        description="Part 2: Ignorance-guided triplet selection + LLM fine-tuning."
    )
    p.add_argument(
        "--input-dir", required=True,
        help="Processed dataset folder (metadata.json, triplet_prompts.csv, raw KG)",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Where to write checkpoints, model, evaluation results",
    )
    p.add_argument(
        "--llm-backend", choices=["opensource", "openai"], default="opensource",
        help="LLM used for anchor querying + evaluation (default: opensource)",
    )
    p.add_argument(
        "--openai-model", default="gpt-3.5-turbo",
        help="OpenAI model name (ignored for opensource backend)",
    )
    p.add_argument(
        "--test-fraction", type=float, default=0.02,
        help="Fraction of all triplets reserved as held-out test set (default: 0.02)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    run_finetune(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        llm_backend=args.llm_backend,
        openai_model=args.openai_model,
        test_fraction=args.test_fraction,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
