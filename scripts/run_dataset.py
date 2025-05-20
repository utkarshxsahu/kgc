#!/usr/bin/env python3
import argparse
import os
from my_project.io import (
    ensure_dir, discover_csvs,
    load_csv, save_csv, write_metadata
)
from my_project.preprocess import uniform_headers
from my_project.selection import compute_entities, select_triplets
from my_project.prompting import load_templates, build_prompts

def main(input_dir, output_base):
    kg_files = discover_csvs(input_dir)
    for kg_path in kg_files:
        name = os.path.splitext(os.path.basename(kg_path))[0]
        out_dir = os.path.join(output_base, name)
        ensure_dir(out_dir)

        kg = load_csv(kg_path)
        kg = uniform_headers(kg)
        entities = compute_entities(kg)
        sel_ids, sel_entities = select_triplets(kg, entities)


        import pandas as pd
        df_ent = pd.DataFrame({'entity_label': sel_entities})
        save_csv(df_ent, os.path.join(out_dir, 'selected_entities.csv'))

        tpl_path = os.path.join(input_dir, f"{name}_templates.csv")
        templates = load_templates(tpl_path)
        df_prompts = build_prompts(sel_ids, kg, templates)
        save_csv(df_prompts, os.path.join(out_dir, 'triplet_prompts.csv'))

        write_metadata(out_dir, {
            'input_kg'  : kg_path,
            'template'  : tpl_path,
            'entities'  : 'selected_entities.csv',
            'prompts'   : 'triplet_prompts.csv'
        })

        print(f"✔ Finished dataset '{name}'. Outputs in {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run entity selection + prompt generation on a raw KG dataset."
    )
    parser.add_argument('--input-dir', required=True,
                        help="Folder with <dataset>.csv and <dataset>_templates.csv")
    parser.add_argument('--output-base', default='./data/processed',
                        help="Where to create per-dataset output folders")
    args = parser.parse_args()
    main(args.input_dir, args.output_base)
