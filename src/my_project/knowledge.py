# src/my_project/knowledge.py
import os, json
import pandas as pd
from my_project.config import KNOWLEDGE_CSV
from my_project.io     import load_csv, write_metadata

def compute_knowledge_scores(df_merged, tf_col="tf_value"):
    # stack Sub and Obj as 'entity'
    df_sub = df_merged[['Sub', tf_col]].rename(columns={'Sub':'entity'})
    df_obj = df_merged[['Obj', tf_col]].rename(columns={'Obj':'entity'})
    all_ = pd.concat([df_sub, df_obj], ignore_index=True)
    agg = (
        all_
        .groupby('entity')[tf_col]
        .agg(total=tf_col + '_sum', count='count')
        .rename(columns={f'{tf_col}_sum':'sum'})
        .reset_index()
    )
    agg['kg_value'] = agg['sum'] / agg['count']
    return agg[['entity','kg_value']]

def run_knowledge(input_dir):
    # 1. load metadata
    meta_path = os.path.join(input_dir, "metadata.json")
    with open(meta_path) as fp:
        meta = json.load(fp)

    # 2. load raw KG & llm results & selected entities
    df_kg      = pd.read_csv(meta["input_kg"])
    df_llm     = pd.read_csv(os.path.join(input_dir, meta["prompts"])) \
                      .merge(pd.read_csv(os.path.join(input_dir, "llm_results.csv")),
                             on=["triplet_id","triplet_prompt"])
    df_entities= pd.read_csv(os.path.join(input_dir, "selected_entities.csv"))

    # ensure we have subject/object columns
    df_kg      = df_kg.rename(columns={
        'sub_label':'subject_uri', 'obj_label':'object_uri'
    }) if 'sub_label' in df_kg.columns else df_kg
    df_merged  = (
        df_llm
        .merge(
            df_kg[['triplet_id','subject_uri','object_uri']],
            on='triplet_id', how='left'
        )
        .rename(columns={'subject_uri':'Sub','object_uri':'Obj'})
    )

    # 3. compute scores
    kg_df      = compute_knowledge_scores(df_merged, tf_col='tf_value')

    # 4. restrict to selected entities
    kg_train   = kg_df[kg_df['entity'].isin(df_entities['entity_label'])] \
                        .reset_index(drop=True)

    # 5. save
    out_path = os.path.join(input_dir, KNOWLEDGE_CSV)
    kg_train.to_csv(out_path, index=False)
    print(f"Computed kg_value for {len(kg_train)} entities → {out_path}")
