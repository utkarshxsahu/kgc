"""
homophily.py
------------
Part 1 orchestration: compute entity knowledge scores and export graphml.

Two modes
---------
  "full"   – query every triplet against the LLM, then aggregate to entity scores.
  "sparse" – query only `sparse_fraction` of triplets as anchors, train GNN to
             predict scores for all entities, then export both observed+predicted.

Two LLM backends
----------------
  "openai"     – uses the OpenAI chat completions API (requires OPENAI_API_KEY env var)
  "opensource" – uses a local unsloth model (default)

Usage (library)
---------------
    from my_project.homophily import run_homophily
    run_homophily(input_dir, output_dir, mode="sparse", llm_backend="opensource")

Usage (CLI via scripts/run_homophily.py)
"""

from __future__ import annotations

import os
import json
import time
import re
import pickle as pkl
from typing import Literal

import pandas as pd
from tqdm import tqdm

from my_project.config import (
    MODEL_DIR, MAX_SEQ_LENGTH, LOAD_IN_4BIT, TEMPERATURE, MAX_NEW_TOKENS,
    CHECKPOINT_STEP, KNOWLEDGE_CSV,
)
from my_project.knowledge    import compute_knowledge_scores
from my_project.gnn          import run_gnn
from my_project.graphml_export import build_and_export_graphml


# ─────────────────────────────────────────────────────────────────────────────
# LLM query helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tf(text: str, cid: str):
    p = text.strip().lower()
    if re.fullmatch(r"true[.!?]*",  p): return 1
    if re.fullmatch(r"false[.!?]*", p): return 0
    return (-1, f"Bad response for {cid}: '{text}'")


def _build_opensource_querier(model_dir: str):
    """Returns a query_fn(prompt, triplet_id) -> int|tuple."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3")

    SYSTEM = (
        'Evaluate the following statement based on your knowledge and '
        'respond only with "true" or "false". This is for research purposes only.'
    )

    def query_fn(prompt: str, cid: str):
        try:
            msgs = [
                {"role": "system",  "content": SYSTEM},
                {"role": "user",    "content": prompt},
            ]
            inputs = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
            outs = model.generate(
                input_ids=inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=None if TEMPERATURE == 0 else TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            answer = tokenizer.decode(
                outs[0, inputs.shape[1]:], skip_special_tokens=True
            )
            return _parse_tf(answer, cid)
        except Exception as e:
            return (-1, f"Exception for {cid}: {repr(e)}")

    return query_fn


def _build_openai_querier(model_name: str = "gpt-3.5-turbo"):
    """Returns a query_fn(prompt, triplet_id) -> int|tuple. Requires OPENAI_API_KEY."""
    import openai
    client = openai.OpenAI()  # reads OPENAI_API_KEY from env

    SYSTEM = (
        'Evaluate the following statement based on your knowledge and '
        'respond only with "true" or "false". This is for research purposes only.'
    )

    def query_fn(prompt: str, cid: str):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=1,
                temperature=0,
            )
            answer = resp.choices[0].message.content or ""
            return _parse_tf(answer, cid)
        except Exception as e:
            return (-1, f"OpenAI exception for {cid}: {repr(e)}")

    return query_fn


def _run_llm_on_prompts(
    df_prompts: pd.DataFrame,
    query_fn,
    checkpoint_dir: str,
    checkpoint_prefix: str = "llm",
) -> pd.DataFrame:
    """
    Queries query_fn on every row of df_prompts (columns: triplet_id, triplet_prompt).
    Supports resume from checkpoint.
    Returns df_prompts with an added 'tf_value' column.
    """
    save_path  = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_results.pkl")
    error_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_errors.pkl")

    df = df_prompts.copy()
    df["tf_value"] = pd.NA

    results, errors = [], []
    if os.path.exists(save_path) and os.path.exists(error_path):
        with open(save_path, "rb") as f: results = pkl.load(f)
        with open(error_path, "rb") as f: errors  = pkl.load(f)
        done = {cid for cid, _ in results}
        for i, row in df.iterrows():
            if row["triplet_id"] in done:
                val = next(v for c, v in results if c == row["triplet_id"])
                df.at[i, "tf_value"] = val

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM query"):
        if not pd.isna(row["tf_value"]):
            continue
        cid = row["triplet_id"]
        res = query_fn(row["triplet_prompt"], cid)
        if isinstance(res, tuple) and res[0] == -1:
            results.append((cid, -1))
            errors.append({"id": cid, "error": res[1]})
            df.at[idx, "tf_value"] = -1
        else:
            results.append((cid, res))
            df.at[idx, "tf_value"] = res

        if (idx + 1) % CHECKPOINT_STEP == 0 or (idx + 1) == len(df):
            with open(save_path,  "wb") as f: pkl.dump(results, f)
            with open(error_path, "wb") as f: pkl.dump(errors,  f)

    df.to_csv(os.path.join(checkpoint_dir, f"{checkpoint_prefix}_results.csv"), index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_homophily(
    input_dir: str,
    output_dir: str,
    mode: Literal["full", "sparse"] = "sparse",
    llm_backend: Literal["opensource", "openai"] = "opensource",
    sparse_fraction: float = 0.10,
    openai_model: str = "gpt-3.5-turbo",
    random_seed: int = 42,
):
    """
    Part 1: compute entity knowledge scores and export graphml.

    Parameters
    ----------
    input_dir       : processed dataset folder (contains metadata.json, triplet_prompts.csv)
    output_dir      : where to write outputs (graphml, entity scores, etc.)
    mode            : 'full' (query all) or 'sparse' (10% anchor + GNN)
    llm_backend     : 'opensource' (unsloth) or 'openai'
    sparse_fraction : fraction of triplets to query when mode='sparse'
    openai_model    : OpenAI model name (ignored when llm_backend='opensource')
    random_seed     : RNG seed
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── load metadata & data ─────────────────────────────────────────────────
    with open(os.path.join(input_dir, "metadata.json")) as fp:
        meta = json.load(fp)

    kg_df       = pd.read_csv(meta["input_kg"])
    # normalise column names
    kg_df = kg_df.rename(columns={
        c: c for c in kg_df.columns  # identity; actual rename handled below
    })
    for src, dst in [("subject", "sub_label"), ("object", "obj_label"),
                     ("relation", "rel_label"), ("id", "triplet_id")]:
        if src in kg_df.columns and dst not in kg_df.columns:
            kg_df = kg_df.rename(columns={src: dst})

    prompts_df  = pd.read_csv(os.path.join(input_dir, meta["prompts"]))

    # ── build LLM query function ─────────────────────────────────────────────
    if llm_backend == "openai":
        query_fn = _build_openai_querier(openai_model)
    else:
        query_fn = _build_opensource_querier(MODEL_DIR)

    # ── select triplets to query ─────────────────────────────────────────────
    if mode == "full":
        query_prompts = prompts_df.copy()
    else:  # sparse
        n_query = max(1, int(len(prompts_df) * sparse_fraction))
        query_prompts = prompts_df.sample(n=n_query, random_state=random_seed)
        print(f"[homophily/sparse] querying {n_query}/{len(prompts_df)} triplets")

    # ── run LLM ──────────────────────────────────────────────────────────────
    queried_df = _run_llm_on_prompts(
        query_prompts, query_fn,
        checkpoint_dir=output_dir,
        checkpoint_prefix="homophily",
    )

    # ── compute entity knowledge scores from queried triplets ─────────────────
    merged = queried_df.merge(
        kg_df[["triplet_id", "sub_label", "obj_label"]].rename(
            columns={"sub_label": "Sub", "obj_label": "Obj"}
        ),
        on="triplet_id", how="left",
    )
    # filter out -1 (error) rows
    merged = merged[merged["tf_value"] >= 0]

    kg_scores = compute_knowledge_scores(merged, tf_col="tf_value")
    kg_scores = kg_scores.rename(columns={"entity": "entity_label"})

    obs_path = os.path.join(output_dir, KNOWLEDGE_CSV)
    kg_scores.to_csv(obs_path, index=False)
    print(f"[homophily] observed scores for {len(kg_scores)} entities → {obs_path}")

    # ── GNN for sparse mode ───────────────────────────────────────────────────
    # Write a temporary metadata.json pointing to output_dir so run_gnn works
    tmp_meta = dict(meta)
    tmp_meta_path = os.path.join(output_dir, "metadata.json")
    with open(tmp_meta_path, "w") as fp:
        json.dump(tmp_meta, fp, indent=2)

    run_gnn(output_dir)   # writes entity_kg_predictions.csv into output_dir

    # ── load GNN predictions ─────────────────────────────────────────────────
    pred_path = os.path.join(output_dir, "entity_kg_predictions.csv")
    pred_df   = pd.read_csv(pred_path)

    # ── build merged scores df for graphml (observed overrides predicted) ─────
    scores_for_export = pred_df[["entity_label", "predicted_kg_value", "kg_value"]].copy()

    # ── export graphml ────────────────────────────────────────────────────────
    graphml_path = os.path.join(output_dir, "knowledge_graph.graphml")
    build_and_export_graphml(
        kg_df=kg_df,
        entity_scores_df=scores_for_export,
        output_path=graphml_path,
        observed_col="kg_value",
        predicted_col="predicted_kg_value",
    )

    print(f"\n[homophily] Done. Outputs in {output_dir}")
    print(f"  Observed scores : {obs_path}")
    print(f"  GNN predictions : {pred_path}")
    print(f"  GraphML         : {graphml_path}")
    return graphml_path
