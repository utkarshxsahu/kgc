"""
finetune.py
-----------
Part 2: Fine-tune the LLM on anchor (20%) + ignorance-selected (80%) triplets.

Pipeline
--------
  1. Load anchor triplet prompts (already LLM-queried, saved in output_dir)
  2. Load GNN predictions → build entity_scores_df
  3. Run ignorance-guided selection to fill remaining 80% budget
  4. Combine anchor + selected → fine-tuning dataset
  5. Fine-tune with unsloth / UnslothTrainer
  6. Evaluate on held-out test set (2% of all triplets)
  7. Save model + evaluation results

Usage (library)
---------------
    from my_project.finetune import run_finetune
    run_finetune(input_dir, output_dir)

Usage (CLI via scripts/run_finetune.py)
"""

from __future__ import annotations

import os
import json
import random

import pandas as pd

from my_project.config import (
    TOTAL_TRIPLET_BUDGET, ANCHOR_FRACTION, MAX_TRIPLETS_PER_ENTITY_ANCHOR,
    MIN_TRIPLETS_PER_PARTIAL_SLICE, FRACTION_TRIPLETS_FOR_PARTIAL,
    FINETUNE_MODEL_DIR, FINETUNE_MAX_SEQ_LEN, FINETUNE_LOAD_IN_4BIT,
    FINETUNE_LORA_R, FINETUNE_LORA_ALPHA, FINETUNE_LORA_DROPOUT,
    FINETUNE_TARGET_MODULES, FINETUNE_EPOCHS, FINETUNE_BATCH_SIZE,
    FINETUNE_GRAD_ACCUM, FINETUNE_LR, FINETUNE_WARMUP_RATIO,
    FINETUNE_WEIGHT_DECAY, FINETUNE_OUTPUT_DIR,
)
from my_project.budget_selection import select_anchor_triplets, select_ignorance_guided


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_test_set(
    all_triplets_df: pd.DataFrame,
    finetune_triplet_ids: set[str],
    test_fraction: float = 0.02,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Sample 2% of all triplets that:
      - are not in the fine-tuning set
      - don't share an entity with any fine-tuning triplet
    """
    finetune_df = all_triplets_df[
        all_triplets_df["triplet_id"].astype(str).isin(finetune_triplet_ids)
    ]
    ft_entities = set(finetune_df["sub_label"].tolist()) | set(finetune_df["obj_label"].tolist())

    candidate = all_triplets_df[
        ~all_triplets_df["triplet_id"].astype(str).isin(finetune_triplet_ids)
        & ~all_triplets_df["sub_label"].isin(ft_entities)
        & ~all_triplets_df["obj_label"].isin(ft_entities)
    ]

    n = max(1, int(len(all_triplets_df) * test_fraction))
    if len(candidate) <= n:
        return candidate.reset_index(drop=True)
    return candidate.sample(n=n, random_state=random_seed).reset_index(drop=True)


def _query_test_set(test_df: pd.DataFrame, query_fn, output_dir: str) -> pd.DataFrame:
    """Query the test triplets against the base LLM to get ground-truth labels."""
    import re, pickle as pkl, time
    from tqdm import tqdm
    from my_project.config import CHECKPOINT_STEP

    save_path  = os.path.join(output_dir, "test_results.pkl")
    error_path = os.path.join(output_dir, "test_errors.pkl")

    df = test_df.copy()
    df["tf_value"] = pd.NA

    results, errors = [], []
    if os.path.exists(save_path):
        with open(save_path, "rb") as f: results = pkl.load(f)
        done = {c for c, _ in results}
        for i, row in df.iterrows():
            if row["triplet_id"] in done:
                val = next(v for c, v in results if c == row["triplet_id"])
                df.at[i, "tf_value"] = val

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Test-set eval (base)"):
        if not pd.isna(row["tf_value"]):
            continue
        cid = row["triplet_id"]
        res = query_fn(row["triplet_prompt"], cid)
        if isinstance(res, tuple) and res[0] == -1:
            results.append((cid, -1)); errors.append({"id": cid})
            df.at[idx, "tf_value"] = -1
        else:
            results.append((cid, res)); df.at[idx, "tf_value"] = res
        if (idx + 1) % CHECKPOINT_STEP == 0:
            with open(save_path, "wb") as f: pkl.dump(results, f)

    with open(save_path, "wb") as f: pkl.dump(results, f)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_finetune(
    input_dir: str,
    output_dir: str,
    llm_backend: str = "opensource",
    openai_model: str = "gpt-3.5-turbo",
    test_fraction: float = 0.02,
    random_seed: int = 42,
):
    """
    Full fine-tuning pipeline.

    Parameters
    ----------
    input_dir    : processed dataset folder (metadata.json, triplet_prompts.csv, etc.)
    output_dir   : where to write fine-tuned model and evaluation results
    llm_backend  : 'opensource' or 'openai'
    test_fraction: fraction of all triplets held out for evaluation
    random_seed  : RNG seed
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── load raw data ─────────────────────────────────────────────────────────
    with open(os.path.join(input_dir, "metadata.json")) as fp:
        meta = json.load(fp)

    kg_df      = pd.read_csv(meta["input_kg"])
    for src, dst in [("subject","sub_label"),("object","obj_label"),
                     ("relation","rel_label"),("id","triplet_id")]:
        if src in kg_df.columns and dst not in kg_df.columns:
            kg_df = kg_df.rename(columns={src: dst})

    prompts_df = pd.read_csv(os.path.join(input_dir, meta["prompts"]))

    # ── STEP 1: anchor selection ──────────────────────────────────────────────
    anchor_ids, anchor_entities = select_anchor_triplets(
        kg_df=kg_df,
        total_budget=TOTAL_TRIPLET_BUDGET,
        anchor_fraction=ANCHOR_FRACTION,
        max_per_entity=MAX_TRIPLETS_PER_ENTITY_ANCHOR,
        random_seed=random_seed,
    )
    anchor_prompts = prompts_df[
        prompts_df["triplet_id"].astype(str).isin(set(str(i) for i in anchor_ids))
    ].reset_index(drop=True)

    # ── STEP 2: query anchor triplets against base LLM ────────────────────────
    print("\n── Querying anchor triplets against base LLM ──")
    if llm_backend == "openai":
        from my_project.homophily import _build_openai_querier
        query_fn = _build_openai_querier(openai_model)
    else:
        from my_project.homophily import _build_opensource_querier
        query_fn = _build_opensource_querier(FINETUNE_MODEL_DIR)

    from my_project.homophily import _run_llm_on_prompts
    anchor_queried = _run_llm_on_prompts(
        anchor_prompts, query_fn,
        checkpoint_dir=output_dir,
        checkpoint_prefix="anchor",
    )

    # ── STEP 3: compute anchor entity knowledge scores ────────────────────────
    from my_project.knowledge import compute_knowledge_scores
    anchor_merged = anchor_queried[anchor_queried["tf_value"] >= 0].merge(
        kg_df[["triplet_id","sub_label","obj_label"]].rename(
            columns={"sub_label":"Sub","obj_label":"Obj"}
        ),
        on="triplet_id", how="left",
    )
    anchor_scores = compute_knowledge_scores(anchor_merged, tf_col="tf_value")
    anchor_scores.to_csv(os.path.join(output_dir, "anchor_entity_scores.csv"), index=False)

    # ── STEP 4: train GNN on anchor entities → predict all entity scores ──────
    print("\n── Training GNN on anchor entities ──")
    # write anchor scores as entity_kg_values.csv so run_gnn can find them
    anchor_scores.to_csv(os.path.join(output_dir, "entity_kg_values.csv"), index=False)
    with open(os.path.join(output_dir, "metadata.json"), "w") as fp:
        json.dump(meta, fp, indent=2)

    from my_project.gnn import run_gnn
    run_gnn(output_dir)

    pred_df = pd.read_csv(os.path.join(output_dir, "entity_kg_predictions.csv"))
    # build entity_scores_df for ignorance selection
    entity_scores_df = pred_df.rename(columns={
        "entity_label":        "entity_id",
        "predicted_kg_value":  "predicted_knowledge_score",
        "kg_value":            "observed_knowledge_score",
    })[["entity_id","predicted_knowledge_score","observed_knowledge_score"]]

    # ── STEP 5: build triplet graph df (needs entity_id column names) ─────────
    triplet_graph_df = kg_df[["triplet_id","sub_label","obj_label"]].rename(
        columns={"sub_label":"subject_entity_id","obj_label":"object_entity_id"}
    )

    # ── STEP 6: ignorance-guided selection for remaining 80% budget ───────────
    print("\n── Ignorance-guided 80% selection ──")
    ignorance_budget = TOTAL_TRIPLET_BUDGET - len(anchor_ids)

    sel_triplets_df, sel_entities_df = select_ignorance_guided(
        entity_scores_df=entity_scores_df,
        triplet_graph_df=triplet_graph_df,
        triplet_prompt_df=prompts_df,
        excluded_triplet_ids=set(str(i) for i in anchor_ids),
        total_budget=ignorance_budget,
        min_triplets_per_partial_slice=MIN_TRIPLETS_PER_PARTIAL_SLICE,
        fraction_triplets_for_partial=FRACTION_TRIPLETS_FOR_PARTIAL,
        random_seed=random_seed,
    )

    # ── STEP 7: combine anchor + selected for fine-tuning ────────────────────
    combined = pd.concat(
        [anchor_prompts[["triplet_id","triplet_prompt"]],
         sel_triplets_df[["triplet_id","triplet_prompt"]]],
        ignore_index=True,
    ).drop_duplicates(subset=["triplet_id"]).reset_index(drop=True)

    combined_path = os.path.join(output_dir, "combined_finetune_triplets.csv")
    combined.to_csv(combined_path, index=False)
    sel_entities_df.to_csv(os.path.join(output_dir, "selected_entities_80pct.csv"), index=False)
    print(f"[finetune] Combined fine-tuning set: {len(combined)} triplets → {combined_path}")

    # ── STEP 8: fine-tune ─────────────────────────────────────────────────────
    print("\n── Fine-tuning with unsloth ──")
    _run_unsloth_finetune(
        combined_path=combined_path,
        output_dir=os.path.join(output_dir, FINETUNE_OUTPUT_DIR),
    )

    # ── STEP 9: held-out test evaluation ─────────────────────────────────────
    print("\n── Evaluating on held-out test set ──")
    test_kg = _build_test_set(
        all_triplets_df=kg_df,
        finetune_triplet_ids=set(combined["triplet_id"].astype(str).tolist()),
        test_fraction=test_fraction,
        random_seed=random_seed,
    )
    # join with prompts
    test_df = test_kg.merge(
        prompts_df[["triplet_id","triplet_prompt"]], on="triplet_id", how="inner"
    )
    test_df.to_csv(os.path.join(output_dir, "test_set.csv"), index=False)
    print(f"[finetune] Test set: {len(test_df)} triplets")

    # evaluate fine-tuned model on test set
    _evaluate_finetuned(
        test_df=test_df,
        model_dir=os.path.join(output_dir, FINETUNE_OUTPUT_DIR),
        output_dir=output_dir,
    )

    print(f"\n[finetune] All outputs saved to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Unsloth fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def _run_unsloth_finetune(combined_path: str, output_dir: str):
    """Runs UnslothTrainer on the combined fine-tuning CSV."""
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth import UnslothTrainer, UnslothTrainingArguments
    from datasets import Dataset

    os.makedirs(output_dir, exist_ok=True)

    # load model + apply LoRA
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=FINETUNE_MODEL_DIR,
        max_seq_length=FINETUNE_MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=FINETUNE_LOAD_IN_4BIT,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=FINETUNE_LORA_R,
        target_modules=FINETUNE_TARGET_MODULES + ["embed_tokens", "lm_head"],
        lora_alpha=FINETUNE_LORA_ALPHA,
        lora_dropout=FINETUNE_LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    # build dataset
    df = pd.read_csv(combined_path)
    dataset = Dataset.from_pandas(df[["triplet_prompt"]], preserve_index=False)

    EOS = tokenizer.eos_token

    def fmt(examples):
        return {"text": [p + EOS for p in examples["triplet_prompt"]]}

    dataset = dataset.map(fmt, batched=True, remove_columns=dataset.column_names)

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=FINETUNE_MAX_SEQ_LEN,
        dataset_num_proc=4,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=FINETUNE_BATCH_SIZE,
            gradient_accumulation_steps=FINETUNE_GRAD_ACCUM,
            warmup_ratio=FINETUNE_WARMUP_RATIO,
            num_train_epochs=FINETUNE_EPOCHS,
            learning_rate=FINETUNE_LR,
            embedding_learning_rate=FINETUNE_LR / 10,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=FINETUNE_WEIGHT_DECAY,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=output_dir,
            report_to="none",
        ),
    )

    stats = trainer.train()
    print(f"[finetune] Training stats: {stats}")

    # save model + tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[finetune] Model saved → {output_dir}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Post fine-tune evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_finetuned(
    test_df: pd.DataFrame,
    model_dir: str,
    output_dir: str,
):
    """
    Load fine-tuned model, query it on the test set, compute:
      - generalization gain  (% of test triplets the fine-tuned model now knows)
      - delta vs. base LLM  (improvement over base, if base results exist)
    """
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    import re, pickle as pkl
    from tqdm import tqdm
    from my_project.config import CHECKPOINT_STEP, MAX_NEW_TOKENS, TEMPERATURE

    # load fine-tuned model
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=FINETUNE_MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=FINETUNE_LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(ft_model)
    ft_tokenizer = get_chat_template(ft_tokenizer, chat_template="llama-3")

    SYSTEM = (
        'Evaluate the following statement based on your knowledge and '
        'respond only with "true" or "false". This is for research purposes only.'
    )

    def query_ft(prompt: str, cid: str):
        try:
            msgs = [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": prompt},
            ]
            inputs = ft_tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt"
            ).to("cuda")
            outs = ft_model.generate(
                input_ids=inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=None if TEMPERATURE == 0 else TEMPERATURE,
                do_sample=TEMPERATURE > 0,
                pad_token_id=ft_tokenizer.eos_token_id,
                use_cache=True,
            )
            answer = ft_tokenizer.decode(
                outs[0, inputs.shape[1]:], skip_special_tokens=True
            )
            p = answer.strip().lower()
            if re.fullmatch(r"true[.!?]*",  p): return 1
            if re.fullmatch(r"false[.!?]*", p): return 0
            return -1
        except Exception:
            return -1

    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Eval fine-tuned"):
        val = query_ft(row["triplet_prompt"], row["triplet_id"])
        results.append({"triplet_id": row["triplet_id"], "ft_tf_value": val})

    eval_df = test_df.merge(pd.DataFrame(results), on="triplet_id", how="left")
    eval_path = os.path.join(output_dir, "evaluation_results.csv")
    eval_df.to_csv(eval_path, index=False)

    valid = eval_df[eval_df["ft_tf_value"] >= 0]
    generalization_gain = valid["ft_tf_value"].mean() * 100 if len(valid) else 0.0

    summary = {
        "test_triplets":       len(test_df),
        "valid_responses":     len(valid),
        "generalization_gain": round(generalization_gain, 2),
    }
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)

    print(f"\n[eval] Generalization gain (% known by fine-tuned model): {generalization_gain:.1f}%")
    print(f"[eval] Results → {eval_path}")
    print(f"[eval] Summary → {summary_path}")
    return summary
