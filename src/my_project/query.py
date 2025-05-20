# src/my_project/query.py
import os, time, re, pickle, json
import pandas as pd
from tqdm import tqdm

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from my_project.config import (
    MODEL_DIR, MAX_SEQ_LENGTH, LOAD_IN_4BIT, TEMPERATURE, MAX_NEW_TOKENS,
    CHECKPOINT_STEP, RESULTS_PKL, ERRORS_PKL, OUTPUT_CSV
)

def load_metadata(input_dir):
    meta_path = os.path.join(input_dir, "metadata.json")
    with open(meta_path) as fp:
        return json.load(fp)

def init_model(model_dir=MODEL_DIR):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_dir,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype          = None,
        load_in_4bit   = LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    return model, tokenizer

def normalize_response(resp, tid):
    txt = resp.strip().lower()
    if re.fullmatch(r"true[\.\!\?]*", txt):  return 1
    if re.fullmatch(r"false[\.\!\?]*",txt):  return 0
    return (-1, f"Bad response for {tid}: “{resp}”")

def query_one(model, tokenizer, prompt, tid):
    try:
        system = (
            'Evaluate the given statement based on your knowledge '
            'and respond with only "true" or "false". This is for research purposes only.'
        )
        msgs = [
            {"role":"system", "content": system},
            {"role":"user",   "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        outs = model.generate(
            input_ids      = inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            temperature    = None if TEMPERATURE==0 else TEMPERATURE,
            do_sample      = False if TEMPERATURE==0 else True,
            pad_token_id   = tokenizer.eos_token_id,
            use_cache      = True,
        )
        # strip off prompt tokens
        ans = tokenizer.decode(outs[0, inputs.shape[1]:], skip_special_tokens=True)
        return normalize_response(ans, tid)
    except Exception as e:
        return (-1, f"Exception for {tid}: {repr(e)}")

def run_queries(input_dir, model_dir=None):
    meta      = load_metadata(input_dir)
    prompts_f = os.path.join(input_dir, meta["prompts"])
    df        = pd.read_csv(prompts_f)
    if "triplet_id" not in df.columns or "triplet_prompt" not in df.columns:
        raise ValueError("Need columns 'triplet_id' and 'triplet_prompt'")
    df["tf_value"] = pd.NA

    # prepare checkpoint files
    res_path = os.path.join(input_dir, RESULTS_PKL)
    err_path = os.path.join(input_dir, ERRORS_PKL)
    results, errors = [], []
    done_ids = set()
    if os.path.exists(res_path) and os.path.exists(err_path):
        with open(res_path,"rb") as f: results = pickle.load(f)
        with open(err_path,"rb") as f: errors  = pickle.load(f)
        done_ids = {tid for tid,_ in results}
        for i,v in results:
            df.loc[df.triplet_id==i, "tf_value"] = v

    model, tokenizer = init_model(model_dir or MODEL_DIR)

    start = time.time()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="LLM Inference"):
        tid, prompt = row["triplet_id"], row["triplet_prompt"]
        if tid in done_ids: continue

        resp = query_one(model, tokenizer, prompt, tid)
        if isinstance(resp, tuple) and resp[0]==-1:
            results.append((tid, -1))
            errors.append({"triplet_id": tid, "prompt": prompt, "error": resp[1]})
            df.at[idx, "tf_value"] = -1
        else:
            results.append((tid, resp))
            df.at[idx, "tf_value"] = resp

        # checkpoint
        if (len(results) % CHECKPOINT_STEP)==0 or idx+1==len(df):
            with open(res_path,"wb") as f: pickle.dump(results, f)
            with open(err_path,"wb") as f: pickle.dump(errors, f)

    duration = time.time() - start
    print(f"Done in {duration:.1f}s — successes: {sum(1 for _,v in results if v>=0)}, errors: {len(errors)}")

    # final CSV
    out_csv = os.path.join(input_dir, OUTPUT_CSV)
    df.to_csv(out_csv, index=False)
    print(f"Results saved → {out_csv}")
