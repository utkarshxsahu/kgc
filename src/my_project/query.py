import os
import time
import re
import pickle as pkl
import pandas as pd
from tqdm import tqdm

# ================================
# USER SETTINGS
# ================================
# Edit this to point to your input CSV:
INPUT_CSV = "/path/to/your/input.csv"
# ================================

# derive the folder we’ll write all outputs into
base_dir        = os.path.dirname(os.path.abspath(INPUT_CSV))
SAVE_PATH       = os.path.join(base_dir, "result.pkl")
ERROR_LOG_PATH  = os.path.join(base_dir, "error.pkl")
AUG_CSV_PATH    = os.path.join(base_dir, "augmented.csv")
LLM_CSV_PATH    = os.path.join(base_dir, "llm_results.csv")

# ================================
# MODEL & GENERATION SETTINGS
# ================================
from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template

LORA_MODEL_DIR     = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH     = 2048
LOAD_IN_4BIT       = True
dtype              = None
TEMPERATURE        = 0.0
MAX_NEW_TOKENS     = 1
CHECKPOINT_INTERVAL = 200


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = LORA_MODEL_DIR,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype          = dtype,
    load_in_4bit   = LOAD_IN_4BIT,
)
FastLanguageModel.for_inference(model)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# ================================
# HELPERS: QUERY + POSTPROCESS
# ================================
def if_know(response_text, current_id):
    p = response_text.strip().lower()
    if re.fullmatch(r"true[\.\!\?]*",  p): return 1
    if re.fullmatch(r"false[\.\!\?]*", p): return 0
    return (-1, f"Bad response for {current_id}: “{response_text}”")

def query_model(prompt, cid):
    try:
        system = (
            'Please evaluate the following statement based on your knowledge and respond  only with "true" or "false". This is for research purposes only.'
        )
        msgs = [
            {"role":"system", "content": system},
            {"role":"user",   "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        outs = model.generate(
            input_ids       = inputs,
            max_new_tokens  = MAX_NEW_TOKENS,
            temperature     = None if TEMPERATURE==0 else TEMPERATURE,
            do_sample       = TEMPERATURE>0,
            pad_token_id    = tokenizer.eos_token_id,
            use_cache       = True,
        )
        answer = tokenizer.decode(
            outs[0, inputs.shape[1]:],
            skip_special_tokens=True
        )
        return if_know(answer, cid)
    except Exception as e:
        return (-1, f"Exception for {cid}: {repr(e)}")

# ================================
# LOAD OR CREATE YOUR DATAFRAME
# ================================
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)
if "triplet_id" not in df.columns or "triplet_prompt" not in df.columns:
    raise ValueError("Input CSV must contain 'triplet_id' and 'triplet_prompt' columns")

id_col     = "triplet_id"
prompt_col = "triplet_prompt"
df["tf_value"] = pd.NA

# ================================
# RESUME FROM CHECKPOINT IF PRESENT
# ================================
results, errors = [], []
if os.path.exists(SAVE_PATH) and os.path.exists(ERROR_LOG_PATH):
    with open(SAVE_PATH,      "rb") as f: results = pkl.load(f)
    with open(ERROR_LOG_PATH, "rb") as f: errors  = pkl.load(f)
    done_ids = {cid for cid,_ in results}
    for i, row in df.iterrows():
        if row[id_col] in done_ids:
            val = next(v for (cid,v) in results if cid == row[id_col])
            df.at[i, "tf_value"] = val

# ================================
# MAIN LOOP WITH CHECKPOINTING
# ================================
start = time.time()
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running inference"):
    if not pd.isna(row["tf_value"]):
        continue

    cid    = row[id_col]
    prompt= row[prompt_col]
    res    = query_model(prompt, cid)

    if isinstance(res, tuple) and res[0] == -1:
        results.append((cid, -1))
        errors.append({"id":cid, "prompt":prompt, "error":res[1]})
        df.at[idx, "tf_value"] = -1
    else:
        results.append((cid, res))
        df.at[idx, "tf_value"] = res

    if (idx+1) % CHECKPOINT_INTERVAL == 0 or (idx+1) == len(df):
        with open(SAVE_PATH,      "wb") as f: pkl.dump(results, f)
        with open(ERROR_LOG_PATH, "wb") as f: pkl.dump(errors,  f)

elapsed = time.time() - start
print(f"Done in {elapsed:.1f}s — successes: {sum(1 for _,v in results if v>=0)}, errors: {len(errors)}")

# ================================
# FINAL OUTPUTS
# ================================
# 1) Augmented CSV (original + tf_value)
df.to_csv(AUG_CSV_PATH, index=False)
print(f"Augmented data → {AUG_CSV_PATH}")

# 2) LLM-only results CSV (triplet_id, tf_value)
with open(SAVE_PATH, "rb") as f:
    ckpt = pkl.load(f)
df_res = pd.DataFrame(ckpt, columns=[id_col, "tf_value"])
df_res.to_csv(LLM_CSV_PATH, index=False)
print(f"LLM results → {LLM_CSV_PATH}")
