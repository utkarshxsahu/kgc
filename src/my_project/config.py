# ── Step 1: column name aliases ───────────────────────────────────────────────
REQUIRED_HEADERS = {
    'triplet_id': ['triplet_id', 'id', 'tid', 'triple_id'],
    'sub_label' : ['subject', 'sub_label', 'subject_label','sub'],
    'rel_label' : ['relation', 'rel_label', 'predicate', 'rel'],
    'obj_label' : ['object', 'obj_label', 'object_label', 'obj'],
}

# ── Budget (shared across anchor selection & ignorance-guided selection) ───────
# Total triplet budget for the full fine-tuning pipeline.
# 20 % is spent on LLM-queried anchor triplets; 80 % is filled by GNN-guided
# ignorance-based selection.
TOTAL_TRIPLET_BUDGET              = 4000
ANCHOR_FRACTION                   = 0.20          # 20 % → anchor / LLM query

# Hard cap on how many triplets one entity may contribute to the anchor set.
# Prevents a single high-degree entity from consuming the whole anchor budget.
MAX_TRIPLETS_PER_ENTITY_ANCHOR    = 40

# Legacy aliases (kept so old code still works)
T_QUERY      = int(TOTAL_TRIPLET_BUDGET * ANCHOR_FRACTION)
Q4_CAP       = MAX_TRIPLETS_PER_ENTITY_ANCHOR
MAX_FULL_TRY = 2
NUM_BINS     = 4

# ── Ignorance-guided selection (pseudocode algorithm) ─────────────────────────
MIN_TRIPLETS_PER_PARTIAL_SLICE    = 5
FRACTION_TRIPLETS_FOR_PARTIAL     = 0.30   # target fraction of entity degree

# ── Step 2: LLM querying ──────────────────────────────────────────────────────
MODEL_DIR        = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH   = 2048
LOAD_IN_4BIT     = True
TEMPERATURE      = 0.0
MAX_NEW_TOKENS   = 1
CHECKPOINT_STEP  = 200
RESULTS_PKL      = "results.pkl"
ERRORS_PKL       = "errors.pkl"
OUTPUT_CSV       = "llm_results.csv"

# ── Step 3: knowledge scoring ─────────────────────────────────────────────────
KNOWLEDGE_CSV = "entity_kg_values.csv"

# ── Step 5: fine-tuning ───────────────────────────────────────────────────────
FINETUNE_MODEL_DIR     = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
FINETUNE_MAX_SEQ_LEN   = 2048
FINETUNE_LOAD_IN_4BIT  = True
FINETUNE_LORA_R        = 16
FINETUNE_LORA_ALPHA    = 16
FINETUNE_LORA_DROPOUT  = 0.0
FINETUNE_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
FINETUNE_EPOCHS        = 3
FINETUNE_BATCH_SIZE    = 4
FINETUNE_GRAD_ACCUM    = 4
FINETUNE_LR            = 2e-4
FINETUNE_WARMUP_RATIO  = 0.03
FINETUNE_WEIGHT_DECAY  = 0.01
FINETUNE_OUTPUT_DIR    = "finetuned_model"