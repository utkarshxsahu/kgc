# Step 1: Defaults
REQUIRED_HEADERS = {
    'triplet_id': ['triplet_id', 'id', 'tid'],
    'sub_label' : ['subject', 'sub_label', 'subject_label'],
    'rel_label' : ['relation', 'rel_label', 'predicate'],
    'obj_label' : ['object', 'obj_label', 'object_label'],
}
T_QUERY      = 2000
Q4_CAP       = 40
MAX_FULL_TRY = 2
NUM_BINS     = 4

# ── Step 2 defaults ────────────────────────────────────────────────────────────
MODEL_DIR        = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH   = 2048
LOAD_IN_4BIT     = True
TEMPERATURE      = 0.0
MAX_NEW_TOKENS   = 1
CHECKPOINT_STEP  = 200
RESULTS_PKL      = "results.pkl"
ERRORS_PKL       = "errors.pkl"
OUTPUT_CSV       = "llm_results.csv"

# ── Step 3 defaults ────────────────────────────────────────────────────────────
KNOWLEDGE_CSV = "entity_kg_values.csv"