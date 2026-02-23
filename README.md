# Knowledge Homophily in Large Language Models

This repository contains the code and data associated with our WSDM 2026 paper, [**"Knowledge Homophily in Large Language Models"**](https://arxiv.org/abs/2509.23773).

## Abstract
Large Language Models (LLMs) have been increasingly studied as neural knowledge bases for supporting knowledge-intensive applications such as question answering and fact checking. However, the structural organization of their knowledge remains unexplored. Inspired by cognitive neuroscience findings, such as semantic clustering and priming, where knowing one fact increases the likelihood of recalling related facts, we investigate an analogous knowledge homophily pattern in LLMs. To this end, we map LLM knowledge into a graph representation through knowledge checking at both the triplet and entity levels. After that, we analyze the knowledgeability relationship between an entity and its neighbors, discovering that LLMs tend to possess a similar level of knowledge about entities positioned closer in the graph. Motivated by this homophily principle, we propose a Graph Neural Network (GNN) regression model to estimate entity-level knowledgeability scores for triplets by leveraging their neighborhood scores. The predicted knowledgeability enables us to prioritize checking less well-known triplets, thereby maximizing knowledge coverage under the same labeling budget. This not only improves the efficiency of active labeling for fine-tuning to inject knowledge into LLMs but also enhances multi-hop path retrieval in reasoning-intensive question answering.

> **TL;DR** — LLMs tend to know similar amounts about topologically close entities in a knowledge graph. We measure this "knowledge homophily", use a GNN to predict which entities an LLM doesn't know, and then fine-tune the LLM on those unknown facts.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Step 0 — Preprocess](#step-0--preprocess-dataset-and-build-prompts)
  - [Part 1 — Homophily](#part-1--homophily-knowledge-checking--graphml-export)
  - [Part 2 — Fine-tuning](#part-2--knowledge-injection--fine-tuning)
- [Outputs Reference](#outputs-reference)
- [Design Decisions & Improvements](#design-decisions--improvements)
- [Datasets](#datasets)
- [Citation](#citation)

---

## Overview

The pipeline has two independent parts:

```
┌─────────────────────────────────────────────────────────────────┐
│  PART 1 — HOMOPHILY                                             │
│                                                                 │
│  KG triplets  ──►  LLM true/false query  ──►  entity scores    │
│                         (full or 10%)                           │
│                              │                                  │
│                         GNN training  ──►  predict all scores   │
│                              │                                  │
│                       knowledge_graph.graphml  (Gephi-ready)    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  PART 2 — KNOWLEDGE INJECTION                                   │
│                                                                 │
│  Budget (e.g. 4000 triplets)                                    │
│    ├── 20%  Anchor set  ──►  LLM query  ──►  train GNN          │
│    └── 80%  Ignorance-guided selection  (GNN predictions)       │
│                    │                                            │
│              Combined fine-tuning dataset                       │
│                    │                                            │
│            UnslothTrainer (LoRA / continual pre-train)          │
│                    │                                            │
│         Evaluate on 2% held-out test set                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
kgc/
├── src/my_project/
│   ├── config.py               # All hyperparameters and paths
│   ├── budget_selection.py     # Anchor selection + ignorance-guided selection
│   ├── homophily.py            # Part 1 orchestration
│   ├── finetune.py             # Part 2 orchestration
│   ├── graphml_export.py       # Export NetworkX graph → .graphml
│   ├── gnn.py                  # GraphSAGE training + prediction
│   ├── knowledge.py            # Aggregate triplet scores → entity scores
│   ├── query.py                # Low-level LLM querying with checkpointing
│   ├── selection.py            # Legacy entity/triplet sampling
│   ├── prompting.py            # Build natural-language prompts from triplets
│   └── io.py / preprocess.py   # CSV I/O and column normalisation
│
├── scripts/
│   ├── run_dataset.py          # Step 0: preprocess raw KG + build prompts
│   ├── run_homophily.py        # Part 1 CLI
│   ├── run_finetune.py         # Part 2 CLI
│   ├── run_gnn.py              # Standalone: train/predict GNN only
│   └── run_knowledge.py        # Standalone: compute entity scores only
│
└── data/
    ├── raw/                    # Original KG CSVs + relation templates
    └── processed/              # Per-dataset outputs (auto-created)
```

---

## Installation

```bash
conda create --name kgc \
    python=3.11 pytorch-cuda=12.1 pytorch cudatoolkit xformers \
    -c pytorch -c nvidia -c xformers -y
conda activate kgc

pip install unsloth networkx sentence-transformers torch-geometric

# Optional — only needed for the OpenAI LLM backend
pip install openai
```

---

## Configuration

All parameters are in `src/my_project/config.py`. The most commonly changed ones:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOTAL_TRIPLET_BUDGET` | `4000` | Total triplets for the fine-tuning pipeline |
| `ANCHOR_FRACTION` | `0.20` | Fraction of budget queried against the LLM as anchors |
| `MAX_TRIPLETS_PER_ENTITY_ANCHOR` | `40` | Per-entity cap in anchor set (prevents one hub entity consuming the whole anchor budget) |
| `MIN_TRIPLETS_PER_PARTIAL_SLICE` | `5` | Minimum slice size in the ignorance-guided selection |
| `FRACTION_TRIPLETS_FOR_PARTIAL` | `0.30` | Target fraction of an entity's degree for partial picks |
| `MODEL_DIR` | `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | Model used for LLM querying (Part 1 / anchor queries) |
| `FINETUNE_MODEL_DIR` | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | Model to fine-tune |
| `FINETUNE_EPOCHS` | `3` | Training epochs |
| `FINETUNE_LR` | `2e-4` | LoRA learning rate |

---

## Usage

### Step 0 — Preprocess Dataset and Build Prompts

Prepare a raw dataset folder with two files:
- `<dataset>.csv` — the knowledge graph with columns: `triplet_id`, `subject`, `relation`, `object`
- `<dataset>_templates.csv` — relation-to-prompt templates with columns: `rel_label`, `prompt_template`

```bash
python scripts/run_dataset.py \
    --input-dir  data/raw/trex \
    --output-base data/processed
```

**Outputs** in `data/processed/trex/`:

| File | Description |
|------|-------------|
| `triplet_prompts.csv` | `triplet_id`, `triplet_prompt` |
| `selected_entities.csv` | Entities included in prompt set |
| `metadata.json` | Paths index for downstream scripts |

---

### Part 1 — Homophily: Knowledge Checking + GraphML Export

Measures entity-level knowledgeability and exports a graph file ready for visualization.

#### Sparse mode *(default — recommended)*

Queries a fraction of triplets, trains a GNN, and predicts scores for all remaining entities.

```bash
python scripts/run_homophily.py \
    --input-dir  data/processed/trex \
    --output-dir data/processed/trex/homophily \
    --mode sparse \
    --sparse-fraction 0.10
```

#### Full mode

Queries every triplet against the LLM. More accurate but significantly more expensive.

```bash
python scripts/run_homophily.py \
    --input-dir  data/processed/trex \
    --output-dir data/processed/trex/homophily \
    --mode full
```

#### LLM backend options

| Flag | Backend | Notes |
|------|---------|-------|
| `--llm-backend opensource` *(default)* | Local unsloth model | Set `MODEL_DIR` in config |
| `--llm-backend openai --openai-model gpt-3.5-turbo` | OpenAI API | Requires `OPENAI_API_KEY` environment variable |

#### Visualizing the GraphML

Open `knowledge_graph.graphml` in **Gephi**:
1. Import graph file
2. Run **ForceAtlas2** layout (preserves topological proximity)
3. Color nodes by `kg_value` (red = high knowledge, blue = low knowledge)
4. Node attribute `kg_source` tells you whether the score is `observed` (directly queried) or `predicted` (GNN)

---

### Part 2 — Knowledge Injection + Fine-tuning

Selects the most ignorance-rich triplets within a fixed budget and fine-tunes the LLM on them.

```bash
python scripts/run_finetune.py \
    --input-dir  data/processed/trex \
    --output-dir data/processed/trex/finetune
```

**What happens inside:**

1. **Anchor selection** — Degree-stratified sampling fills 20% of budget, capped per entity
2. **LLM querying** — Anchor triplets are queried for ground-truth true/false labels
3. **GNN training** — GraphSAGE trained on anchor entity scores; predicts scores for all entities
4. **Ignorance-guided selection** — Entities ranked by `1 − knowledge_score`; budget greedily allocated (full or partial slices)
5. **Fine-tuning** — `UnslothTrainer` runs on the combined anchor + selected triplet set
6. **Evaluation** — Fine-tuned model is queried on a held-out 2% test set; results saved to `evaluation_summary.json`

#### Full CLI options

```
--input-dir       Processed dataset folder
--output-dir      Output folder for model + results
--llm-backend     opensource (default) | openai
--openai-model    OpenAI model name (default: gpt-3.5-turbo)
--test-fraction   Fraction of all triplets held out for evaluation (default: 0.02)
--seed            Random seed (default: 42)
```

---

## Outputs Reference

### Part 1 outputs (`--output-dir`)

| File | Description |
|------|-------------|
| `entity_kg_values.csv` | Observed knowledge scores for queried entities |
| `entity_kg_predictions.csv` | GNN-predicted scores for all entities |
| `knowledge_graph.graphml` | Full graph with `kg_value` and `kg_source` on every node |
| `homophily_results.csv` | Raw LLM true/false responses |
| `homophily_results.pkl` | Checkpoint (allows resuming a partial run) |

### Part 2 outputs (`--output-dir`)

| File | Description |
|------|-------------|
| `anchor_entity_scores.csv` | Observed knowledge scores for anchor entities |
| `entity_kg_predictions.csv` | GNN predictions for all entities |
| `combined_finetune_triplets.csv` | Final training set (anchor + ignorance-selected) |
| `selected_entities_80pct.csv` | Entities chosen during ignorance-guided step |
| `finetuned_model/` | Saved LoRA model + tokenizer |
| `test_set.csv` | Held-out evaluation triplets |
| `evaluation_results.csv` | Per-triplet fine-tuned model responses |
| `evaluation_summary.json` | `generalization_gain` — % of test triplets now known |

---

## Design Decisions & Improvements

Two deliberate improvements were made over the original pseudocode:

**1. Stratified anchor sampling**

The original approach selects entities at random until the anchor budget is filled. This tends to over-represent high-degree hub entities, giving the GNN a biased training signal. The new approach divides entities into degree bins and samples round-robin across all bins, ensuring the GNN sees a diverse range of neighborhood structures.

**2. Safe partial-slice fallback**

In the ignorance-guided selection, when not enough *new* triplets exist for a partial slice, the pseudocode falls back to `all_triplets_for_entity` — which can re-include anchor or previously-excluded triplets. The implementation here keeps the fallback pool filtered against the excluded set, so anchor triplets can never be accidentally queued a second time.

---

## Datasets

The paper evaluates on five knowledge graphs:

| Dataset | Domain | Entities | Triplets |
|---------|--------|----------|----------|
| [T-Rex](https://hadyelsahar.github.io/t-rex/) | General (Wikipedia) | 46,891 | 193,781 |
| [WD50K](https://arxiv.org/abs/2009.10847) | General (Wikidata) | 5,140 | 34,208 |
| [CoDEx-S](https://arxiv.org/abs/2009.07810) | General (Wikidata/Wikipedia) | 2,034 | 36,543 |
| [PharmKG8K](https://academic.oup.com/bib/article/22/4/bbaa344/6042240) | Biomedical | 6,877 | 98,537 |
| [MVPKG](https://dl.acm.org/doi/10.1145/3589334.3645464) | Political / Legislative | 9,055 | 255,697 |

Raw files go in `data/raw/<dataset>/`. Each dataset folder needs a KG CSV and a relation templates CSV (see Step 0).

---

## Citation

```bibtex
@inproceedings{sahu2026knowledge,
  title     = {Knowledge Homophily in Large Language Models},
  author    = {Sahu, Utkarsh and Qi, Zhisheng and Halappanavar, Mahantesh and
               Lipka, Nedim and Rossi, Ryan and Dernoncourt, Franck and
               Zhang, Yu and Ma, Yao and Wang, Yu},
  booktitle = {Proceedings of the Nineteenth ACM International Conference on
               Web Search and Data Mining (WSDM '26)},
  year      = {2026},
  doi       = {10.1145/3773966.3779394}
}
```
