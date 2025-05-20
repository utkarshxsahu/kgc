# A Graph Perspective to Probe Structural Patterns of Knowledge in Large Language Models

## Enviornment Setup
To setup the environment, we need to install unsloth to use LLMs to query our data.
```
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env
pip install unsloth
```
## Data Setup

<li> Dataset files are available in the directory data/raw

---
## Process
### Step 1: Initial entity and triplets selection + prompt building
Replace sample_data with the name of the dataset folder you want to run. For example, to run "trex" dataset, replace it with sample_data. It should look like data/raw/trex<br>

```
python scripts/run_dataset.py --input-dir data/raw/sample_data --output-base data/processed

```
### Step 2: Query triplet prompts to LLM for True/False evaluation
Open the src/my_project/query.py and in the input field enter the triplets csv file from step 1. Then run:
```
python src/my_project/run_query.py 
```
### Step 3: Compute entity knowledge scores
```
python scripts/run_knowledge.py --input-dir data/processed/sample_data

```

### Step 4: Predicting entity knowledge values across dataset using GNN
```
python scripts/run_gnn.py --input-dir data/processed/sample_data
```
