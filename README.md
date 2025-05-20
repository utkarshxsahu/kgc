# A Graph Perspective to Probe Structural Patterns of Knowledge in Large Language Models

## Enviornment Setup
To setup the environment, dependencies can be installed using conda or pip:

---
## Data Setup

<li> Dataset files are available at:<br>
https://drive.google.com/drive/folders/1tj1rzJiXlbJtLMFpKWTMSuLmCtvHk5VL?usp=sharing/


<li> Please download them and save them within the directory data/raw/

---
## Process
### Step 1: Initial entity and triplets selection + prompt building
Replace sample_data with the name of the dataset folder you want to run. For example, to run "trex" dataset, replace it with sample_data. It should look like data/raw/trex<br>

```
python scripts/run_dataset.py --input-dir data/raw/sample_data --output-base data/processed

```
### Step 2: Query triplet prompts to LLM for True/False evaluation

```
python scripts/run_query.py --input-dir data/processed/sample_data
```
### Step 3: Compute entity knowledge scores
```
python scripts/run_knowledge.py --input-dir data/processed/sample_data

```

### Predicting entity knowledge values across dataset using GNN
```
python scripts/run_gnn.py --input-dir data/processed/sample_data
```
