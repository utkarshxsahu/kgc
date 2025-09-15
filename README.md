# Knowledge Homophily in Large Language Models

Large Language Models (LLMs) have been increasingly studied as neural knowledge bases for supporting knowledge-intensive applications such as question answering and fact checking. However, their structural organization remains largely unexplored. Inspired by findings in cognitive neuroscience, such as semantic clustering and priming, where knowing one fact increases the likelihood of recalling related facts, we investigate an analogous knowledge homophily pattern in LLMs. To this end, we map LLM knowledge into a graph representation through knowledge checking at both the triplet and entity levels. After that, we analyze the knowledgeability relationship between an entity and its neighbors, discovering that LLMs tend to possess a similar level of knowledge about entities positioned closer in the graph. Motivated by the principle of homophily, we propose a Graph Neural Network (GNN) regression model to estimate entity-level knowledgeability scores for triplets by leveraging the scores of their neighbors. The predicted knowledgeability enables us to prioritize checking less well-known triplets, thereby maximizing knowledge coverage under the same labeling budget. This not only improves the efficiency of active labeling for knowledge injection but also enhances multi-hop path retrieval in reasoning-intensive question answering.

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
Open the src/my_project/query.py and in the input field enter the triplets csv file processed from step 1. Save it and then run:
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
