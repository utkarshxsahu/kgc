# A Graph Perspective to Probe Structural Patterns of Knowledge in Large Language Models

## Enviornment Setup
To setup the environment, dependencies can be installed using conda or pip:

---
## Data Setup

<li> Dataset files are available at:<br>
https://drive.google.com/drive/folders/1tj1rzJiXlbJtLMFpKWTMSuLmCtvHk5VL?usp=sharing

<li> Please download them and save them within the directory data/raw/

---
## Process
### Step 1: Initial Entity and Triplets selection + Prompt Building
Replace sample_data with the name of the dataset you want to run.<br>

'''
python scripts/run_dataset.py --input-dir data/raw/sample_data --output-base data/processed

'''
