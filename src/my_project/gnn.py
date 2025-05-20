# src/my_project/gnn.py

import os
import json
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# ── HYPERPARAMETERS ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
HIDDEN_DIM           = 64
LR                   = 0.01
EPOCHS               = 100
TEST_SIZE            = 0.30     # fraction for (val+test)
VAL_FRACTION_OF_TEST = 0.50     # fraction of that to use as validation
SPLIT_RANDOM_STATE   = 42
DEVICE               = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Filenames (inside input_dir)
KG_VALUES_CSV        = "entity_kg_values.csv"
OUTPUT_PRED_CSV      = "entity_kg_predictions.csv"

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_ch, hidden_ch):
        super().__init__()
        self.conv1 = SAGEConv(in_ch, hidden_ch)
        self.conv2 = SAGEConv(hidden_ch, hidden_ch)
        self.lin   = torch.nn.Linear(hidden_ch, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x)

def run_gnn(input_dir):
    # 1. load metadata.json
    with open(os.path.join(input_dir, "metadata.json")) as fp:
        meta = json.load(fp)

    # 2. load raw KG and knowledge‐value CSV
    kg_df      = pd.read_csv(meta["input_kg"]) 
    values_df  = pd.read_csv(os.path.join(input_dir, KG_VALUES_CSV))
    # 3. prepare triplet edges
    # rename for clarity
    df_edges = kg_df.rename(columns={'sub_label':'head','obj_label':'tail'})
    entities = pd.unique(df_edges[['head','tail']].values.ravel())
    ent2idx  = {e:i for i,e in enumerate(entities)}
    df_edges['head_idx'] = df_edges['head'].map(ent2idx)
    df_edges['tail_idx'] = df_edges['tail'].map(ent2idx)

    # 4. node features: embeddings of entity strings
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings  = embed_model.encode(list(entities), convert_to_tensor=True)

    # 5. build edge_index tensor
    edge_index = torch.tensor(
        [df_edges['head_idx'].tolist(), df_edges['tail_idx'].tolist()],
        dtype=torch.long
    ).to(DEVICE)

    # 6. prepare labels & masks
    # map kg_values to indices
    values_df = values_df.rename(columns={'entity':'entity_label'})
    values_df['idx'] = values_df['entity_label'].map(ent2idx)
    values_df = values_df.dropna(subset=['idx'])
    values_df['idx'] = values_df['idx'].astype(int)

    # train/val/test splits
    train_idx, val_test_idx = train_test_split(
        values_df['idx'], test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE
    )
    val_idx, test_idx = train_test_split(
        val_test_idx, test_size=VAL_FRACTION_OF_TEST, random_state=SPLIT_RANDOM_STATE
    )

    num_nodes = len(entities)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    labels     = torch.zeros(num_nodes, dtype=torch.float)

    for _, row in values_df.iterrows():
        labels[row['idx']] = row['kg_value']
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    # 7. build PyG Data object
    data = Data(
        x         = embeddings.to(DEVICE),
        edge_index= edge_index,
        y         = labels.unsqueeze(1).to(DEVICE)
    )
    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    # 8. model, optimizer
    model = GraphSAGE(embeddings.shape[1], HIDDEN_DIM).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    # 9. training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        opt.zero_grad()
        out   = model(data.x, data.edge_index).squeeze()
        loss  = F.mse_loss(out[data.train_mask], data.y[data.train_mask].squeeze())
        loss.backward()
        opt.step()

        if epoch % 10 == 0 or epoch==EPOCHS:
            model.eval()
            with torch.no_grad():
                val_out   = model(data.x, data.edge_index).squeeze()
                val_loss  = F.mse_loss(
                    val_out[data.val_mask],
                    data.y[data.val_mask].squeeze()
                )
            print(f"Epoch {epoch:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

    # 10. predict for all nodes
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).squeeze().cpu().numpy()

    # 11. build result DataFrame
    result = pd.DataFrame({
        'entity_label': entities,
        'predicted_kg_value': preds,
    })
    # merge in true values & assign split
    result = result.merge(
        values_df[['entity_label','kg_value','idx']],
        on='entity_label',
        how='left'
    )
    def which_split(i):
        if i in train_idx.values: return 'train'
        if i in val_idx.values:   return 'valid'
        if i in test_idx.values:  return 'test'
        return 'none'
    result['split'] = result['idx'].apply(
        lambda x: which_split(x) if pd.notna(x) else 'none'
    )
    result = result[['entity_label','predicted_kg_value','kg_value','split']]

    # 12. save
    out_path = os.path.join(input_dir, OUTPUT_PRED_CSV)
    result.to_csv(out_path, index=False)
    print(f"Saved predictions → {out_path}")
