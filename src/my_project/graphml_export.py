"""
graphml_export.py
-----------------
Builds a NetworkX graph from the KG dataframe + entity knowledge scores,
then exports it as a .graphml file ready for Gephi / Cytoscape / NetworkX.

Node attributes: entity_label, kg_value, kg_source ('observed'|'predicted'|'none')
Edge attributes: triplet_id, rel_label
"""

from __future__ import annotations

import os
import pandas as pd
import networkx as nx


def build_and_export_graphml(
    kg_df: pd.DataFrame,
    entity_scores_df: pd.DataFrame,
    output_path: str,
    observed_col: str = "kg_value",
    predicted_col: str = "predicted_kg_value",
) -> str:
    """
    Parameters
    ----------
    kg_df            : raw KG with columns triplet_id, sub_label, rel_label, obj_label
    entity_scores_df : merged scores; should contain 'entity_label' column plus
                       at least one of observed_col / predicted_col
    output_path      : full path for the .graphml output file
    observed_col     : column name for observed/ground-truth kg_value
    predicted_col    : column name for GNN-predicted kg_value

    Returns
    -------
    output_path
    """
    # build score lookup
    score_map: dict[str, float]  = {}
    source_map: dict[str, str]   = {}

    for _, row in entity_scores_df.iterrows():
        ent = str(row["entity_label"])
        if observed_col in entity_scores_df.columns and pd.notna(row.get(observed_col)):
            score_map[ent]  = float(row[observed_col])
            source_map[ent] = "observed"
        elif predicted_col in entity_scores_df.columns and pd.notna(row.get(predicted_col)):
            score_map[ent]  = float(row[predicted_col])
            source_map[ent] = "predicted"

    G = nx.DiGraph()

    # add nodes
    all_entities = set(kg_df["sub_label"].tolist()) | set(kg_df["obj_label"].tolist())
    for ent in all_entities:
        G.add_node(
            ent,
            entity_label=ent,
            kg_value=score_map.get(ent, float("nan")),
            kg_source=source_map.get(ent, "none"),
        )

    # add edges
    for _, row in kg_df.iterrows():
        G.add_edge(
            str(row["sub_label"]),
            str(row["obj_label"]),
            triplet_id=str(row["triplet_id"]),
            rel_label=str(row.get("rel_label", "")),
        )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    nx.write_graphml(G, output_path)
    print(
        f"[graphml] nodes={G.number_of_nodes()}  "
        f"edges={G.number_of_edges()}  → {output_path}"
    )
    return output_path
