"""
advcell_pathway_loader.py
=========================
Reads pathway_config_geneformer.json and converts it into a form compatible
with the existing PATHWAYS dictionary.

Shared by the multi_pathway scripts of all models.

Usage:
    from advcell_pathway_loader import load_pathway_config

    pathways, h5ad_paths = load_pathway_config("data/pathway_config_geneformer.json")
    # pathways: {pw_id: {"source_type": [...], "target_type": [...], "gt_tfs": [...]}}
    # h5ad_paths: {pw_id: "data/subsets/P01_Fib_Cardiomyocyte.h5ad"}
"""

import json


def load_pathway_config(config_path, model="scgpt"):
    """
    Read pathway_config_geneformer.json and convert it into a dictionary compatible with existing code.

    Args:
        config_path: pathway_config_geneformer.json
        model: "scgpt", "scfoundation", or "geneformer"

    Returns:
        pathways: dict  {pw_id: {"source_type": list, "target_type": list,
                                  "gt_tfs": list, "name": str, "status": str}}
        data_paths: dict  {pw_id: path_to_h5ad_or_dataset}
    """
    with open(config_path) as f:
        config = json.load(f)

    pathways = {}
    data_paths = {}

    for pw in config["pathways"]:
        pw_id = pw["id"]

        # skip failed pathways
        if pw.get("status") == "FAILED":
            continue
        if model == "geneformer" and not pw.get("tokenized_path"):
            continue

        pathways[pw_id] = {
            "name": pw["name"],
            "source_type": pw["source_cell_types"],      # list
            "target_type": pw["target_cell_types"],      # list
            "gt_tfs": pw["tf_union"],                    # list
            "tf_pool_size": pw.get("tf_pool_size", len(pw["tf_union"])),
            "status": pw.get("status", "new"),
        }

        if model == "geneformer":
            data_paths[pw_id] = pw.get("tokenized_path")
        else:
            data_paths[pw_id] = pw.get("h5ad_path")

    return pathways, data_paths


def get_pathway_ids(pathways, selected=None):
    """
    Return the list of pathway IDs to run.

    Args:
        pathways: return value of load_pathway_config()
        selected: None (all pathways) or list of pw_ids

    Returns:
        list of pw_id strings
    """
    if selected:
        return [pid for pid in selected if pid in pathways]
    return list(pathways.keys())
