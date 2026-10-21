#!/usr/bin/env python3
"""
advCell Pipeline Step 1 repair: restore the var metadata
==================================================
Restores the var information (Ensembl ID, feature_name) lost in the sc.concat of Step 1.

Problem: sc.concat(join='outer') drops the var columns,
      and var_names become integers (0,1,2,...).

Repair method:
  1. Get the var information of Tabula Sapiens (61,497 genes) from the Census API
  2. Restore var of each pathway h5ad and overwrite the file

Usage:
  python repair_var_metadata.py --pathway_config data/pathway_config_geneformer.json

After the repair, rerun Step 1.5 (tokenize).
"""

import os
import json
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import cellxgene_census


def parse_args():
    p = argparse.ArgumentParser(description="Repair var metadata in pathway h5ads")
    p.add_argument("--pathway_config", type=str, required=True)
    p.add_argument("--census_version", type=str, default="2025-11-08")
    return p.parse_args()


def get_var_metadata_from_census(census_version, ts_tissue="liver"):
    """
    Fetch one tissue from the Census and extract the var metadata (feature_id, feature_name).
    All tissues have the same 61,497 genes, so any tissue will do.
    The lightweight liver is used.
    """
    print("Fetching var metadata from the Census API...")
    with cellxgene_census.open_soma(census_version=census_version) as census:
        # dataset_ids
        datasets = census["census_info"]["datasets"].read().concat().to_pandas()
        ts = datasets[datasets['collection_name'].str.contains('Tabula Sapiens', case=False, na=False)]
        ts_ids = ts['dataset_id'].tolist()

        id_filter = "dataset_id in ['" + "','".join(ts_ids) + "']"
        filter_str = f"{id_filter} and tissue == '{ts_tissue}'"

        # fetch only a small subset (only the var information is needed)
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_value_filter=filter_str,
            obs_column_names=["cell_type"],
        )

    print(f"  Fetched: {adata.n_obs} cells, {adata.n_vars} genes")
    print(f"  var columns: {list(adata.var.columns)}")
    print(f"  var_names[0]: {adata.var_names[0]}")
    print(f"  var_names[-1]: {adata.var_names[-1]}")

    if 'feature_name' in adata.var.columns:
        print(f"  feature_name[0]: {adata.var['feature_name'].iloc[0]}")

    return adata.var.copy(), adata.var_names.tolist()


def repair_h5ad(h5ad_path, ref_var, ref_var_names):
    """
    Restore var of the h5ad of one pathway.

    The sc.concat(join='outer') of Step 1 outer-joins the gene axis, so the
    number of columns (n_vars) should match ref.
    Set var_names to Ensembl IDs and restore the var columns.
    """
    adata = sc.read_h5ad(h5ad_path)

    n_vars = adata.n_vars
    n_ref = len(ref_var_names)

    if n_vars != n_ref:
        print(f"    WARNING: n_vars mismatch: h5ad={n_vars}, ref={n_ref}")
        # if the join grew the gene axis, ref has to be extended;
        # if an inner join shrank it, the corresponding genes have to be identified.
        # All tissues from the same Census should share 61,497 genes, so
        # a mismatch is most likely caused by the behaviour of sc.concat.
        if n_vars == n_ref:
            pass  # OK
        else:
            print(f"    Attempting positional mapping (assuming same gene order)...")
            # assume that, if n_vars <= n_ref, the first n_vars entries correspond
            # this may be inaccurate, so warn
            if n_vars < n_ref:
                ref_var = ref_var.iloc[:n_vars].copy()
                ref_var_names = ref_var_names[:n_vars]
            else:
                print(f"    SKIP: h5ad has MORE genes than ref. Cannot repair.")
                return False

    # restore var
    adata.var_names = ref_var_names
    for col in ref_var.columns:
        adata.var[col] = ref_var[col].values

    # also add the ensembl_id column (for Geneformer)
    if 'ensembl_id' not in adata.var.columns:
        if 'feature_id' in adata.var.columns:
            adata.var['ensembl_id'] = adata.var['feature_id']
        else:
            adata.var['ensembl_id'] = adata.var_names

    # save (overwrite)
    adata.write_h5ad(h5ad_path)
    return True


def main():
    args = parse_args()

    with open(args.pathway_config) as f:
        config = json.load(f)

    # get the reference var from the Census
    ref_var, ref_var_names = get_var_metadata_from_census(args.census_version)

    print(f"\nReference var: {len(ref_var_names)} genes")
    print(f"  Columns: {list(ref_var.columns)}")

    # repair the h5ad of each pathway
    print(f"\n{'='*70}")
    print(f"Starting repair: {len(config['pathways'])} pathways")
    print(f"{'='*70}")

    success = 0
    for pw in config["pathways"]:
        if pw.get("status") == "FAILED":
            continue

        h5ad_path = pw["h5ad_path"]
        pw_id = pw["id"]
        pw_name = pw["name"]

        print(f"\n  {pw_id} {pw_name}: {h5ad_path}")

        if not os.path.exists(h5ad_path):
            print(f"    SKIP: file not found")
            continue

        # check the state before the repair
        adata_check = sc.read_h5ad(h5ad_path, backed='r')
        print(f"    Before: var_names[0]={adata_check.var_names[0]}, "
              f"var columns={list(adata_check.var.columns)}, "
              f"n_vars={adata_check.n_vars}")
        adata_check.file.close()

        ok = repair_h5ad(h5ad_path, ref_var, ref_var_names)

        if ok:
            # check after the repair
            adata_check = sc.read_h5ad(h5ad_path, backed='r')
            print(f"    After:  var_names[0]={adata_check.var_names[0]}, "
                  f"var columns={list(adata_check.var.columns)}")
            adata_check.file.close()
            success += 1
            print(f"    ✓ Repaired")
        else:
            print(f"    ✗ Failed")

    print(f"\n{'='*70}")
    print(f"Repair finished: {success}/{len(config['pathways'])} pathways")
    print(f"{'='*70}")
    print(f"\nNext step: rerun Step 1.5")
    print(f"  python advcell_pipeline_step1_5_tokenize.py \\")
    print(f"      --pathway_config {args.pathway_config}")


if __name__ == "__main__":
    main()
