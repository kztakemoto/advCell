#!/usr/bin/env python3
"""
advCell Pipeline Step 1.5: tokenization for Geneformer V1
=====================================================
Tokenize the per-pathway h5ads built in Step 1 for Geneformer V1.

The Census h5ad has:
  - raw counts (the format Geneformer requires)
  - var_names = Ensembl IDs (feature_id)
  - var['feature_name'] = gene symbols

TranscriptomeTokenizer:
  - requires the adata.var['ensembl_id'] column
  - sorts genes by rank from the raw counts and converts them to a token sequence
  - writes a .dataset (HuggingFace Dataset)

Usage:
  python advcell_pipeline_step1_5_tokenize.py \\
      --pathway_config data/pathway_config_geneformer.json \\
      --output_dir data/tokenized_v1

Output:
  data/tokenized_v1/
  ├── P01_Fib_Cardiomyocyte.dataset/
  ├── P02_Fib_Hepatocyte.dataset/
  ...
  └── P18_Fib_AT2.dataset/

No configuration file is written. data/pathway_config_geneformer.json is distributed
with tokenized_path included, and the datasets created here land at the paths
recorded there. The configuration file passed as input is not modified.
"""

import os
import json
import argparse
import numpy as np
import scanpy as sc
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Tokenize pathway h5ads for Geneformer V1")
    p.add_argument("--pathway_config", type=str, required=True,
                   help="Path to the pathway config (data/pathway_config_geneformer.json)")
    p.add_argument("--output_dir", type=str, default="data/tokenized_v1",
                   help="Output directory for tokenized datasets")
    p.add_argument("--model_version", type=str, default="V1",
                   choices=["V1", "V2"],
                   help="Geneformer model version (V1 recommended)")
    p.add_argument("--nproc", type=int, default=4,
                   help="Number of processes for tokenization")
    return p.parse_args()


def prepare_h5ad_for_geneformer(h5ad_path, prep_dir):
    """
    Convert a Census h5ad into the format that Geneformer's TranscriptomeTokenizer
    requires and save it to a temporary directory.
    
    Required conversions:
      1. adata.var['ensembl_id'] = Ensembl ID
      2. obs has the metadata columns ('cell_type' etc.)
      3. X is raw counts (the Census is raw counts, so unchanged)
    """
    adata = sc.read_h5ad(h5ad_path)

    # check the var structure of the Census h5ad
    # var_names are normally Ensembl IDs (ENSG...)
    # var['feature_id'] holds the same Ensembl IDs
    # var['feature_name'] is the gene symbol

    # add the ensembl_id column
    if 'ensembl_id' not in adata.var.columns:
        if 'feature_id' in adata.var.columns:
            adata.var['ensembl_id'] = adata.var['feature_id']
        elif str(adata.var_names[0]).startswith('ENSG'):
            adata.var['ensembl_id'] = adata.var_names
        else:
            raise ValueError(
                f"Cannot find Ensembl IDs in {h5ad_path}. "
                f"var columns: {list(adata.var.columns)}, "
                f"var_names[0]: {adata.var_names[0]}"
            )

    # check that cell_type is in the metadata
    if 'cell_type' not in adata.obs.columns:
        raise ValueError(f"'cell_type' not in obs columns: {list(adata.obs.columns)}")

    # add n_counts (required by TranscriptomeTokenizer)
    # total counts per cell = row sum of the raw counts
    if 'n_counts' not in adata.obs.columns:
        import scipy.sparse as sp
        if sp.issparse(adata.X):
            adata.obs['n_counts'] = np.array(adata.X.sum(axis=1)).flatten()
        else:
            adata.obs['n_counts'] = adata.X.sum(axis=1)
        print(f"    Added n_counts: median={adata.obs['n_counts'].median():.0f}, "
              f"range=[{adata.obs['n_counts'].min():.0f}, {adata.obs['n_counts'].max():.0f}]")

    # save to a temporary directory
    os.makedirs(prep_dir, exist_ok=True)
    basename = Path(h5ad_path).stem
    prep_path = os.path.join(prep_dir, f"{basename}.h5ad")
    adata.write_h5ad(prep_path)

    return prep_path, basename


def tokenize_single_pathway(h5ad_path, output_dir, model_version, nproc):
    """Tokenize the h5ad of one pathway"""
    from geneformer import TranscriptomeTokenizer
    import shutil

    basename = Path(h5ad_path).stem

    # independent temporary directory per pathway (so h5ads of other pathways do not leak in)
    prep_dir = os.path.join(output_dir, f"_prep_{basename}")
    if os.path.exists(prep_dir):
        shutil.rmtree(prep_dir)
    prep_path, _ = prepare_h5ad_for_geneformer(h5ad_path, prep_dir)

    # tokenize
    tk = TranscriptomeTokenizer(
        {"cell_type": "cell_type"},
        model_version=model_version,
        nproc=nproc,
    )

    tk.tokenize_data(
        data_directory=prep_dir,
        output_directory=output_dir,
        output_prefix=basename,
        file_format="h5ad",
    )

    # return the .dataset path
    dataset_path = os.path.join(output_dir, f"{basename}.dataset")
    if not os.path.exists(dataset_path):
        candidates = list(Path(output_dir).glob(f"{basename}*.dataset"))
        if candidates:
            dataset_path = str(candidates[0])
        else:
            print(f"    WARNING: .dataset not found at {dataset_path}")
            return None

    # delete together with the temporary directory
    shutil.rmtree(prep_dir, ignore_errors=True)

    return dataset_path


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # read pathway_config_geneformer.json
    with open(args.pathway_config) as f:
        config = json.load(f)

    print(f"{'='*70}")
    print(f"advCell Pipeline Step 1.5: Geneformer Tokenization")
    print(f"{'='*70}")
    print(f"Model version: {args.model_version}")
    print(f"Pathways: {len(config['pathways'])}")
    print(f"Output: {args.output_dir}")

    expected_paths = {pw["id"]: pw.get("tokenized_path") for pw in config["pathways"]}

    # tokenize each pathway
    results = []
    for pw in config["pathways"]:
        if pw.get("status") == "FAILED":
            results.append({**pw, "tokenized_path": None})
            continue

        pw_id = pw["id"]
        pw_name = pw["name"]
        h5ad_path = pw["h5ad_path"]

        print(f"\n--- {pw_id} {pw_name} ---")
        print(f"  Input: {h5ad_path} ({pw['total_n_cells']} cells)")

        try:
            dataset_path = tokenize_single_pathway(
                h5ad_path, args.output_dir,
                args.model_version, args.nproc
            )
            if dataset_path:
                print(f"  ✓ Output: {dataset_path}")
                pw["tokenized_path"] = dataset_path
            else:
                print(f"  ✗ Tokenization produced no output")
                pw["tokenized_path"] = None
        except Exception as e:
            print(f"  ✗ Error: {e}")
            pw["tokenized_path"] = None

        results.append(pw)

    # summary
    print(f"\n{'='*70}")
    print(f"STEP 1.5 COMPLETE")
    print(f"{'='*70}")
    ok = [r for r in results if r.get("tokenized_path")]
    fail = [r for r in results if not r.get("tokenized_path")]

    for r in ok:
        print(f"  ✓ {r['id']} {r['name']}")
    for r in fail:
        print(f"  ✗ {r['id']} {r['name']}")

    print(f"\n  Tokenized: {len(ok)}/{len(results)}")

    # The configuration file is not rewritten. Only check agreement with the recorded tokenized_path.
    for r in ok:
        expected = expected_paths.get(r["id"])
        if expected and os.path.normpath(expected) != os.path.normpath(r["tokenized_path"]):
            print(f"  ! {r['id']}: output location differs from the configuration file "
                  f"({r['tokenized_path']} vs {expected})")


if __name__ == "__main__":
    main()