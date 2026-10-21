#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recount cells per cell type x tissue in Tabula Sapiens.

Counted with the census version recorded in pathway_config_geneformer.json. Outputs:

    data/tabula_celltype_counts.json   totals per cell type and the per-tissue breakdown
    data/tabula_celltype_counts.tsv    the same as a table (for inspection)

Usage:
    python make_celltype_counts.py                     # all cell types
    python make_celltype_counts.py --grep adipo        # show only partial matches
    python make_celltype_counts.py --grep 'adipo|schwann|dendritic'

Only obs is read, so it takes a few minutes and a few GB. The X matrix is not read.
"""

import argparse
import json
import os
import re
import sys

CONFIG = "data/pathway_config_geneformer.json"
OUT_JSON = "data/tabula_celltype_counts.json"
OUT_TSV = "data/tabula_celltype_counts.tsv"


def census_version_from_config(path=CONFIG):
    """Use the census version recorded in the configuration file."""
    if not os.path.exists(path):
        sys.exit(f"[error] {path} not found")
    with open(path) as f:
        return json.load(f)["census_version"]


def load_obs(version, dataset_filter):
    import cellxgene_census
    print(f"[info] opening census {version}")
    with cellxgene_census.open_soma(census_version=version) as census:
        cols = ["cell_type", "tissue", "dataset_id", "is_primary_data"]
        obs = (census["census_data"]["homo_sapiens"]
               .obs.read(column_names=cols,
                         value_filter="is_primary_data == True")
               .concat().to_pandas())
    print(f"[info] read {len(obs):,} cells")

    if dataset_filter:
        import cellxgene_census
        with cellxgene_census.open_soma(census_version=version) as census:
            ds = (census["census_info"]["datasets"]
                  .read().concat().to_pandas())
        keep = ds[ds["collection_name"].str.contains(dataset_filter,
                                                     case=False, na=False)]
        if keep.empty:
            sys.exit(f"[error] no collection matches '{dataset_filter}'. "
                     f"Drop --dataset or check the spelling")
        print("[info] target collections: "
              + ", ".join(sorted(keep["collection_name"].unique())))
        obs = obs[obs["dataset_id"].isin(set(keep["dataset_id"]))]
        print(f"[info] {len(obs):,} cells after filtering")
    return obs


def build(obs):
    by_type = obs.groupby("cell_type").size().sort_values(ascending=False)
    by_pair = obs.groupby(["cell_type", "tissue"]).size()
    out = {}
    for ct, n in by_type.items():
        tis = by_pair.loc[ct].sort_values(ascending=False)
        out[ct] = {"total": int(n),
                   "n_tissues": int(len(tis)),
                   "tissues": {t: int(v) for t, v in tis.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census_version", default=None,
                    help="default: the value in pathway_config_geneformer.json")
    ap.add_argument("--dataset", default="Tabula Sapiens",
                    help="partial match on the collection name. Pass an empty string "
                         "to use the whole census")
    ap.add_argument("--grep", default=None,
                    help="regular expression on cell-type names. Only matches are shown")
    ap.add_argument("--min_cells", type=int, default=0,
                    help="omit cell types with fewer cells than this from the display")
    args = ap.parse_args()

    version = args.census_version or census_version_from_config()
    obs = load_obs(version, args.dataset)
    counts = build(obs)

    os.makedirs("data", exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"census_version": version,
                   "dataset_filter": args.dataset,
                   "n_cell_types": len(counts),
                   "counts": counts}, f, indent=2, ensure_ascii=False)
    os.replace(tmp, OUT_JSON)

    with open(OUT_TSV, "w") as f:
        f.write("cell_type\ttotal\tn_tissues\ttop_tissue\ttop_tissue_n\n")
        for ct, r in sorted(counts.items(), key=lambda x: -x[1]["total"]):
            t0 = next(iter(r["tissues"].items()))
            f.write(f"{ct}\t{r['total']}\t{r['n_tissues']}\t{t0[0]}\t{t0[1]}\n")

    print(f"[saved] {OUT_JSON}  ({len(counts)} cell types)")
    print(f"[saved] {OUT_TSV}")

    if args.grep:
        pat = re.compile(args.grep, re.I)
        hits = {c: r for c, r in counts.items() if pat.search(c)}
        if not hits:
            print(f"\n No cell type matches '{args.grep}'")
            return
        print(f"\n=== Cell types matching '{args.grep}' ===")
        for ct, r in sorted(hits.items(), key=lambda x: -x[1]["total"]):
            if r["total"] < args.min_cells:
                continue
            print(f"\n{ct}   total {r['total']:,}   tissues {r['n_tissues']}")
            for t, v in list(r["tissues"].items())[:8]:
                print(f"    {t:34s} {v:7,}")


if __name__ == "__main__":
    main()
