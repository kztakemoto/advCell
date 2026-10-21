#!/usr/bin/env python3
"""
advCell Pipeline Step 1: Data Acquisition & Preprocessing
=========================================================
Fetch data for 18 pathways from Tabula Sapiens (v1+v2) and build,
for each pathway, a combined h5ad (source + target in one file).

Source-cell tissues are chosen to follow the original papers.
The output is compatible with the existing gpr_*_multi_pathway.py scripts.

Usage:
  python advcell_pipeline_step1.py [--output_dir data] [--max_cells 2000] [--seed 42]

Output:
  {output_dir}/
  ├── pathway_config_geneformer.json
  └── subsets/
      ├── P01_Fib_Cardiomyocyte.h5ad   (source + target together)
      ├── P02_Fib_Hepatocyte.h5ad
      ...
      └── P18_Fib_AT2.h5ad
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import cellxgene_census
import warnings
warnings.filterwarnings('ignore')

# ==============================================================
# Definition of the 18 pathways (following the original papers)
# tf_union / paper_source / curation_note are the curated factor sets and their
# sources. The configuration file written by this script has the same content
# as the distributed data/pathway_config_geneformer.json.
# ==============================================================
PATHWAYS = [
    # ── Evaluated 4 ──
    {
        "id": "P01", "name": "Fib_Cardiomyocyte",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["regular atrial cardiac myocyte"],
        "target_tissues": ["cardiac atrium", "heart", "right cardiac atrium"],
        "tf_union": ["ESRRG","GATA4","HAND2","MEF2C","MESP1","MYOCD","NKX2-5",
                     "SMARCD3","SRF","TBX5","ZFPM2","ZNF281"],
        "paper_source": "Song et al. (2012) Nature 485:599–604; Wada et al. (2013) Proc. Natl. Acad. Sci. U.S.A. 110:12667–12672; Zhou et al. (2016) Cell Stem Cell 18:382–395; Addis et al. (2013) J. Mol. Cell. Cardiol. 60:97–106; Singh et al. (2016) J. Am. Heart Assoc. 5:e003922; Muraoka et al. (2014) EMBO J. 33:1565–1581; Fu et al. (2013) Stem Cell Rep. 1:235–247; Protze et al. (2012) J. Mol. Cell. Cardiol. 53:323–332; Christoforou et al. (2013) PLoS One 8:e63577; Ieda et al. (2010) Cell 142:375–386; Nam et al. (2013) Proc. Natl. Acad. Sci. U.S.A. 110:5588–5593; Zhou et al. (2017) Genes Dev. 31:1770–1783",
        "status": "evaluated",
    },
    {
        "id": "P02", "name": "Fib_Hepatocyte",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen", "skin of chest"],
        "target_cell_types": ["hepatocyte"],
        "target_tissues": ["liver"],
        "tf_union": ["ATF5","CEBPA","FOXA1","FOXA2","FOXA3","GATA3","GATA4","HNF1A",
                     "HNF1B","HNF4A","KDM2B","KLF4","MYC","NR1I2","ONECUT1","POU5F1",
                     "PROX1","SOX2"],
        "paper_source": "Sekiya and Suzuki (2011) Nature 475:390–393; Huang et al. (2011) Nature 475:386–389; Du et al. (2014) Cell Stem Cell 14:394–403; Chen et al. (2018) Biotechnol. Bioeng. 115:2807–2816; Park et al. (2019) Stem Cell Res Ther 10:21; Nakamori et al. (2017) Sci. Rep. 7:16675; Huang et al. (2014) Cell Stem Cell 14:370–384; Simeonov and Uppal (2014) PLoS One 9:e100134; Song et al. (2016) Cell Stem Cell 18:797–808; Pournasr et al. (2015) Eur. J. Cell Biol. 94:603–610; Lim et al. (2016) Cell Rep. 15:814–829; Zakikhan et al. (2016) Biochem. Biophys. Res. Commun. 474:97–103; Yu et al. (2013) Cell Stem Cell 13:328–340; Rezvani et al. (2016) Cell Stem Cell 18:809–816; Wu et al. (2017) Exp. Ther. Med. 13:2685–2690",
        "status": "evaluated",
        "curation_note": "SOX2 is included because it is a factor of protocol C101 (Pournasr et al., 2015). The factor union is the set of factors of all protocols listed in protocols_by_pathway.json.",
    },
    {
        "id": "P03", "name": "Bcell_Macrophage",
        "source_cell_types": ["B cell"],
        "source_tissues": ["blood", "spleen", "bone marrow"],
        "target_cell_types": ["macrophage"],
        "target_tissues": ["blood", "spleen"],
        "tf_union": ["CEBPA","CEBPB"],
        "paper_source": "Xie et al. (2004) Cell 117:663–676; Bussmann et al. (2009) Cell Stem Cell 5:554–566",
        "status": "evaluated",
    },
    {
        "id": "P04", "name": "Fib_SkeletalMuscle",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["skeletal muscle satellite stem cell"],
        "target_tissues": ["muscle of abdomen", "muscle of pelvic diaphragm"],
        "tf_union": ["MEF2B","MYCL","MYOD1","PAX3","PAX7","PITX1"],
        "paper_source": "Ito et al. (2017) Sci. Rep. 7:8097; Wakao et al. (2017) Biochem. Biophys. Res. Commun. 488:368–373; Davis et al. (1987) Cell 51:987–1000",
        "status": "evaluated",
    },

    # ── New 14 ──
    {
        "id": "P05", "name": "Tcell_Macrophage",
        "source_cell_types": ["CD4-positive, alpha-beta T cell",
                              "CD8-positive, alpha-beta T cell"],
        "source_tissues": ["blood", "spleen"],
        "target_cell_types": ["macrophage"],
        "target_tissues": ["blood", "spleen"],
        "tf_union": ["CEBPA","CEBPB"],
        "paper_source": "Laiosa et al. (2006) Immunity 25:731–744",
        "status": "new",
        "curation_note": "The starting cells in the original study are committed pre-T progenitors.",
    },
    {
        "id": "P06", "name": "Fib_Endothelial",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["endothelial cell"],
        "target_tissues": ["aorta", "lung"],
        "tf_union": ["ETV2"],
        "paper_source": "Morita et al. (2015) Proc. Natl. Acad. Sci. U.S.A. 112:160–165; Lee et al. (2017) Circ. Res. 120:848–861",
        "status": "new",
        "curation_note": "Only the conversion of fibroblasts to endothelial cells by ETV2 alone is included. The ERG/FLI1/ETV2 protocol of Ginsberg et al. (2012) targets amniotic cells and is not included.",
    },
    {
        "id": "P07", "name": "Fib_SmoothMuscle",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["smooth muscle cell"],
        "target_tissues": ["aorta", "bladder organ"],
        "tf_union": ["MYOCD"],
        "paper_source": "Wang et al. (2003) Proc. Natl. Acad. Sci. U.S.A. 100:7129–7134",
        "status": "new",
        "curation_note": "The factor set is MYOCD alone.",
    },
    {
        "id": "P08", "name": "Fib_DendriticCell",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["myeloid dendritic cell"],
        "target_tissues": ["blood", "spleen", "lung"],
        "tf_union": ["BATF3","IRF8","SPI1"],
        "paper_source": "Rosa et al. (2018) Sci. Immunol. 3:eaau4292",
        "status": "new",
    },
    {
        "id": "P09", "name": "Fib_ErythroidProg",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["erythroid progenitor cell"],
        "target_tissues": ["bone marrow"],
        "tf_union": ["GATA1","LMO2","MYC","TAL1"],
        "paper_source": "Capellera-Garcia et al. (2016) Cell Rep. 15:2550–2562",
        "status": "new",
    },
    {
        "id": "P10", "name": "Fib_HSC",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["hematopoietic stem cell"],
        "target_tissues": ["bone marrow", "blood"],
        "tf_union": ["ERG","GATA2","LMO2","RUNX1","TAL1"],
        "paper_source": "Batta et al. (2014) Cell Rep. 9:1871–1884",
        "status": "new",
    },
    {
        "id": "P11", "name": "Fib_Enterocyte",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["enterocyte of epithelium proper of duodenum"],
        "target_tissues": ["duodenum", "small intestine"],
        "tf_union": ["CDX2","FOXA3","GATA6","HNF4A"],
        "paper_source": "Miura and Suzuki (2017) Cell Stem Cell 21:456–471",
        "status": "new",
        "curation_note": "The factor set is the four factors Hnf4a, Foxa3, Gata6 and Cdx2 of Miura and Suzuki (2017).",
    },
    {
        "id": "P12", "name": "Fib_BetaCell",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["type B pancreatic cell"],
        "target_tissues": ["exocrine pancreas", "endocrine pancreas"],
        "tf_union": ["MAFA","NEUROG3","NKX2-2","PAX4","PDX1"],
        "paper_source": "Fontcuberta-PiSunyer et al. (2023) Commun. Biol. 6:256",
        "status": "new",
    },
    {
        "id": "P13", "name": "Hepatocyte_BetaCell",
        "source_cell_types": ["hepatocyte"],
        "source_tissues": ["liver"],
        "target_cell_types": ["type B pancreatic cell"],
        "target_tissues": ["exocrine pancreas", "endocrine pancreas"],
        "tf_union": ["MAFA","MAFB","NEUROD1","NEUROG3","PAX4","PDX1","TGIF2"],
        "paper_source": "Wang et al. (2007) Mol. Ther. 15:255–263; Nagasaki et al. (2014) PLoS One 9:e113022; Meivar-Levy et al. (2011) J. Transplant. 2011:252387; Ferber et al. (2000) Nat. Med. 6:568–572; Ber et al. (2003) J. Biol. Chem. 278:31950–31957; Berneman-Zeitouni et al. (2014) PLoS One 9:e87812; Cerdá-Esteban et al. (2017) Nat. Commun. 8:14127; Lu et al. (2014) Biochem. Biophys. Res. Commun. 453:405–410",
        "status": "new",
    },
    {
        "id": "P14", "name": "Exocrine_BetaCell",
        "source_cell_types": ["pancreatic acinar cell"],
        "source_tissues": ["exocrine pancreas"],
        "target_cell_types": ["type B pancreatic cell"],
        "target_tissues": ["exocrine pancreas", "endocrine pancreas"],
        "tf_union": ["MAFA","NEUROG3","PDX1"],
        "paper_source": "Zhou et al. (2008) Nature 455:627–632",
        "status": "new",
    },
    {
        "id": "P15", "name": "Fib_Melanocyte",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["melanocyte"],
        "target_tissues": ["skin of body", "skin of abdomen", "skin of chest"],
        "tf_union": ["MITF","PAX3","SOX10"],
        "paper_source": "Yang et al. (2014) Nat. Commun. 5:5807",
        "status": "new",
    },
    {
        "id": "P16", "name": "Fib_SchwannCell",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["Schwann cell"],
        "target_tissues": ["lacrimal gland", "anterior part of tongue",
                           "posterior part of tongue"],
        "tf_union": ["EGR2","SOX10"],
        "paper_source": "Mazzara et al. (2017) Nat. Commun. 8:14088",
        "status": "new",
        "curation_note": "The factor set is the two factors Sox10 and Egr2 of Mazzara et al. (2017). POU3F1 did not increase the yield in that study and is not included.",
    },
    {
        "id": "P17", "name": "Fib_RenalEpithelial",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["kidney epithelial cell"],
        "target_tissues": ["kidney", "cortex of kidney"],
        "tf_union": ["EMX2","HNF1B","HNF4A","PAX8"],
        "paper_source": "Kaminski et al. (2016) Nat. Cell Biol. 18:1269–1280",
        "status": "new",
    },
    {
        "id": "P18", "name": "Fib_AT2",
        "source_cell_types": ["fibroblast"],
        "source_tissues": ["skin of body", "skin of abdomen"],
        "target_cell_types": ["pulmonary alveolar type 2 cell"],
        "target_tissues": ["lung"],
        "tf_union": ["FOXA1","FOXA2","GATA6","NKX2-1"],
        "paper_source": "Morita et al. (2025) npj Regen. Med. 10:30",
        "status": "new",
        "curation_note": "The factor set is the four factors Nkx2-1, Foxa1, Foxa2 and Gata6 of Morita et al. (2025).",
    },
]


def parse_args():
    p = argparse.ArgumentParser(description="advCell Pipeline Step 1")
    p.add_argument("--output_dir", type=str, default="data",
                   help="Output directory")
    p.add_argument("--max_cells", type=int, default=2000,
                   help="Max cells per cell type per pathway")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sampling")
    p.add_argument("--census_version", type=str, default="2025-11-08",
                   help="CELLxGENE Census version (pin for reproducibility)")
    return p.parse_args()


def get_all_required_tissues():
    """Collect the tissues required across all pathways"""
    tissues = set()
    for pw in PATHWAYS:
        tissues.update(pw["source_tissues"])
        tissues.update(pw["target_tissues"])
    return sorted(tissues)


def get_ts_dataset_ids(census):
    """Get the list of Tabula Sapiens dataset_ids"""
    datasets = census["census_info"]["datasets"].read().concat().to_pandas()
    ts = datasets[
        datasets['collection_name'].str.contains('Tabula Sapiens',
                                                  case=False, na=False)
    ]
    print(f"  Tabula Sapiens datasets found: {len(ts)}")
    return ts['dataset_id'].tolist()


def download_tissue(tissue_name, census, ts_ids):
    """Fetch the data of one tissue from the Census API"""
    id_filter = "dataset_id in ['" + "','".join(ts_ids) + "']"
    filter_str = f"{id_filter} and tissue == '{tissue_name}'"
    try:
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_value_filter=filter_str,
            obs_column_names=[
                "cell_type", "tissue", "donor_id", "assay",
                "cell_type_ontology_term_id", "is_primary_data",
            ],
        )
        return adata
    except Exception as e:
        print(f"    WARNING: {tissue_name} failed: {e}")
        return None


def build_pathway_h5ad(pw, tissue_adatas, max_cells, rng):
    """
    Build the combined h5ad of one pathway.
    Compatible with the existing gpr_*_multi_pathway.py:
      - source/target are distinguished by obs['cell_type']
      - both are in a single h5ad
    """
    pw_id = pw["id"]
    pw_name = pw["name"]

    # ── Source cells ──
    source_frames = []
    for t in pw["source_tissues"]:
        if t in tissue_adatas and tissue_adatas[t] is not None:
            ad = tissue_adatas[t]
            mask = ad.obs['cell_type'].isin(pw["source_cell_types"])
            if mask.sum() > 0:
                source_frames.append(ad[mask].copy())

    if not source_frames:
        print(f"  ✗ {pw_id} {pw_name}: NO SOURCE CELLS")
        return None

    source = sc.concat(source_frames, join='outer')
    combined.var = source.var.copy()
    combined.var_names = source.var_names.copy()
    # Remove duplicates (same cell from overlapping tissue queries)
    source.obs_names_make_unique()

    # ── Target cells ──
    target_frames = []
    for t in pw["target_tissues"]:
        if t in tissue_adatas and tissue_adatas[t] is not None:
            ad = tissue_adatas[t]
            mask = ad.obs['cell_type'].isin(pw["target_cell_types"])
            if mask.sum() > 0:
                target_frames.append(ad[mask].copy())

    if not target_frames:
        print(f"  ✗ {pw_id} {pw_name}: NO TARGET CELLS")
        return None

    target = sc.concat(target_frames, join='outer')
    target.obs_names_make_unique()

    # ── Sample ──
    if source.n_obs > max_cells:
        idx = rng.choice(source.n_obs, max_cells, replace=False)
        source = source[sorted(idx)].copy()
    if target.n_obs > max_cells:
        idx = rng.choice(target.n_obs, max_cells, replace=False)
        target = target[sorted(idx)].copy()

    # ── Combine ──
    combined = sc.concat([source, target], join='outer')
    combined.obs_names_make_unique()

    n_src = source.n_obs
    n_tgt = target.n_obs
    src_types = source.obs['cell_type'].unique().tolist()
    tgt_types = target.obs['cell_type'].unique().tolist()

    print(f"  ✓ {pw_id} {pw_name}: "
          f"src={n_src} ({', '.join(src_types)}) | "
          f"tgt={n_tgt} ({', '.join(tgt_types)}) | "
          f"genes={combined.n_vars}")

    return combined, n_src, n_tgt, src_types, tgt_types


def main():
    args = parse_args()
    subset_dir = os.path.join(args.output_dir, "subsets")
    os.makedirs(subset_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    required_tissues = get_all_required_tissues()
    print(f"\n{'='*70}")
    print(f"advCell Pipeline Step 1")
    print(f"{'='*70}")
    print(f"Pathways: {len(PATHWAYS)}")
    print(f"Required tissues: {len(required_tissues)}")
    print(f"Max cells/type: {args.max_cells}")
    print(f"Census version: {args.census_version}")

    # ── Download ──
    tissue_adatas = {}
    print(f"\n--- Downloading from Census API ---")
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        ts_ids = get_ts_dataset_ids(census)

        for i, tissue_name in enumerate(required_tissues):
            print(f"  [{i+1}/{len(required_tissues)}] {tissue_name}...", end=" ")
            ad = download_tissue(tissue_name, census, ts_ids)
            if ad is not None and ad.n_obs > 0:
                tissue_adatas[tissue_name] = ad
                print(f"{ad.n_obs:,} cells, {ad.n_vars:,} genes")
            else:
                print("EMPTY or FAILED")

    print(f"\n  Downloaded {len(tissue_adatas)}/{len(required_tissues)} tissues")

    # ── Build per-pathway h5ad ──
    print(f"\n--- Building pathway subsets ---")
    config = {
        "pipeline": "advCell Step 1",
        "census_version": args.census_version,
        "max_cells_per_type": args.max_cells,
        "random_seed": args.seed,
        "n_pathways": len(PATHWAYS),
        "pathways": [],
    }

    for pw in PATHWAYS:
        result = build_pathway_h5ad(pw, tissue_adatas, args.max_cells, rng)
        if result is None:
            config["pathways"].append({
                "id": pw["id"], "name": pw["name"],
                "status": "FAILED", "error": "no cells found",
            })
            continue

        combined, n_src, n_tgt, src_types, tgt_types = result

        # Save
        h5ad_path = os.path.join(subset_dir, f"{pw['id']}_{pw['name']}.h5ad")
        combined.write_h5ad(h5ad_path)

        entry = {
            "id": pw["id"],
            "name": pw["name"],
            "h5ad_path": h5ad_path,
            "source_cell_types": src_types,
            "target_cell_types": tgt_types,
            "source_n_cells": n_src,
            "target_n_cells": n_tgt,
            "total_n_cells": n_src + n_tgt,
            "n_genes": int(combined.n_vars),
            "tf_union": pw["tf_union"],
            "tf_pool_size": len(pw["tf_union"]),
            "paper_source": pw["paper_source"],
            "status": pw["status"],
            # Output location of Step 1.5 (tokenization); matches the default --output_dir.
            "tokenized_path": os.path.join(
                args.output_dir, "tokenized_v1",
                f"{pw['id']}_{pw['name']}.dataset"),
        }
        if "curation_note" in pw:
            entry["curation_note"] = pw["curation_note"]
        config["pathways"].append(entry)

    config["geneformer_model_version"] = "V1"

    # ── Save config ──
    config_path = os.path.join(args.output_dir, "pathway_config_geneformer.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"STEP 1 COMPLETE")
    print(f"{'='*70}")
    ok = [p for p in config["pathways"] if p.get("status") != "FAILED"]
    fail = [p for p in config["pathways"] if p.get("status") == "FAILED"]

    for p in ok:
        tag = "[EVAL]" if p["status"] == "evaluated" else "[NEW] "
        print(f"  ✓ {p['id']} {p['name']:25s} {tag} "
              f"src:{p['source_n_cells']:5d}  tgt:{p['target_n_cells']:5d}  "
              f"TFpool:{p['tf_pool_size']}")
    for p in fail:
        print(f"  ✗ {p['id']} {p['name']:25s} FAILED: {p.get('error','')}")

    print(f"\n  Successful: {len(ok)}/{len(PATHWAYS)}")
    print(f"  Output: {subset_dir}/")
    print(f"  Config: {config_path}")

    # ── Usage hint ──
    print(f"""
Next steps (run for each model):

  # Geneformer V1
  python gpr_geneformer_multi_pathway.py \\
      --model_dir ./geneformer_v1 \\
      --pathway_config {config_path} \\
      --tf_list ./tf_list_human.txt \\
      --n_cells 50 --k_adv 2 --igpr_steps 5 10 20

  # scGPT
  python gpr_scgpt_multi_pathway.py \\
      --model_dir ./scGPT_human \\
      --pathway_config {config_path} \\
      --tf_list ./tf_list_human.txt \\
      --n_cells 50 --k_adv 4 --igpr_steps 5 10 20
""")


if __name__ == "__main__":
    main()
