# Data

The analysis uses 18 direct reprogramming pathways built from Tabula Sapiens.
The per-pathway `.h5ad` subsets and tokenized datasets are tens of gigabytes
and are not distributed; this file describes how to rebuild them. The result
files in `results/` do not depend on them.

## Sources

- **Tabula Sapiens** (v1 and v2), CELLxGENE collection
  https://cellxgene.cziscience.com/collections/e5f58829-1a66-40b5-a624-9046778e74f5,
  accessed through the CELLxGENE Census, release **2025-11-08**
  (https://chanzuckerberg.github.io/cellxgene-census/).
- **Geneformer V1**: https://huggingface.co/ctheodoris/Geneformer
  (`Geneformer-V1-10M` and `geneformer/gene_dictionaries_30m`).
- **scGPT human**: https://github.com/bowang-lab/scGPT (`scGPT_human`
  checkpoint). The Google Drive folder listed in its README at the time of this
  study was
  https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y?usp=sharing;
  the link may change.
- **Transcription factors**: AnimalTFDB4, http://bioinfo.life.hust.edu.cn/AnimalTFDB4/
  (`tf_list_human.txt`, included).

The models are not redistributed. Place them at the paths set at the top of
`run_pathways18.sh`.

## Rebuilding

Steps 1 and 1.5 need `cellxgene-census`, `scanpy` and `anndata`; step 1.5 also
needs the Geneformer environment.

```bash
# 0. cell-type inventory and pathway feasibility
python download_tabula_sapiens_v2_filtered.py

# 1. per-pathway subsets -> data/subsets/, data/pathway_config_geneformer.json
python advcell_pipeline_step1.py --output_dir data --max_cells 2000 --seed 42

# 1.1 restore the var metadata dropped by the concatenation in step 1 (required)
python repair_var_metadata.py --pathway_config data/pathway_config_geneformer.json

# 1.5 tokenize for Geneformer -> data/tokenized_v1/
python advcell_pipeline_step1_5_tokenize.py \
    --pathway_config data/pathway_config_geneformer.json

# 2. cell-type counts for the supplementary table
python make_celltype_counts.py
```

`data/pathway_config_geneformer.json` is the only configuration file. The
factor sets and `paper_source` entries are defined in `PATHWAYS` in
`advcell_pipeline_step1.py`; step 1.5 does not modify the config.
