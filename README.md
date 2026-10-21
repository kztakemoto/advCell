# advCell

This repository contains data and code for the research paper "In silico perturbation as an adversarial attack: combinatorial search over cell states in single-cell foundation models."

## Terms of Use

This project is licensed under the MIT License. When using this code, please cite our paper:

Oishi Y, Hamano M & Takemoto K (2026) **In silico perturbation as an adversarial attack: combinatorial search over cell states in single-cell foundation models.** bioRxiv. doi: <bioRxiv DOI>.

## Reproducing the numbers

The result files in `results/` contain everything behind the tables and
statistics, so they can be regenerated without a GPU or model weights. The
analysis scripts use the Python 3.10+ standard library only
(`figures/figures.py` also needs `numpy` and `matplotlib`).

```bash
# main comparison table and the values in figures/figure_data.py
python make_table3.py \
    --gf_attack results/pw18_gf_k{2,4,8}/search_comparison.json \
    --gf_greedy results/greedy_pool18_gf_k{2,4,8}/search_comparison.json \
    --sc_attack results/pw18_scgpt_k{2,4,8}/search_comparison.json \
    --sc_greedy results/greedy_pool18_scgpt_k{2,4,8}/search_comparison.json

# pathway-level sign tests
python pathway_level_signs.py \
    --attack results/pw18_gf_k{2,4,8}/search_comparison.json \
    --greedy results/greedy_pool18_gf_k{2,4,8}/search_comparison.json \
    --label Geneformer

# ranking evaluation
python eval_specificity.py \
    --gt_from results/multi_pathway_geneformer_k4_n50.json --step 20 \
    --results "Geneformer I-GPR(20)" results/multi_pathway_geneformer_k4_n50.json \
    --results "scGPT I-GPR(20)"      results/multi_pathway_scgpt_k4_n50.json \
    --deg "DEG(wilcoxon)"    results/deg_baseline_wilcoxon.json \
    --deg "DEG(target_expr)" results/deg_baseline_target_expr.json

# set-level recovery
python analyze_set_overlap.py \
    --results results/pw18set_gf_k8/search_comparison.json \
    --greedy  results/greedy_pool18set_gf_k8/search_comparison.json \
    --protocols protocols_by_pathway.json \
    --scores "Top-k single-gene"      results/multi_pathway_geneformer_k4_n50.json \
    --scores "DEG(wilcoxon) top-k"    results/deg_baseline_wilcoxon.json \
    --scores "DEG(target_expr) top-k" results/deg_baseline_target_expr.json \
    --n_vocab 2632 --out set_overlap_k8

# figures
cd figures && python figures.py
```

## Rerunning the searches

Requires a CUDA device, the Tabula Sapiens subsets and tokenized datasets
(built as described in `data/README.md`), and the pretrained models
(Geneformer V1, scGPT human; see `data/README.md` for sources). The two models
need separate conda environments.

```bash
conda env create -f environment-geneformer.yml
conda activate advcell-geneformer
export CUBLAS_WORKSPACE_CONFIG=:4096:8
bash run_pathways18.sh gf            # k = 2, 4, 8

conda env create -f environment-scgpt.yml
conda activate advcell-scgpt
export CUBLAS_WORKSPACE_CONFIG=:4096:8
bash run_pathways18.sh scgpt

bash run_greedy_pool18.sh            # greedy baselines at matched budget
```

With `--seed 42` and `CUBLAS_WORKSPACE_CONFIG=:4096:8` the Geneformer results
reproduce exactly.

## Layout

```
compare_search_algorithms.py   search driver, Geneformer
compare_search_scgpt.py        search driver, scGPT (uses search_algorithms.py)
gpr_*_multi_pathway.py         single-gene scores (input of the ranking evaluation)
analyze_additivity.py, summarize_additivity.py, analyze_2x2_interaction.py
                               non-additivity and the 2x2 factorial
eval_specificity.py, analyze_set_overlap.py, analyze_complementarity.py
                               ranking evaluation and set-level recovery
make_table3.py, pathway_level_signs.py
                               main comparison table and its cross-check
advcell_pipeline_step1*.py, repair_var_metadata.py, make_celltype_counts.py
                               data preparation (data/README.md)
run_pathways18.sh, run_greedy_pool18.sh
                               experiment runs
data/pathway_config_geneformer.json   18 pathways, cell types, factor sets
protocols_by_pathway.json      published factor sets per protocol
tf_list_human.txt              transcription factor list
results/                       result JSONs behind every number
figures/                       figure data and plotting code
```
