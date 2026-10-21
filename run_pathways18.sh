#!/usr/bin/env bash
# ============================================================
# Matched-budget comparison over 18 pathways (rebuilding the main table)
# ============================================================
# Same settings as the existing 5-pathway run; only the pathways are widened to P01..P18.
# Budget is 1,290 for k=2 and 2,580 for k=4 and k=8 (same as the 5-pathway run).
#
# Usage:
#   bash run_pathways18.sh <gf|scgpt|both> [ks] [methods]
#
#   bash run_pathways18.sh gf                    # k=2 4 8, gcg_faithful only
#   bash run_pathways18.sh gf 8 "gcg_faithful hill_random"
#
# Estimate (Geneformer, gcg_faithful only, 3 restarts):
#   k=2  1.39M forward   k=4  2.79M   k=8  2.79M   about 5 hours in total
#   scGPT takes 2-3 times as long
#
# The all-TF greedy (--greedy) is not used: the main table compares against
# greedy_grad_pool / greedy_random_pool, not the all-TF greedy.
# Order sensitivity (--order_sets) is also set to 0.
#
# Note: run with bash
set -uo pipefail

ENV_GF="geneformer"
ENV_SCGPT="scgpt"

init_conda () {
    local base
    base="$(conda info --base 2>/dev/null)" || {
        echo "[ERROR] conda not found"; exit 1; }
    set +u
    # shellcheck disable=SC1091
    . "$base/etc/profile.d/conda.sh"
    set -u
}

use_env () {
    set +u
    conda activate "$1" || { echo "[ERROR] conda activate $1 failed"; set -u; return 1; }
    set -u
    echo "[env] $(python -c 'import sys; print(sys.prefix)')"
}

# ---------- Paths (same as run_budget_match.sh) ----------
GF_MODEL="./Geneformer/Geneformer-V1-10M"
GF_DICT="./Geneformer/geneformer/gene_dictionaries_30m"
SCGPT_MODEL="./scGPT_human"
PATHWAY_CONFIG="./data/pathway_config_geneformer.json"
TF_LIST="./tf_list_human.txt"

# ---------- Experiment settings ----------
PATHWAYS="P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12 P13 P14 P15 P16 P17 P18"
N_PW=18
N_CELLS=20
N_TARGET=200
N_RESTARTS=3
N_CAND=128
SEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8

WHICH="${1:-both}"
KS="${2:-2 4 8}"
METHODS="${3:-gcg_faithful}"

budget_for () { if [ "$1" = "2" ]; then echo 1290; else echo 2580; fi; }

# Completion is judged by whether the number of pathways is complete.
# Checking only whether the file exists would take a run that crashed midway for complete.
done_already () {
    local f="$1"
    [ -s "$f" ] || return 1
    local n
    n="$(python -c "
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(len(d.get('results', {})))
except Exception:
    print(0)
" "$f")"
    [ "$n" = "$N_PW" ]
}

run_gf () {
    local K=$1 B OUT
    B=$(budget_for "$K")
    OUT="./pw18set_gf_k${K}"
    if done_already "$OUT/search_comparison.json"; then
        echo "=== Geneformer k=${K}: all 18 pathways present, skipping ($OUT) ==="
        return 0
    fi
    mkdir -p "$OUT"
    use_env "$ENV_GF" || return 1
    echo "=== Geneformer k=${K} budget=${B} pathways=${N_PW} methods=[${METHODS}]  $(date '+%F %T') ==="
    python compare_search_algorithms.py \
        --model_dir "$GF_MODEL" --pathway_config "$PATHWAY_CONFIG" \
        --dict_dir "$GF_DICT" --tf_list "$TF_LIST" \
        --pathways $PATHWAYS --n_cells "$N_CELLS" --n_target_cells "$N_TARGET" \
        --k_adv "$K" --budget "$B" --n_cand "$N_CAND" \
        --n_restarts "$N_RESTARTS" --emb_layer 0 \
        --methods $METHODS --order_sets 0 \
        --seed "$SEED" --device cuda --output_dir "$OUT" \
        2>&1 | tee -a "$OUT/log.txt"
}

run_scgpt () {
    local K=$1 B OUT
    B=$(budget_for "$K")
    OUT="./pw18set_scgpt_k${K}"
    if done_already "$OUT/search_comparison.json"; then
        echo "=== scGPT k=${K}: all 18 pathways present, skipping ($OUT) ==="
        return 0
    fi
    mkdir -p "$OUT"
    use_env "$ENV_SCGPT" || return 1
    echo "=== scGPT k=${K} budget=${B} pathways=${N_PW} methods=[${METHODS}]  $(date '+%F %T') ==="
    python compare_search_scgpt.py \
        --model_dir "$SCGPT_MODEL" --pathway_config "$PATHWAY_CONFIG" \
        --tf_list "$TF_LIST" \
        --pathways $PATHWAYS --n_cells "$N_CELLS" --n_target_cells "$N_TARGET" \
        --max_seq_len 1200 --binning_mode per_cell --high_value 50 \
        --k_adv "$K" --budget "$B" --n_cand "$N_CAND" \
        --n_restarts "$N_RESTARTS" \
        --methods $METHODS --order_sets 0 \
        --seed "$SEED" --device cuda --output_dir "$OUT" \
        2>&1 | tee -a "$OUT/log.txt"
}

init_conda
case "$WHICH" in
    gf)    NEED="$ENV_GF" ;;
    scgpt) NEED="$ENV_SCGPT" ;;
    both)  NEED="$ENV_GF $ENV_SCGPT" ;;
    *)     echo "[ERROR] first argument must be gf / scgpt / both"; exit 1 ;;
esac
for e in $NEED; do
    conda env list | awk '{print $1}' | grep -qx "$e" \
        || { echo "[ERROR] conda environment '$e' not found"; exit 1; }
done
echo "[preflight] target:$WHICH / k:$KS / methods:[$METHODS] / pathways:$N_PW"

for K in $KS; do
    case "$WHICH" in
        both)  run_gf "$K"    || echo "[WARN] Geneformer k=$K failed (continuing)"
               run_scgpt "$K" || echo "[WARN] scGPT k=$K failed (continuing)" ;;
        gf)    run_gf "$K"    || echo "[WARN] Geneformer k=$K failed (continuing)" ;;
        scgpt) run_scgpt "$K" || echo "[WARN] scGPT k=$K failed (continuing)" ;;
    esac
done

echo ""
echo "All conditions finished  $(date '+%Y-%m-%d %H:%M:%S')"

