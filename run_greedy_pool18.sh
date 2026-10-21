#!/usr/bin/env bash
# ============================================================
# Matched-budget greedy control experiment (18-pathway version)
# ============================================================
# Same settings as run_greedy_pool.sh; only the pathways are widened to P01..P18.
# The attack side corresponds to pw18_<model>_k<K> written by run_pathways18.sh.
#
#     k=2: budget 1290, pool 645 -> greedy 1,290
#     k=4: budget 2580, pool 645 -> greedy 2,580
#     k=8: budget 2580, pool 322 -> greedy 2,576
#
#   bash run_greedy_pool18.sh          # both models, k=2/4/8
#   bash run_greedy_pool18.sh gf       # Geneformer only
#   bash run_greedy_pool18.sh scgpt    # scGPT only
#
# Estimated time: Geneformer about 3 hours, scGPT 6-9 hours
set -uo pipefail

# ---------- conda environments ----------
# The original run_greedy_pool.sh does not switch conda environments, so activate
# the environment of the target model before running. Avoid "both".
ENV_GF="geneformer"
ENV_SCGPT="scgpt"

# ---------- Paths (check before running) ----------
GF_MODEL="./Geneformer/Geneformer-V1-10M"
GF_DICT="./Geneformer/geneformer/gene_dictionaries_30m"
SCGPT_MODEL="./scGPT_human"
PATHWAY_CONFIG="./data/pathway_config_geneformer.json"
TF_LIST="./tf_list_human.txt"

# ---------- Experiment settings (must be identical to the pw18 run) ----------
PATHWAYS="P01 P02 P03 P04 P05 P06 P07 P08 P09 P10 P11 P12 P13 P14 P15 P16 P17 P18"
N_PW=18
N_CELLS=20
N_TARGET=200
SEED=42
DEVICE="cuda"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Completion is judged by the number of pathways. Checking only whether the file
# exists would take a run that crashed midway for complete and skip it silently.
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
    local K=$1 B=$2 P=$3
    local OUT="./greedy_pool18set_gf_k${K}"
    if done_already "$OUT/search_comparison.json"; then
        echo "=== Geneformer k=${K}: all 18 pathways present, skipping ($OUT) ==="
        return 0
    fi
    mkdir -p "$OUT"
    echo "=== Geneformer k=${K} budget=${B} pool=${P} pathways=${N_PW}  $(date '+%F %T') ==="
    CUDA_VISIBLE_DEVICES=1 python compare_search_algorithms.py \
        --model_dir        "$GF_MODEL" \
        --pathway_config   "$PATHWAY_CONFIG" \
        --dict_dir         "$GF_DICT" \
        --tf_list          "$TF_LIST" \
        --pathways         $PATHWAYS \
        --n_cells          "$N_CELLS" \
        --n_target_cells   "$N_TARGET" \
        --k_adv            "$K" \
        --budget           "$B" \
        --emb_layer        0 \
        --greedy \
        --greedy_pool      "$P" \
        --only_greedy \
        --order_sets       0 \
        --seed             "$SEED" \
        --device           "$DEVICE" \
        --output_dir       "$OUT" \
        2>&1 | tee -a "$OUT/log.txt"
}

run_scgpt () {
    local K=$1 B=$2 P=$3
    local OUT="./greedy_pool18_scgpt_k${K}"
    if done_already "$OUT/search_comparison.json"; then
        echo "=== scGPT k=${K}: all 18 pathways present, skipping ($OUT) ==="
        return 0
    fi
    mkdir -p "$OUT"
    echo "=== scGPT k=${K} budget=${B} pool=${P} pathways=${N_PW}  $(date '+%F %T') ==="
    CUDA_VISIBLE_DEVICES=1 python compare_search_scgpt.py \
        --model_dir        "$SCGPT_MODEL" \
        --pathway_config   "$PATHWAY_CONFIG" \
        --tf_list          "$TF_LIST" \
        --pathways         $PATHWAYS \
        --n_cells          "$N_CELLS" \
        --n_target_cells   "$N_TARGET" \
        --max_seq_len      1200 \
        --binning_mode     per_cell \
        --high_value       50 \
        --k_adv            "$K" \
        --budget           "$B" \
        --greedy \
        --greedy_pool      "$P" \
        --only_greedy \
        --order_sets       0 \
        --seed             "$SEED" \
        --device           "$DEVICE" \
        --output_dir       "$OUT" \
        2>&1 | tee -a "$OUT/log.txt"
}

WHICH="${1:-both}"

for CFG in "2 1290 645" "4 2580 645" "8 2580 322"; do
    set -- $CFG
    K=$1; B=$2; P=$3
    if [ "$WHICH" = "both" ] || [ "$WHICH" = "gf" ]; then
        run_gf "$K" "$B" "$P" || echo "[WARN] Geneformer k=${K} failed (continuing)"
    fi
    if [ "$WHICH" = "both" ] || [ "$WHICH" = "scgpt" ]; then
        run_scgpt "$K" "$B" "$P" || echo "[WARN] scGPT k=${K} failed (continuing)"
    fi
done

cat <<'EOF'

============================================================
  Done. Command to aggregate the main table
============================================================
python make_table3.py \
    --gf_attack  ./pw18_gf_k{2,4,8}/search_comparison.json \
    --gf_greedy  ./greedy_pool18_gf_k{2,4,8}/search_comparison.json \
    --sc_attack  ./pw18_scgpt_k{2,4,8}/search_comparison.json \
    --sc_greedy  ./greedy_pool18_scgpt_k{2,4,8}/search_comparison.json
EOF
