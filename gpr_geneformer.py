#!/usr/bin/env python3
"""
GPR (Gradient Projection Ranking) for Geneformer V1
=====================================================
Predict direct reprogramming TFs by gradient projection ranking in the word embedding space of
Geneformer V1 (10M).

Design:
  - insert k TF tokens at the front of the token sequence of a source cell
  - set requires_grad on the word embedding at the adversarial positions
  - project the gradient g_i onto the word embedding E[j] of every TF to obtain the score
  - output the mean over n_init random initializations (default 5)
  - same function signatures and CLI argument scheme as the scGPT version (gpr_scgpt.py)

Usage:
    python gpr_geneformer.py \
        --model_dir ./Geneformer/Geneformer-V1-10M \
        --dataset_path ./geneformer_cardiac/tokenized_v1/cardiac_v1.dataset \
        --dict_dir ./Geneformer/geneformer/gene_dictionaries_30m \
        --tf_list ./tf_list_human.txt \
        --pathway iCM \
        --k_adv 4 --n_cells 50 --n_init 5
"""

import argparse
import json
import os
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import BertModel

from advcell_aggregate import (
    GammaAccumulator,
    geometry_diagnostics,
    format_geometry,
)
from advcell_overexpress import (
    build_inputs_embeds,
    masked_mean_pool,
    overexpress_batch,
    overexpress_single,
    set_overexpress_mode,
)

warnings.filterwarnings("ignore")

# ============================================================
# Definition of the validation pathways (shared with scGPT/scFoundation)
# ============================================================
PATHWAYS = {
    "iCM": {
        "source_type": "cardiac_fibroblast",
        "target_type": "regular atrial cardiac myocyte",
        "gt_tfs": ["GATA4", "MEF2C", "TBX5", "HAND2", "MYOCD", "NKX2-5"],
    },
    "iHep": {
        "source_type": "cardiac_fibroblast",
        "target_type": "hepatocyte",
        "gt_tfs": ["HNF4A", "FOXA1", "FOXA2", "FOXA3", "HNF1A", "GATA4", "CEBPA"],
    },
    "B_to_Mac": {
        "source_type": "B cell",
        "target_type": "macrophage",
        "gt_tfs": ["CEBPA", "CEBPB", "SPI1"],
    },
    "Muscle": {
        "source_type": "cardiac_fibroblast",
        "target_type": "skeletal muscle satellite stem cell",
        "gt_tfs": ["MYOD1", "MYOG", "MYF5", "MYF6", "PAX7", "MEF2C"],
    },
}

# Geneformer V1 constants
MAX_SEQ_LEN = 2048
PAD_TOKEN_ID = 0
SPECIAL_TOKEN_IDS = {0, 1}  # <pad>, <mask>


# ============================================================
# Dictionary loading
# ============================================================
def load_dicts(dict_dir: str):
    """Load the Geneformer V1 dictionaries.

    Returns:
        token_dict: ensembl_id -> token_id
        name_dict:  gene_symbol -> ensembl_id
        id_to_sym:  token_id -> gene_symbol
        sym_to_id:  gene_symbol -> token_id
    """
    dict_dir = Path(dict_dir)

    # prefer the gc30M version, else the standard name
    def _find(base_name):
        gc30m = base_name.replace(".pkl", "_gc30M.pkl")
        for candidate in [gc30m, base_name]:
            p = dict_dir / candidate
            if p.exists():
                return p
        raise FileNotFoundError(
            f"Cannot find {base_name} or {gc30m} in {dict_dir}"
        )

    with open(_find("token_dictionary.pkl"), "rb") as f:
        token_dict = pickle.load(f)
    with open(_find("gene_name_id_dict.pkl"), "rb") as f:
        name_dict = pickle.load(f)

    ens_to_sym = {v: k for k, v in name_dict.items()}
    id_to_sym = {tid: ens_to_sym.get(ens, ens) for ens, tid in token_dict.items()}
    sym_to_id = {sym: tid for tid, sym in id_to_sym.items()}

    return token_dict, name_dict, id_to_sym, sym_to_id


def load_tf_list(tf_list_path: str, token_dict, name_dict):
    """Read the TF list and return a list of (symbol, token_id).

    Returns:
        tf_token_ids: list of (symbol, token_id)  -- same format as the scGPT version
    """
    tf_symbols = [
        line.strip()
        for line in Path(tf_list_path).read_text().strip().split("\n")
        if line.strip()
    ]
    sym_to_ens = name_dict
    tf_token_ids = []
    seen_ids = set()
    for sym in tf_symbols:
        ens = sym_to_ens.get(sym)
        tid = token_dict.get(ens) if ens else None
        if tid is not None and tid not in SPECIAL_TOKEN_IDS and tid not in seen_ids:
            tf_token_ids.append((sym, tid))
            seen_ids.add(tid)
    return tf_token_ids


# ============================================================
# Model loading
# ============================================================
def load_geneformer_model(model_dir: str, device):
    """Load Geneformer V1 (BertModel).

    Returns:
        model, model_config
    """
    model = BertModel.from_pretrained(model_dir)
    model.eval().to(device)
    # stop gradient accumulation on the parameters.
    # adv_emb is set with requires_grad_(True) as a leaf tensor, so backward still works.
    model.requires_grad_(False)

    config = model.config
    model_config = {
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size,
    }
    print(
        f"Loaded Geneformer V1: vocab={config.vocab_size}, "
        f"d={config.hidden_size}, layers={config.num_hidden_layers}"
    )
    return model, model_config


# ============================================================
# Embedding extraction (layer selection managed in one place)
# ============================================================
# Layer specification following Geneformer convention:
#   0  : last layer (last_hidden_state)
#   -1 : second to last layer. The default of the official EmbExtractor and
#        the layer the Geneformer paper used for cell classification.
#
# If the layer differs between the centroid and the perturbed embedding the comparison breaks,
# so it is a global setting and all forward passes go through the same path.
_EMB_LAYER = 0


def set_emb_layer(layer: int) -> None:
    """Set the layer from which embeddings are taken (meant to be called once from main)."""
    global _EMB_LAYER
    if layer not in (0, -1):
        raise ValueError(f"emb_layer must be 0 or -1, got {layer}")
    _EMB_LAYER = layer


def get_emb_layer() -> int:
    return _EMB_LAYER


def forward_cell_embedding(model, attn, *, inputs_embeds=None, input_ids=None):
    """Run forward and return the cell embedding mean-pooled from the configured layer.

    Args:
        attn: (B, T) attention mask. Used to exclude padding.
        inputs_embeds / input_ids: pass one or the other.
                                   Use inputs_embeds when the gradient is needed.

    Returns:
        (B, D) tensor
    """
    need_hidden = _EMB_LAYER != 0
    out = model(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        attention_mask=attn,
        output_hidden_states=need_hidden,
    )
    if need_hidden:
        # hidden_states = (embedding output, layer1, ..., layerN), of length N+1,
        # so [-1] is the last layer and [-2] the second to last.
        h = out.hidden_states[_EMB_LAYER - 1]
    else:
        h = out.last_hidden_state
    return masked_mean_pool(h, attn)


# ============================================================
# Data preparation (Geneformer tokenized dataset)
# ============================================================
def prepare_cell_data(dataset, cell_type: str, n_cells: int = 50,
                      seed: int = 42):
    """Extract the input_ids of the specified cell type from the tokenized dataset.

    Returns:
        cells: list of dict with keys {"input_ids": list[int]}
    """
    indices = [
        i for i, ct in enumerate(dataset["cell_type"]) if ct == cell_type or (isinstance(cell_type, list) and ct in cell_type)
    ]
    if len(indices) == 0:
        # try partial match
        indices = [
            i for i, ct in enumerate(dataset["cell_type"])
            if cell_type.lower() in ct.lower()
        ]
    if len(indices) == 0:
        from collections import Counter
        counts = Counter(dataset["cell_type"])
        top = counts.most_common(20)
        raise ValueError(
            f"Cell type '{cell_type}' not found.\n"
            f"Available (top 20): {top}"
        )

    print(f"  Found {len(indices)} cells for '{cell_type}'")

    if len(indices) > n_cells:
        rng = np.random.RandomState(seed)
        indices = rng.choice(indices, n_cells, replace=False).tolist()

    cells = []
    for idx in indices:
        ids = dataset[idx]["input_ids"]
        # remove padding
        ids = [t for t in ids if t != PAD_TOKEN_ID]
        cells.append({"input_ids": ids})

    if cells:
        avg_len = np.mean([len(c["input_ids"]) for c in cells])
        print(f"  {len(cells)} cells prepared, avg {avg_len:.0f} tokens/cell")

    return cells


# ============================================================
# Target centroid
# ============================================================
@torch.no_grad()
def compute_target_centroid(model, dataset, cell_type: str,
                            n_cells: int, device,
                            max_seq_len: int = MAX_SEQ_LEN):
    """Compute the cell-embedding centroid of the target cell group."""
    indices = [
        i for i, ct in enumerate(dataset["cell_type"]) if ct == cell_type or (isinstance(cell_type, list) and ct in cell_type)
    ]
    if len(indices) > n_cells:
        rng = np.random.RandomState(42)
        indices = rng.choice(indices, n_cells, replace=False).tolist()

    print(f"  Computing centroid for '{cell_type}' ({len(indices)} cells)...")
    embeddings = []
    for idx in indices:
        ids = torch.tensor(dataset[idx]["input_ids"], device=device).unsqueeze(0)
        ids = ids[:, :max_seq_len]
        mask = (ids != PAD_TOKEN_ID).long()
        cell_emb = forward_cell_embedding(model, mask, input_ids=ids).squeeze(0)
        embeddings.append(cell_emb)

    centroid = torch.stack(embeddings).mean(dim=0)
    centroid = F.normalize(centroid, dim=0)
    print(f"  centroid norm (after normalize) = {centroid.norm():.4f}")
    return centroid


# ============================================================
# Loss functions
# ============================================================
def compute_loss(cell_emb, target_centroid, loss_mode="angular"):
    """Compute the loss.

    Args:
        loss_mode:
          "angular":  arccos(cos_sim) -- amplifies the gradient near cos_sim~1 (recommended)
          "cosine":   -cos_sim        -- conventional

    Returns:
        loss: scalar tensor (can be backpropagated)
        cos_sim_val: float (for logging and comparison)
    """
    cos_sim = F.cosine_similarity(
        cell_emb.unsqueeze(0), target_centroid.unsqueeze(0)
    )
    cos_sim_val = cos_sim.item()

    if loss_mode == "angular":
        cos_clamped = cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)
        loss = torch.acos(cos_clamped)
    elif loss_mode == "cosine":
        loss = -cos_sim
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    return loss, cos_sim_val


# ============================================================
# GPR scoring (core: 1 cell x 1 initialization)
# ============================================================
def gpr_score_single_cell(
    model, cell, target_centroid, tf_token_ids,
    device, k_adv=4, max_seq_len=MAX_SEQ_LEN,
    rng=None, loss_mode="angular", stats=None,
):
    """Compute the GPR score for one cell (Geneformer version).

    Geneformer overexpress scheme (following the official InSilicoPerturber):
      if the target TF is already in the rank value encoding, remove it from its original position
      and then insert it at the front (pop-and-insert).
      Merely prepending makes the same token appear twice, an input outside the pretraining distribution.

    Args:
        cell: dict with "input_ids"
        target_centroid: (D,) tensor
        tf_token_ids: list of (symbol, token_id)
        rng: numpy random generator
        loss_mode: kept for compatibility, but the gradient is always computed on the cosine basis.
                   It has been confirmed that angular is only (1-c^2)^(-1/2) times cosine,
                   and the weighting is done on the aggregation side (GammaAccumulator).
        stats: optional dict. If given, "n_removed" (the number of inserted TFs
               that were originally expressed) is written to it.

    Returns:
        scores: dict {symbol: float}  -- gradient-projection score on the cosine basis
        cos_sim: float                -- cos_sim at the point where the gradient was taken
    """
    if rng is None:
        rng = np.random.default_rng()

    E = model.embeddings.word_embeddings.weight  # (V, D)
    n_tf = len(tf_token_ids)

    # choose k random TFs
    adv_indices = rng.choice(n_tf, size=k_adv, replace=False)
    adv_ids = torch.tensor(
        [tf_token_ids[i][1] for i in adv_indices],
        dtype=torch.long, device=device,
    )

    # pop-and-insert following the official implementation (truncation is done on the overexpress side)
    orig_ids_t = torch.tensor(
        cell["input_ids"][:max_seq_len], dtype=torch.long, device=device,
    )
    ids, attn, n_removed = overexpress_single(orig_ids_t, adv_ids, max_seq_len)
    if stats is not None:
        stats["n_removed"] = int(n_removed[0].item())

    # embedding at the adversarial positions: requires_grad=True
    adv_emb = E[adv_ids].detach().clone().requires_grad_(True)  # (k, D)

    # [adv_tokens || remaining token sequence] (padding excluded by attn)
    combined = build_inputs_embeds(E, ids, k_adv, adv_emb=adv_emb)

    # forward
    cell_emb = forward_cell_embedding(
        model, attn, inputs_embeds=combined
    ).squeeze(0)  # (D,)

    # loss computation
    # Always the cosine basis. angular is only a positive scalar multiple, and it has been confirmed
    # that it does not change the ranking per (cell, init).
    loss, cos_sim_val = compute_loss(
        cell_emb, target_centroid, loss_mode="cosine"
    )
    loss.backward()

    # project the gradient onto the word embedding of all TFs
    g = adv_emb.grad  # (k, D)
    if g is None:
        raise RuntimeError("No gradient on adv_emb. Check computation graph.")

    all_tf_ids = torch.tensor(
        [tid for _, tid in tf_token_ids], dtype=torch.long, device=device,
    )
    with torch.no_grad():
        tf_embs = E[all_tf_ids]  # (n_tf, D)
        # score[j] = mean_i(-g_i · E[j])
        neg_g = -g  # (k, D)
        proj = neg_g @ tf_embs.T  # (k, n_tf)
        scores_tensor = proj.mean(dim=0)  # (n_tf,)

    scores = {
        name: float(scores_tensor[i].item())
        for i, (name, _) in enumerate(tf_token_ids)
    }
    return scores, cos_sim_val


# ============================================================
# GPR run (all cells x n_init)
# ============================================================
def run_gpr(model, source_cells, target_centroid, tf_token_ids,
            device, k_adv=4, n_init=5, max_seq_len=MAX_SEQ_LEN,
            loss_mode="angular", stats_out=None, gammas=(0.0, 1.0)):
    """Run GPR on all source cells.

    The gradient is computed once on the cosine basis, and the gamma-weighted means are reconstructed
    from the same trajectory. There is no need to rerun with a different loss.

    Args:
        loss_mode: weight used for the main result. angular -> gamma=1, cosine -> gamma=0.
        gammas:    list of additional gammas to aggregate.
        stats_out: optional list. Appends n_removed per (cell, init).

    Returns:
        mean_scores:     dict  -- mean over all cells at the gamma corresponding to loss_mode
        all_cell_scores: list  -- per-cell means of the same
        gamma_results:   dict  — {"0": {...}, "1": {...}, ...}
        cos_stats:       dict  -- distribution of cos_sim and lambda
    """
    tf_names = [n for n, _ in tf_token_ids]
    gset = sorted({float(g) for g in gammas} | {0.0, 1.0})
    acc = GammaAccumulator(tf_names, gset)

    for cell_idx, cell in enumerate(tqdm(source_cells, desc="GPR scoring")):
        acc.start_cell()
        for init_i in range(n_init):
            rng = np.random.default_rng(seed=cell_idx * 1000 + init_i)
            st = {} if stats_out is not None else None
            scores, cos_sim_val = gpr_score_single_cell(
                model, cell, target_centroid, tf_token_ids,
                device, k_adv=k_adv, max_seq_len=max_seq_len,
                rng=rng, loss_mode=loss_mode, stats=st,
            )
            if stats_out is not None:
                stats_out.append(st["n_removed"])
            acc.add(scores, cos_sim_val)
        acc.end_cell()

    gamma_results = acc.finalize()
    primary = "1" if loss_mode == "angular" else "0"
    mean_scores = gamma_results[primary]["mean"]
    all_cell_scores = gamma_results[primary]["per_cell"]

    return mean_scores, all_cell_scores, gamma_results, acc.cos_sim_stats()


# ============================================================
# Evaluation (same signature as the scGPT version)
# ============================================================
def evaluate_auc(scores, gt_tfs, tf_names):
    """Compute the ROC-AUC and the rank of each GT TF."""
    gt_set = set(gt_tfs)
    score_values = np.array([scores.get(n, 0.0) for n in tf_names])
    labels = np.array([1 if n in gt_set else 0 for n in tf_names])

    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan"), {}

    auc = roc_auc_score(labels, score_values)

    ranked_idx = np.argsort(score_values)[::-1]
    rank_map = {tf_names[idx]: rank + 1 for rank, idx in enumerate(ranked_idx)}
    per_tf_ranks = {tf: rank_map.get(tf, -1) for tf in gt_tfs}

    return auc, per_tf_ranks


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="GPR for Geneformer V1")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Geneformer V1 checkpoint directory")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Tokenized dataset path (load_from_disk)")
    parser.add_argument("--dict_dir", type=str, required=True,
                        help="Gene dictionaries directory (gc30M)")
    parser.add_argument("--tf_list", type=str, required=True,
                        help="TF list file (one symbol per line)")
    parser.add_argument("--pathway", type=str, default="iCM",
                        choices=list(PATHWAYS.keys()))
    parser.add_argument("--k_adv", type=int, default=4)
    parser.add_argument("--n_cells", type=int, default=50)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--n_target_cells", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--output_dir", type=str,
                        default="./gpr_geneformer_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss_mode", type=str, default="angular",
                        choices=["angular", "cosine"],
                        help="Loss function: angular (recommended) or cosine")
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5, 2.0],
                        help="gamma of the aggregation weight w=(1-c^2)^(-gamma/2). "
                             "gamma=0 is cosine, gamma=1 is angular.")
    parser.add_argument("--emb_layer", type=int, default=0, choices=[0, -1],
                        help="layer from which embeddings are taken. 0=last layer (current), "
                             "-1=second to last (default of the official EmbExtractor, follows the paper). "
                             "Aligned automatically between the centroid and the perturbed embedding.")
    parser.add_argument("--overexpress_mode", type=str, default="pop_and_insert",
                        choices=["pop_and_insert", "prepend_only"],
                        help="pop_and_insert: follows the official ISP (default). "
                             "prepend_only: prepend without removal (duplicate tokens arise). "
                             "For A/B comparison.")
    args = parser.parse_args()

    set_overexpress_mode(args.overexpress_mode)
    set_emb_layer(args.emb_layer)
    print(f"[config] emb_layer={args.emb_layer} "
          f"({'last' if args.emb_layer == 0 else 'second-to-last'} layer), "
          f"overexpress_mode={args.overexpress_mode}")
    if args.overexpress_mode != "pop_and_insert":
        print(f"[WARN] overexpress_mode={args.overexpress_mode} "
              f"(does not follow the official ISP. A setting for A/B comparison)")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load the model
    print("Loading Geneformer V1 model...")
    model, model_config = load_geneformer_model(args.model_dir, device)

    # dataset
    print(f"Loading dataset: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    # dictionary
    print(f"Loading dictionaries: {args.dict_dir}")
    token_dict, name_dict, id_to_sym, sym_to_id = load_dicts(args.dict_dir)

    # TF list
    print("Loading TF list...")
    tf_token_ids = load_tf_list(args.tf_list, token_dict, name_dict)
    print(f"  {len(tf_token_ids)} TFs mapped to vocabulary")

    # pathway information
    pw = PATHWAYS[args.pathway]
    print(f"\nPathway: {args.pathway}")
    print(f"  {pw['source_type']} -> {pw['target_type']}")
    print(f"  GT TFs: {pw['gt_tfs']}")

    # check GT TF coverage
    tf_id_set = {tid for _, tid in tf_token_ids}
    gt_in_vocab = [
        tf for tf in pw["gt_tfs"]
        if tf in sym_to_id and sym_to_id[tf] in tf_id_set
    ]
    gt_missing = [tf for tf in pw["gt_tfs"] if tf not in gt_in_vocab]
    if gt_missing:
        print(f"  WARNING: GT TFs not in TF list/vocab: {gt_missing}")
    print(f"  GT TFs in TF list: {len(gt_in_vocab)}/{len(pw['gt_tfs'])}")

    # prepare source cells
    print(f"\nPreparing source cells...")
    source_cells = prepare_cell_data(
        dataset, pw["source_type"],
        n_cells=args.n_cells, seed=args.seed,
    )

    # compute the centroid
    print("Computing target centroid...")
    target_centroid = compute_target_centroid(
        model, dataset, pw["target_type"],
        n_cells=args.n_target_cells, device=device,
        max_seq_len=args.max_seq_len,
    )

    # run GPR
    print(f"\nRunning GPR (k={args.k_adv}, n_init={args.n_init}, loss={args.loss_mode})...")
    overexpress_stats = []
    mean_scores, all_cell_scores, gamma_results, cos_stats = run_gpr(
        model, source_cells, target_centroid, tf_token_ids,
        device, k_adv=args.k_adv, n_init=args.n_init,
        max_seq_len=args.max_seq_len, loss_mode=args.loss_mode,
        stats_out=overexpress_stats, gammas=args.gammas,
    )

    # evaluation
    tf_names = [name for name, _ in tf_token_ids]
    auc, per_tf_ranks = evaluate_auc(mean_scores, pw["gt_tfs"], tf_names)

    # display
    print(f"\n{'='*60}")
    print(f"Results: {args.pathway} (Geneformer V1 GPR)")
    print(f"{'='*60}")
    print(f"ROC-AUC (strict): {auc:.3f}")
    gamma_auc = {
        g: float(evaluate_auc(v["mean"], pw["gt_tfs"], tf_names)[0])
        for g, v in gamma_results.items()
    }
    gs = sorted(gamma_auc, key=float)
    print("γ sweep  w=(1-c^2)^(-γ/2)  [γ=0:cosine, γ=1:angular]")
    print("  " + "".join(f"{'γ='+g:>10s}" for g in gs))
    print("  " + "".join(f"{gamma_auc[g]:10.4f}" for g in gs))
    spread = max(gamma_auc.values()) - min(gamma_auc.values())
    print(f"  spread across gamma = {spread:.4f}"
          + ("  -> the choice of gamma has almost no effect on the result" if spread < 0.005 else ""))
    if cos_stats:
        print(f"  cos_sim {cos_stats['cos_sim_min']:.4f}-"
              f"{cos_stats['cos_sim_max']:.4f}, "
              f"lambda ratio {cos_stats.get('lambda_ratio') or float('nan'):.2f}x")
    if overexpress_stats:
        n_rem = np.array(overexpress_stats, dtype=float)
        print(f"overexpress n_removed: mean={n_rem.mean():.3f}/{args.k_adv} "
              f"(fraction of inserted TFs that were already in the original sequence={n_rem.mean()/args.k_adv:.1%})")
    print(f"\nGT TF ranks (/{len(tf_token_ids)} TFs):")
    for tf, rank in sorted(per_tf_ranks.items(), key=lambda x: x[1]):
        mark = " ***" if rank <= 100 else ""
        print(f"  {tf:12s}: {rank:5d}{mark}")

    # per-cell AUC
    per_cell_aucs = []
    for cs in all_cell_scores:
        a, _ = evaluate_auc(cs, pw["gt_tfs"], tf_names)
        per_cell_aucs.append(a)
    valid_aucs = [a for a in per_cell_aucs if not np.isnan(a)]
    if valid_aucs:
        print(f"\nPer-cell AUC: {np.mean(valid_aucs):.3f} +/- {np.std(valid_aucs):.3f}")

    # Top-20
    ranked = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop-20 ranked TFs:")
    gt_set = set(pw["gt_tfs"])
    for i, (name, score) in enumerate(ranked[:20]):
        mark = " <-- GT" if name in gt_set else ""
        print(f"  {i+1:3d}. {name:12s}: {score:+.8f}{mark}")

    # save the results
    result = {
        "pathway": args.pathway,
        "model": "Geneformer_V1_10M",
        "method": "GPR",
        "k_adv": args.k_adv,
        "n_cells": len(source_cells),
        "n_init": args.n_init,
        "n_target_cells": args.n_target_cells,
        "max_seq_len": args.max_seq_len,
        "n_tfs": len(tf_token_ids),
        "auc_strict": float(auc),
        "per_cell_auc_mean": float(np.mean(valid_aucs)) if valid_aucs else None,
        "per_cell_auc_std": float(np.std(valid_aucs)) if valid_aucs else None,
        "gt_tf_ranks": per_tf_ranks,
        "top50_tfs": [(n, float(s)) for n, s in ranked[:50]],
        "gt_tfs": pw["gt_tfs"],
        "overexpress_mode": args.overexpress_mode,
        "emb_layer": args.emb_layer,
        "gammas": args.gammas,
        "gamma_auc": gamma_auc,
        "cos_stats": cos_stats,
        "n_removed_mean": (float(np.mean(overexpress_stats))
                           if overexpress_stats else None),
    }

    outfile = os.path.join(
        args.output_dir,
        f"gpr_geneformer_{args.pathway}_k{args.k_adv}_n{len(source_cells)}.json",
    )
    with open(outfile, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
