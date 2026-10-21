#!/usr/bin/env python3
"""
GPR (Gradient Projection Ranking) for scGPT
============================================
Predict direct reprogramming TFs by gradient projection ranking in the embedding space of
the scGPT foundation model.

Design:
  - set requires_grad on gene_emb at the adversarial positions
  - value_emb is fixed (high-expression bin or high value)
  - project the gradient g_i onto the gene embedding E[j] of every TF to obtain the score
  - biological interpretation: "given that it is highly expressed, which gene identity should be put in"
  - hybrid scheme: existing genes get their expression value changed, absent ones are added


Usage:
    python gpr_scgpt.py \
        --model_dir ./scGPT_human \
        --h5ad_path ./geneformer_cardiac/h5ad/cardiac_subset.h5ad \
        --tf_list ./tf_list_human.txt \
        --pathway iCM \
        --k_adv 4 --n_cells 50 --n_init 5
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Direct import of scGPT (avoids the torchtext compatibility problem)
from scgpt.model.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils.util import load_pretrained

warnings.filterwarnings("ignore")

# ============================================================
# Definition of the validation pathways
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


# ============================================================
# Model loading
# ============================================================
def load_scgpt_model(model_dir, device, use_continuous_value=False):
    """Load the scGPT pretrained model.

    Args:
        model_dir: directory of the scGPT_human checkpoint
        device: torch.device
        use_continuous_value: if True, use ContinuousValueEncoder.
            If False, use the pretrained binning=51 Embedding (default).
            Note: the gradient target of GPR is gene_emb, so either works.

    Returns:
        model, vocab, model_config
    """
    model_dir = Path(model_dir)

    # load vocab.json
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    pad_token = "<pad>"
    pad_value = -2

    # read the model settings from args.json
    args_file = model_dir / "args.json"
    model_config = {
        "embsize": 512, "nhead": 8, "d_hid": 512,
        "nlayers": 12, "n_input_bins": 51,
    }
    if args_file.exists():
        with open(args_file) as f:
            saved_args = json.load(f)
        for k in model_config:
            if k in saved_args:
                model_config[k] = saved_args[k]

    ntokens = len(vocab)
    n_input_bins = 0 if use_continuous_value else model_config["n_input_bins"]

    model = TransformerModel(
        ntokens,
        model_config["embsize"],
        model_config["nhead"],
        model_config["d_hid"],
        model_config["nlayers"],
        vocab=vocab,
        dropout=0.0,
        pad_token=pad_token,
        pad_value=pad_value,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        n_input_bins=n_input_bins,
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=False,
    )

    # load the weights
    model_file = model_dir / "best_model.pt"
    if not model_file.exists():
        pt_files = list(model_dir.glob("*.pt"))
        model_file = pt_files[0] if pt_files else None
        if model_file is None:
            raise FileNotFoundError(f"No .pt file found in {model_dir}")

    pretrained_dict = torch.load(model_file, map_location=device)
    load_pretrained(model, pretrained_dict)

    model.to(device)
    model.eval()

    print(f"Loaded scGPT: {ntokens} tokens, {model_config['embsize']}d, "
          f"{model_config['nlayers']}L, n_input_bins={n_input_bins}")
    model_config["n_input_bins"] = n_input_bins
    return model, vocab, model_config


# ============================================================
# _digitize compatible with the official scGPT
# ============================================================
def _digitize_scgpt(x, bins, seed=None):
    """Same logic as the official scGPT _digitize (preprocess.py L239-271).

    When there are duplicate bin edges, assign at random between left_digits and right_digits,
    which prevents identical values from concentrating in one bin.

    Args:
        x: 1-d array of values (non-zero only)
        bins: 1-d array of bin edges (sorted)
        seed: int or None. The seed can be fixed for reproducibility.
              If None, non-deterministic as in the official implementation.
    """
    assert x.ndim == 1 and bins.ndim == 1
    left_digits = np.digitize(x, bins)
    right_digits = np.digitize(x, bins, right=True)

    rng = np.random.RandomState(seed)
    rands = rng.rand(len(x))

    digits = rands * (right_digits - left_digits) + left_digits
    digits = np.ceil(digits).astype(np.int64)
    return digits


# ============================================================
# Data preparation
# ============================================================
def prepare_cell_data(adata, cell_type, vocab, n_cells=50, seed=42,
                      n_input_bins=51, binning_mode="global"):
    """Prepare the data of the specified cell type.

    Args:
        binning_mode:
            "global"   - global quantile binning from the nonzero values of all cells
                         (a GPR-specific method absent from the official scGPT.
                          Provides a deterministic scale consistent across cells.)
            "per_cell" - follows the official scGPT Preprocessor. Quantile binning from
                         the nonzero values of each cell + _digitize randomization
            "none"     - no binning; use the raw log1p values as they are

    Note:
        In all binning modes, zero expression -> bin 0 is preserved (official specification:
        "the first bin is always special for zero").
        Only nonzero values are assigned to 1 to n_input_bins-1 (= 1-50).

    Returns:
        cells: list of dict
    """
    # filter by cell type (string or list of strings)
    if isinstance(cell_type, list):
        mask = adata.obs["cell_type"].isin(cell_type)
    else:
        mask = adata.obs["cell_type"] == cell_type
    if mask.sum() == 0 and isinstance(cell_type, str):
        mask = adata.obs["cell_type"].str.contains(cell_type, case=False, na=False)
    if mask.sum() == 0:
        available = adata.obs["cell_type"].value_counts().head(20)
        raise ValueError(f"Cell type '{cell_type}' not found.\n"
                         f"Available (top 20):\n{available}")

    adata_sub = adata[mask].copy()
    print(f"  Found {adata_sub.n_obs} cells for '{cell_type}'")

    if adata_sub.n_obs > n_cells:
        np.random.seed(seed)
        idx = np.random.choice(adata_sub.n_obs, n_cells, replace=False)
        adata_sub = adata_sub[idx].copy()

    # normalization
    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)

    vocab_stoi = vocab.get_stoi()
    if "feature_name" in adata_sub.var.columns:
        gene_names_all = adata_sub.var["feature_name"].tolist()
    elif "gene_name" in adata_sub.var.columns:
        gene_names_all = adata_sub.var["gene_name"].tolist()
    else:
        gene_names_all = adata_sub.var_names.tolist()

    X = adata_sub.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # For global binning: compute the edges from the nonzero values of all cells
    # NOTE: the official scGPT does per-cell binning only. global is a GPR-specific method that
    #       provides a deterministic and consistent scale by using common bin edges for all cells.
    #       The number of edges is n_bins-1 (= 50) quantile points, as in the official implementation.
    #       digitize also uses _digitize_scgpt, as in the official implementation.
    global_bin_edges = None
    if binning_mode == "global" and n_input_bins > 0:
        all_nonzero = X[X > 0].flatten()
        if len(all_nonzero) > 0:
            global_bin_edges = np.quantile(
                all_nonzero, np.linspace(0, 1, n_input_bins - 1)
            )

    cells = []
    for i in range(X.shape[0]):
        expr = X[i]
        nonzero_idx = np.where(expr > 0)[0]
        cell_genes = []
        cell_values_raw = []
        cell_names = []

        for j in nonzero_idx:
            gname = gene_names_all[j]
            if gname in vocab_stoi:
                cell_genes.append(vocab_stoi[gname])
                cell_values_raw.append(float(expr[j]))
                cell_names.append(gname)

        if len(cell_genes) == 0:
            continue

        # sort by descending expression
        sort_idx = np.argsort(cell_values_raw)[::-1]
        cell_genes = [cell_genes[s] for s in sort_idx]
        cell_values_raw = [cell_values_raw[s] for s in sort_idx]
        cell_names = [cell_names[s] for s in sort_idx]

        # value conversion
        # NOTE: at this point cell_values_raw holds only nonzero values (already extracted by nonzero_idx above).
        #       Hence all elements are > 0 and the binning results are all in the range 1-50.
        #       Zero-expression genes are not included at all, so the bin 0 problem does not arise.
        if binning_mode == "none" or n_input_bins <= 0:
            cell_values = list(cell_values_raw)
        elif binning_mode == "global":
            raw_arr = np.array(cell_values_raw)
            # use _digitize_scgpt as in the official implementation (fix the seed for reproducibility)
            binned = _digitize_scgpt(raw_arr, global_bin_edges, seed=seed + i)
            binned = np.clip(binned, 1, n_input_bins - 1)
            cell_values = binned.tolist()
        elif binning_mode == "per_cell":
            # follows the official scGPT Preprocessor:
            #   quantile edges from nonzero values only -> _digitize (randomization)
            # cell_values_raw is already nonzero only, so use it as it is
            raw_arr = np.array(cell_values_raw)
            bins = np.quantile(raw_arr, np.linspace(0, 1, n_input_bins - 1))
            binned = _digitize_scgpt(raw_arr, bins, seed=seed + i)
            binned = np.clip(binned, 1, n_input_bins - 1)
            cell_values = binned.tolist()
        else:
            cell_values = list(cell_values_raw)

        cells.append({
            "gene_ids": cell_genes,
            "gene_names": cell_names,
            "values": cell_values,
            "raw_values": cell_values_raw,
        })

    if cells:
        avg_genes = np.mean([len(c["gene_ids"]) for c in cells])
        v_sample = cells[0]["values"][:3]
        print(f"  {len(cells)} cells prepared, avg {avg_genes:.0f} genes/cell "
              f"(binning={binning_mode}, sample values: {v_sample})")
    else:
        sample_names = gene_names_all[:5]
        sample_vocab = list(vocab_stoi.keys())[:5]
        print(f"  WARNING: 0 cells matched vocab!")
        print(f"    gene_names sample: {sample_names}")
        print(f"    vocab sample: {sample_vocab}")

    return cells


# ============================================================
# Target centroid
# ============================================================
def compute_target_centroid(model, cells, vocab, device,
                            max_seq_len=1200, n_input_bins=51):
    """Compute the cell-embedding centroid of the target cell group."""
    pad_token_id = vocab["<pad>"]
    pad_value = -2 if n_input_bins > 0 else 0.0

    embeddings = []
    with torch.no_grad():
        for cell in tqdm(cells, desc="Computing target centroid"):
            gene_ids = cell["gene_ids"][:max_seq_len]
            values = cell["values"][:max_seq_len]
            seq_len = len(gene_ids)

            n_pad = max(0, max_seq_len - seq_len)
            gene_ids_padded = gene_ids + [pad_token_id] * n_pad
            if n_input_bins > 0:
                values_padded = values + [0] * n_pad  # bin 0 for padding
            else:
                values_padded = values + [pad_value] * n_pad

            src = torch.tensor([gene_ids_padded], dtype=torch.long, device=device)
            val = torch.tensor([values_padded], dtype=torch.float, device=device)
            padding_mask = src.eq(pad_token_id)

            # _encode → mean pooling (non-padding)
            transformer_output = model._encode(
                src, val, padding_mask, batch_labels=None
            )

            valid = ~padding_mask
            valid_f = valid.unsqueeze(-1).float()
            cell_emb = (transformer_output * valid_f).sum(1) / valid_f.sum(1)
            embeddings.append(cell_emb.squeeze(0))

    centroid = torch.stack(embeddings).mean(dim=0)
    centroid = F.normalize(centroid, dim=0)
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
# GPR scoring (core)
# ============================================================
def gpr_score_single_cell(
    model, cell, target_centroid, tf_token_ids, vocab,
    device, k_adv=4, max_seq_len=1200,
    high_value=50, n_input_bins=51, rng=None,
    loss_mode="angular",
):
    """Compute the GPR score for one cell.

    Args:
        high_value: value at overexpression.
            bin index (e.g. 50) with binning, raw value (e.g. 10.0) with none.
    """
    if rng is None:
        rng = np.random.default_rng()

    pad_token_id = vocab["<pad>"]
    high_val = high_value

    # choose k random TFs
    adv_indices = rng.choice(len(tf_token_ids), size=k_adv, replace=False)
    adv_tfs = [tf_token_ids[i] for i in adv_indices]

    # copy of the cell data
    # Reserve k_adv slots for insertion (prevents collisions at the tail).
    # Cells shorter than max_seq_len - k_adv are unaffected.
    base_len = max(0, max_seq_len - k_adv)
    gene_ids = list(cell["gene_ids"][:base_len])
    values = list(cell["values"][:base_len])
    gene_id_to_pos = {gid: pos for pos, gid in enumerate(gene_ids)}

    # determine the adversarial positions (hybrid scheme)
    adv_positions = []
    for tf_name, tf_id in adv_tfs:
        if tf_id in gene_id_to_pos:
            # (a) existing gene -> change the expression value to high_value
            pos = gene_id_to_pos[tf_id]
            values[pos] = high_val
            adv_positions.append(pos)
        else:
            # (b) addition. It goes into a reserved slot, so no collision
            pos = len(gene_ids)
            gene_ids.append(tf_id)
            values.append(high_val)
            adv_positions.append(pos)

    # guarantee that no collision has occurred (it is a kind of bug that raises no exception)
    assert len(set(adv_positions)) == len(adv_positions), \
        f"adv_positions collided: {adv_positions}"
    assert len(gene_ids) <= max_seq_len, \
        f"seq_len {len(gene_ids)} exceeds max_seq_len {max_seq_len}"

    seq_len = len(gene_ids)

    # padding
    n_pad = max(0, max_seq_len - seq_len)
    gene_ids_padded = gene_ids + [pad_token_id] * n_pad
    pad_val = 0  # value at padding positions (0 = not expressed)
    values_padded = values + [pad_val] * n_pad

    src = torch.tensor([gene_ids_padded], dtype=torch.long, device=device)
    val = torch.tensor([values_padded], dtype=torch.float, device=device)
    padding_mask = src.eq(pad_token_id)

    # ---- manual _encode: requires_grad on gene_emb at the adversarial positions ----

    # take the whole gene embedding
    with torch.no_grad():
        gene_emb_full = model.encoder(src)  # (1, seq_len, d)

    # cut out the embedding at the adversarial positions
    gene_emb_data = gene_emb_full.detach().clone()
    adv_emb = gene_emb_data[0, adv_positions, :].clone().requires_grad_(True)

    # swap in
    gene_emb_modified = gene_emb_data.clone()
    for i, pos in enumerate(adv_positions):
        gene_emb_modified[0, pos, :] = adv_emb[i]

    # value embedding
    # ContinuousValueEncoder: Linear(1→d)→ReLU→Linear→LayerNorm
    val_emb = model.value_encoder(val.float())

    # total embedding
    total_embs = gene_emb_modified + val_emb

    # batch normalization
    if hasattr(model, "bn") and model.bn is not None:
        total_embs = model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

    # transformer encoder
    transformer_output = model.transformer_encoder(
        total_embs, src_key_padding_mask=padding_mask
    )

    # cell embedding: non-padding mean pooling
    valid_mask = (~padding_mask).unsqueeze(-1).float()
    cell_emb = (transformer_output * valid_mask).sum(1) / valid_mask.sum(1)
    cell_emb = cell_emb.squeeze(0)  # (d,)

    # loss computation
    loss, _ = compute_loss(cell_emb, target_centroid, loss_mode=loss_mode)
    loss.backward()

    # gradient of gene_emb at the adversarial positions
    g = adv_emb.grad  # (k_adv, d)
    if g is None:
        raise RuntimeError("No gradient on adv_emb. Check computation graph.")

    # ---- project onto gene_emb of all TFs ----
    all_tf_ids = torch.tensor(
        [tid for _, tid in tf_token_ids], dtype=torch.long, device=device
    )
    with torch.no_grad():
        tf_embs = model.encoder(all_tf_ids)  # (n_tf, d)

    # score[j] = mean_i(-g_i . E[j])
    with torch.no_grad():
        neg_g = -g  # (k_adv, d)
        proj = neg_g @ tf_embs.T  # (k_adv, n_tf)
        scores_tensor = proj.mean(dim=0)  # (n_tf,)

    scores = {
        name: float(scores_tensor[i].item())
        for i, (name, _) in enumerate(tf_token_ids)
    }
    return scores


# ============================================================
# GPR run (all cells x n_init)
# ============================================================
def run_gpr(model, source_cells, target_centroid, tf_token_ids, vocab,
            device, k_adv=4, n_init=5, max_seq_len=1200,
            n_input_bins=51, high_value=50, loss_mode="angular"):
    """Run GPR on all source cells and return the mean scores."""

    all_cell_scores = []
    for cell_idx, cell in enumerate(tqdm(source_cells, desc="GPR scoring")):
        init_scores = []
        for init_i in range(n_init):
            rng = np.random.default_rng(seed=cell_idx * 1000 + init_i)
            scores = gpr_score_single_cell(
                model, cell, target_centroid, tf_token_ids, vocab,
                device, k_adv=k_adv, max_seq_len=max_seq_len,
                high_value=high_value,
                n_input_bins=n_input_bins, rng=rng,
                loss_mode=loss_mode,
            )
            init_scores.append(scores)

        # mean over n_init
        avg = {}
        for name in init_scores[0]:
            avg[name] = np.mean([s[name] for s in init_scores])
        all_cell_scores.append(avg)

    # mean over all cells
    mean_scores = {}
    for name in all_cell_scores[0]:
        mean_scores[name] = np.mean([s[name] for s in all_cell_scores])

    return mean_scores, all_cell_scores


# ============================================================
# Evaluation
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
    parser = argparse.ArgumentParser(description="GPR for scGPT")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="scGPT_human checkpoint directory")
    parser.add_argument("--h5ad_path", type=str, required=True,
                        help="Path to h5ad (cardiac_subset.h5ad)")
    parser.add_argument("--tf_list", type=str, required=True,
                        help="TF list file (one symbol per line)")
    parser.add_argument("--pathway", type=str, default="iCM",
                        choices=list(PATHWAYS.keys()))
    parser.add_argument("--k_adv", type=int, default=4)
    parser.add_argument("--n_cells", type=int, default=50)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=1200)
    parser.add_argument("--binning_mode", type=str, default="global",
                        choices=["global", "per_cell", "none"],
                        help="global=GPR-specific (deterministic), per_cell=follows official scGPT, none=raw log1p values")
    parser.add_argument("--high_value_bin", type=int, default=50,
                        help="Bin index for overexpression (binning mode)")
    parser.add_argument("--high_value_cont", type=float, default=10.0,
                        help="Expression value for overexpression (none mode)")
    parser.add_argument("--n_target_cells", type=int, default=200)
    parser.add_argument("--use_continuous_value", action="store_true",
                        help="Use ContinuousValueEncoder instead of binning")
    parser.add_argument("--output_dir", type=str, default="./gpr_scgpt_results")
    parser.add_argument("--min_emb_norm", type=float, default=0.01,
                        help="Filter TFs with gene embedding norm <= this value")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss_mode", type=str, default="angular",
                        choices=["angular", "cosine"],
                        help="Loss function: angular (recommended) or cosine")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load the model
    print("Loading scGPT model...")
    model, vocab, model_config = load_scgpt_model(
        args.model_dir, device, use_continuous_value=args.use_continuous_value
    )
    vocab_stoi = vocab.get_stoi()
    n_input_bins = model_config["n_input_bins"]

    # TF list
    print("Loading TF list...")
    with open(args.tf_list) as f:
        tf_symbols = [line.strip() for line in f if line.strip()]
    tf_token_ids = [(tf, vocab_stoi[tf]) for tf in tf_symbols if tf in vocab_stoi]
    print(f"  {len(tf_symbols)} total TFs, {len(tf_token_ids)} in vocab")

    # filter TFs with zero embedding (genes absent from the training data)
    with torch.no_grad():
        all_tf_ids = torch.tensor([tid for _, tid in tf_token_ids],
                                  dtype=torch.long, device=device)
        all_tf_embs = model.encoder(all_tf_ids)
        norms = all_tf_embs.norm(dim=-1)  # (n_tf,)
        min_norm = args.min_emb_norm
        valid_mask = norms > min_norm
        n_filtered = (~valid_mask).sum().item()

    if n_filtered > 0:
        # save the TF names before filtering
        pre_filter_names = set(name for name, _ in tf_token_ids)
        tf_token_ids = [tf_token_ids[i] for i in range(len(tf_token_ids))
                        if valid_mask[i].item()]
        post_filter_names = set(name for name, _ in tf_token_ids)
        print(f"  Filtered {n_filtered} TFs with embedding norm <= {min_norm}")
        print(f"  Remaining TFs: {len(tf_token_ids)}")

        # check that no GT TF was removed by the filter
        gt_tfs = PATHWAYS[args.pathway]["gt_tfs"]
        gt_filtered = [tf for tf in gt_tfs if tf in pre_filter_names
                       and tf not in post_filter_names]
        if gt_filtered:
            print(f"  WARNING: GT TFs filtered out: {gt_filtered}")

    # pathway information
    pw = PATHWAYS[args.pathway]
    print(f"\nPathway: {args.pathway}")
    print(f"  {pw['source_type']} -> {pw['target_type']}")
    print(f"  GT TFs: {pw['gt_tfs']}")
    gt_in_vocab = [tf for tf in pw["gt_tfs"] if tf in vocab_stoi]
    gt_missing = [tf for tf in pw["gt_tfs"] if tf not in vocab_stoi]
    if gt_missing:
        print(f"  WARNING: GT TFs not in vocab: {gt_missing}")

    # load the data
    print(f"\nLoading h5ad: {args.h5ad_path}")
    adata = sc.read_h5ad(args.h5ad_path)
    print(f"  Shape: {adata.shape}")

    # prepare the source and target cells
    print(f"\nPreparing source cells...")
    source_cells = prepare_cell_data(
        adata, pw["source_type"], vocab,
        n_cells=args.n_cells, seed=args.seed,
        n_input_bins=n_input_bins, binning_mode=args.binning_mode,
    )

    print(f"Preparing target cells...")
    target_cells = prepare_cell_data(
        adata, pw["target_type"], vocab,
        n_cells=args.n_target_cells, seed=args.seed,
        n_input_bins=n_input_bins, binning_mode=args.binning_mode,
    )

    # compute the centroid
    print("Computing target centroid...")
    target_centroid = compute_target_centroid(
        model, target_cells, vocab, device,
        max_seq_len=args.max_seq_len, n_input_bins=n_input_bins,
    )

    # run GPR
    # high_value: determined by binning_mode
    if args.binning_mode == "none":
        high_value = args.high_value_cont
    else:
        high_value = args.high_value_bin
    print(f"  binning_mode={args.binning_mode}, high_value={high_value}")

    print(f"\nRunning GPR (k={args.k_adv}, n_init={args.n_init}, loss={args.loss_mode})...")
    mean_scores, all_cell_scores = run_gpr(
        model, source_cells, target_centroid, tf_token_ids, vocab,
        device, k_adv=args.k_adv, n_init=args.n_init,
        max_seq_len=args.max_seq_len, n_input_bins=n_input_bins,
        high_value=high_value, loss_mode=args.loss_mode,
    )

    # evaluation
    tf_names = [name for name, _ in tf_token_ids]
    auc, per_tf_ranks = evaluate_auc(mean_scores, pw["gt_tfs"], tf_names)

    # display
    print(f"\n{'='*60}")
    print(f"Results: {args.pathway} (scGPT GPR)")
    print(f"{'='*60}")
    print(f"ROC-AUC (strict): {auc:.3f}")
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
        "model": "scGPT_human",
        "method": "GPR",
        "k_adv": args.k_adv,
        "n_cells": len(source_cells),
        "n_init": args.n_init,
        "n_target_cells": len(target_cells),
        "max_seq_len": args.max_seq_len,
        "n_input_bins": n_input_bins,
        "n_tfs": len(tf_token_ids),
        "auc_strict": float(auc),
        "per_cell_auc_mean": float(np.mean(valid_aucs)) if valid_aucs else None,
        "per_cell_auc_std": float(np.std(valid_aucs)) if valid_aucs else None,
        "gt_tf_ranks": per_tf_ranks,
        "top50_tfs": [(n, float(s)) for n, s in ranked[:50]],
        "gt_tfs": pw["gt_tfs"],
    }

    outfile = os.path.join(
        args.output_dir,
        f"gpr_scgpt_{args.pathway}_k{args.k_adv}_n{len(source_cells)}.json",
    )
    with open(outfile, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()