#!/usr/bin/env python3
"""
GPR/I-GPR evaluation across 4 pathways for scGPT
===================================
Evaluate all 4 pathways (iCM, iHep, B->Mphi, Muscle) at once.
I-GPR computes the GPR gradient-projection score after running N steps of GCG discrete optimization.

Usage:
    # GPR (step 0) only
    python gpr_scgpt_multi_pathway.py \
        --model_dir /data/takemoto/advCell/scGPT_human \
        --h5ad_path /data/takemoto/advCell/geneformer_cardiac/h5ad/cardiac_subset.h5ad \
        --tf_list /data/takemoto/advCell/tf_list_human.txt \
        --n_cells 50 --k_adv 4

    # I-GPR (with iterations)
    python gpr_scgpt_multi_pathway.py \
        --model_dir ./scGPT_human \
        --h5ad_path ./geneformer_cardiac/h5ad/cardiac_subset.h5ad \
        --tf_list ./tf_list_human.txt \
        --n_cells 50 --k_adv 4 --igpr_steps 5 10 20 \
        --binning_mode none \
        --output_dir ./gpr_scgpt_results_none \
        --verify
        global
        per_cell
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

from scgpt.model.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils.util import load_pretrained

warnings.filterwarnings("ignore")

# import the common functions from gpr_scgpt.py
# assumed to be in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from advcell_pathway_loader import load_pathway_config, get_pathway_ids
from gpr_scgpt import (
    load_scgpt_model,
    prepare_cell_data,
    compute_target_centroid,
    compute_loss,
    gpr_score_single_cell,
    evaluate_auc,
)


# ============================================================
# I-GPR: GPR scoring after GCG discrete optimization (following Zou et al. 2023)
# ============================================================
def _apply_hybrid_overexpression(base_gene_ids, base_values, adv_tfs,
                                 high_val, max_seq_len):
    """Place the adversarial TFs in the hybrid scheme and return gene_ids, values, adv_positions.

    Placement of absent TFs: put them at distinct positions from the end backwards (1199, 1198, ...).
    Positions already taken by another TF, or where an existing TF sits, are skipped.
    """
    gene_ids = list(base_gene_ids)
    values = list(base_values)
    gene_id_to_pos = {gid: pos for pos, gid in enumerate(gene_ids)}

    adv_positions = []
    used_positions = set()  # to avoid collisions

    for tf_name, tf_id in adv_tfs:
        if tf_id in gene_id_to_pos:
            # TF already in the cell -> only change the value
            pos = gene_id_to_pos[tf_id]
            values[pos] = high_val
            adv_positions.append(pos)
            used_positions.add(pos)
        else:
            if len(gene_ids) < max_seq_len:
                # room in the array -> append
                pos = len(gene_ids)
                gene_ids.append(tf_id)
                values.append(high_val)
                adv_positions.append(pos)
                used_positions.add(pos)
            else:
                # look for free positions backwards from the end
                pos = len(gene_ids) - 1
                while pos >= 0 and pos in used_positions:
                    pos -= 1
                if pos < 0:
                    raise RuntimeError(
                        f"No available position for TF {tf_name} "
                        f"(max_seq_len={max_seq_len}, "
                        f"used={len(used_positions)})"
                    )
                gene_ids[pos] = tf_id
                values[pos] = high_val
                adv_positions.append(pos)
                used_positions.add(pos)

    return gene_ids, values, adv_positions


def _scgpt_forward_cell_emb(model, gene_ids, values, pad_token_id,
                             max_seq_len, n_input_bins, device):
    """Manual _encode of scGPT -> cell embedding (1 sample, no_grad)."""
    seq_len = len(gene_ids)
    n_pad = max(0, max_seq_len - seq_len)
    gene_ids_padded = gene_ids + [pad_token_id] * n_pad
    pad_val = 0 if n_input_bins > 0 else 0.0
    values_padded = values + [pad_val] * n_pad

    src = torch.tensor([gene_ids_padded], dtype=torch.long, device=device)
    val = torch.tensor([values_padded], dtype=torch.float, device=device)
    padding_mask = src.eq(pad_token_id)

    gene_emb = model.encoder(src)
    val_emb = model.value_encoder(val.float())
    total_embs = gene_emb + val_emb

    if hasattr(model, "bn") and model.bn is not None:
        total_embs = model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

    transformer_output = model.transformer_encoder(
        total_embs, src_key_padding_mask=padding_mask
    )
    valid_mask = (~padding_mask).unsqueeze(-1).float()
    cell_emb = (transformer_output * valid_mask).sum(1) / valid_mask.sum(1)
    return cell_emb.squeeze(0)


def _batch_evaluate_candidates(model, base_gene_ids, base_values,
                                candidate_tf_sets, target_centroid,
                                high_val, pad_token_id,
                                max_seq_len, n_input_bins, device):
    """Evaluate a group of candidate TF sets by batch forward and return the cosine similarity of each candidate.

    Args:
        candidate_tf_sets: list of list[(name, id)] -- the k TFs of each candidate

    Returns:
        sims: (n_candidates,) tensor of cosine similarities
    """
    n_cand = len(candidate_tf_sets)
    pad_val = 0 if n_input_bins > 0 else 0.0

    # build the input of each candidate
    all_src = []
    all_val = []
    for cand_tfs in candidate_tf_sets:
        gids, vals, _ = _apply_hybrid_overexpression(
            base_gene_ids, base_values, cand_tfs, high_val, max_seq_len
        )
        seq_len = len(gids)
        n_pad = max(0, max_seq_len - seq_len)
        all_src.append(gids + [pad_token_id] * n_pad)
        all_val.append(vals + [pad_val] * n_pad)

    src = torch.tensor(all_src, dtype=torch.long, device=device)
    val = torch.tensor(all_val, dtype=torch.float, device=device)
    padding_mask = src.eq(pad_token_id)

    gene_emb = model.encoder(src)
    val_emb = model.value_encoder(val.float())
    total_embs = gene_emb + val_emb

    if hasattr(model, "bn") and model.bn is not None:
        total_embs = model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

    transformer_output = model.transformer_encoder(
        total_embs, src_key_padding_mask=padding_mask
    )
    valid_mask = (~padding_mask).unsqueeze(-1).float()
    cell_embs = (transformer_output * valid_mask).sum(1) / valid_mask.sum(1)

    sims = F.cosine_similarity(cell_embs, target_centroid.unsqueeze(0), dim=1)
    return sims


def igpr_score_single_cell(
    model, cell, target_centroid, tf_token_ids, vocab,
    device, k_adv=4, max_seq_len=1200,
    high_value=50,
    n_input_bins=51, n_steps=20, top_k_grad=256,
    batch_candidates=128, rng=None,
    loss_mode="angular",
):
    """I-GPR: run n_steps steps of GCG discrete optimization and then
    compute the GPR gradient-projection score.

    Follows the GCG algorithm of Zou et al. 2023:
      1. get the gradient at the adversarial positions by forward + backward
      2. enumerate the top-k candidate tokens based on the gradient
      3. evaluate, by batch forward, the candidates obtained by swapping one position at a time
      4. adopt only if the cosine similarity improves (greedy acceptance)
      5. repeat n_steps times

    At the final step, compute and return the GPR gradient-projection score.

    Returns:
        scores: dict {symbol: float}
        traj: list[float]  -- cos_sim on entering each step + the final cos_sim.
              Length n_steps+1 when completed.
        completed: bool  -- False if it broke off midway because the gradient was None.
    """
    if rng is None:
        rng = np.random.default_rng()

    pad_token_id = vocab["<pad>"]
    use_binning = n_input_bins > 0
    high_val = high_value
    n_tf = len(tf_token_ids)

    traj = []          # trajectory of cos_sim
    completed = True

    # basic copy of the cell data
    base_gene_ids = list(cell["gene_ids"][:max_seq_len])
    base_values = list(cell["values"][:max_seq_len])

    # embeddings of all TFs (for gradient projection, computed once)
    all_tf_ids_tensor = torch.tensor(
        [tid for _, tid in tf_token_ids], dtype=torch.long, device=device
    )
    with torch.no_grad():
        all_tf_embs = model.encoder(all_tf_ids_tensor)  # (n_tf, d)

    # initial random TFs
    adv_indices = rng.choice(n_tf, size=k_adv, replace=False)
    current_adv_tfs = [tf_token_ids[i] for i in adv_indices]

    # ----- GCG steps (following Zou et al. 2023) -----
    for step in range(n_steps):
        # 1. hybrid placement + forward + backward
        gene_ids, values, adv_positions = _apply_hybrid_overexpression(
            base_gene_ids, base_values, current_adv_tfs, high_val, max_seq_len
        )

        seq_len = len(gene_ids)
        n_pad = max(0, max_seq_len - seq_len)
        gene_ids_padded = gene_ids + [pad_token_id] * n_pad
        pad_val = 0 if use_binning else 0.0
        values_padded = values + [pad_val] * n_pad

        src = torch.tensor([gene_ids_padded], dtype=torch.long, device=device)
        val = torch.tensor([values_padded], dtype=torch.float, device=device)
        padding_mask = src.eq(pad_token_id)

        with torch.no_grad():
            gene_emb_full = model.encoder(src)

        gene_emb_data = gene_emb_full.detach().clone()
        adv_emb = gene_emb_data[0, adv_positions, :].clone().requires_grad_(True)

        gene_emb_modified = gene_emb_data.clone()
        for i, pos in enumerate(adv_positions):
            gene_emb_modified[0, pos, :] = adv_emb[i]

        val_emb = model.value_encoder(val.float())
        total_embs = gene_emb_modified + val_emb

        if hasattr(model, "bn") and model.bn is not None:
            total_embs = model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        transformer_output = model.transformer_encoder(
            total_embs, src_key_padding_mask=padding_mask
        )
        valid_mask = (~padding_mask).unsqueeze(-1).float()
        cell_emb = (transformer_output * valid_mask).sum(1) / valid_mask.sum(1)
        cell_emb = cell_emb.squeeze(0)

        cos_sim = F.cosine_similarity(
            cell_emb.unsqueeze(0), target_centroid.unsqueeze(0)
        )
        cur_sim = cos_sim.item()
        traj.append(cur_sim)  # cos_sim on entering this step

        loss, _ = compute_loss(cell_emb, target_centroid, loss_mode=loss_mode)
        loss.backward()

        g = adv_emb.grad  # (k_adv, d)
        if g is None:
            completed = False
            break

        # 2. top-k candidate TFs based on the gradient
        with torch.no_grad():
            neg_g = -g  # (k_adv, d)
            proj_scores = neg_g @ all_tf_embs.T  # (k_adv, n_tf)
            _, topk_idx = proj_scores.topk(
                min(top_k_grad, n_tf), dim=1
            )  # (k_adv, top_k_grad)

        # 3. generate candidates with a one-position swap
        n_eval = min(batch_candidates, k_adv * top_k_grad)
        positions = rng.integers(0, k_adv, (n_eval,))
        cand_idx = rng.integers(0, min(top_k_grad, n_tf), (n_eval,))

        candidate_tf_sets = []
        seen = set()
        for b in range(n_eval):
            pos_b = positions[b].item()
            new_tf_idx = topk_idx[pos_b, cand_idx[b]]
            new_tf = tf_token_ids[new_tf_idx]

            cand = list(current_adv_tfs)
            cand[pos_b] = new_tf

            # duplicate check (all k must be distinct TFs)
            cand_ids = tuple(tid for _, tid in cand)
            if len(set(cand_ids)) != k_adv:
                continue
            if cand_ids in seen:
                continue
            seen.add(cand_ids)
            candidate_tf_sets.append(cand)

        if not candidate_tf_sets:
            continue

        # 4. batch forward evaluation
        with torch.no_grad():
            sims = _batch_evaluate_candidates(
                model, base_gene_ids, base_values,
                candidate_tf_sets, target_centroid,
                high_val, pad_token_id,
                max_seq_len, n_input_bins, device,
            )

        # 5. greedy acceptance: adopt only on improvement
        best_idx = sims.argmax().item()
        if sims[best_idx].item() > cur_sim:
            current_adv_tfs = candidate_tf_sets[best_idx]

    # ----- final GPR scoring -----
    gene_ids, values, adv_positions = _apply_hybrid_overexpression(
        base_gene_ids, base_values, current_adv_tfs, high_val, max_seq_len
    )

    seq_len = len(gene_ids)
    n_pad = max(0, max_seq_len - seq_len)
    gene_ids_padded = gene_ids + [pad_token_id] * n_pad
    pad_val = 0 if use_binning else 0.0
    values_padded = values + [pad_val] * n_pad

    src = torch.tensor([gene_ids_padded], dtype=torch.long, device=device)
    val = torch.tensor([values_padded], dtype=torch.float, device=device)
    padding_mask = src.eq(pad_token_id)

    with torch.no_grad():
        gene_emb_full = model.encoder(src)

    gene_emb_data = gene_emb_full.detach().clone()
    adv_emb = gene_emb_data[0, adv_positions, :].clone().requires_grad_(True)

    gene_emb_modified = gene_emb_data.clone()
    for i, pos in enumerate(adv_positions):
        gene_emb_modified[0, pos, :] = adv_emb[i]

    val_emb = model.value_encoder(val.float())
    total_embs = gene_emb_modified + val_emb

    if hasattr(model, "bn") and model.bn is not None:
        total_embs = model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

    transformer_output = model.transformer_encoder(
        total_embs, src_key_padding_mask=padding_mask
    )
    valid_mask = (~padding_mask).unsqueeze(-1).float()
    cell_emb = (transformer_output * valid_mask).sum(1) / valid_mask.sum(1)
    cell_emb = cell_emb.squeeze(0)

    cos_sim = F.cosine_similarity(
        cell_emb.unsqueeze(0), target_centroid.unsqueeze(0)
    )
    final_sim = cos_sim.item()

    loss, _ = compute_loss(cell_emb, target_centroid, loss_mode=loss_mode)
    loss.backward()

    g = adv_emb.grad
    with torch.no_grad():
        neg_g = -g
        proj = neg_g @ all_tf_embs.T
        scores_tensor = proj.mean(dim=0)

    scores = {
        name: float(scores_tensor[i].item())
        for i, (name, _) in enumerate(tf_token_ids)
    }
    traj.append(final_sim)  # cos_sim of the finally adopted state
    return scores, traj, completed


def run_igpr(model, source_cells, target_centroid, tf_token_ids, vocab,
             device, k_adv=4, n_init=5, max_seq_len=1200,
             n_input_bins=51, n_steps=20, top_k_grad=256,
             batch_candidates=128, high_value=50, loss_mode="angular"):
    """Run I-GPR on all source cells.

    Returns:
        mean_scores: dict {symbol: float}
        all_cell_scores: list of dict
        traj_mean: list[float]  -- the element-wise mean of the cos_sim trajectories of the (cell, init)
                   that completed (length n_steps+1). None if all dropped out.
        n_excluded: int  -- number of (cell, init) excluded from the trajectory aggregate because they broke off midway.
    """
    all_cell_scores = []
    traj_list = []
    n_excluded = 0
    expected_len = n_steps + 1
    for cell_idx, cell in enumerate(tqdm(
        source_cells, desc=f"I-GPR (step={n_steps})"
    )):
        init_scores = []
        for init_i in range(n_init):
            rng = np.random.default_rng(seed=cell_idx * 1000 + init_i)
            scores, traj, completed = igpr_score_single_cell(
                model, cell, target_centroid, tf_token_ids, vocab,
                device, k_adv=k_adv, max_seq_len=max_seq_len,
                high_value=high_value,
                n_input_bins=n_input_bins,
                n_steps=n_steps, top_k_grad=top_k_grad,
                batch_candidates=batch_candidates, rng=rng,
                loss_mode=loss_mode,
            )
            init_scores.append(scores)
            if completed and len(traj) == expected_len:
                traj_list.append(traj)
            else:
                n_excluded += 1

        avg = {}
        for name in init_scores[0]:
            avg[name] = np.mean([s[name] for s in init_scores])
        all_cell_scores.append(avg)

    mean_scores = {}
    for name in all_cell_scores[0]:
        mean_scores[name] = np.mean([s[name] for s in all_cell_scores])

    if traj_list:
        traj_arr = np.array(traj_list)
        traj_mean = traj_arr.mean(axis=0).tolist()
    else:
        traj_mean = None

    return mean_scores, all_cell_scores, traj_mean, n_excluded


# ============================================================
# (B) Validation: actually insert the top predicted TFs and measure cos_sim
# ============================================================
def _cos_sim_inserted_scgpt(model, cell, target_centroid, insert_tfs, vocab,
                            high_val, max_seq_len, n_input_bins, device):
    """Insert the given TFs with the same hybrid overexpression as the attack and return cos_sim.

    insert_tfs: list[(name, id)]  -- an empty list = source alone.
    """
    pad_token_id = vocab["<pad>"]
    base_gene_ids = list(cell["gene_ids"][:max_seq_len])
    base_values = list(cell["values"][:max_seq_len])
    if len(insert_tfs) == 0:
        gids, vals = base_gene_ids, base_values
    else:
        gids, vals, _ = _apply_hybrid_overexpression(
            base_gene_ids, base_values, insert_tfs, high_val, max_seq_len
        )
    with torch.no_grad():
        cell_emb = _scgpt_forward_cell_emb(
            model, gids, vals, pad_token_id,
            max_seq_len, n_input_bins, device,
        )
        cos = F.cosine_similarity(
            cell_emb.unsqueeze(0), target_centroid.unsqueeze(0)
        ).item()
    return cos


def run_verification(model, source_cells, target_centroid, ranking_scores,
                     tf_token_ids, gt_tfs, vocab_stoi, vocab, device,
                     high_val, max_seq_len, n_input_bins, k_adv=4,
                     topn_list=(1, 2, 4, 8), n_random=5, seed=42):
    """Aggregate, for each condition, the cos_sim when the top-ranked TFs are actually inserted.

    Conditions: source_only / random_k / gt / pred_top{N} / pred_top_gt
    """
    sym2tid = {name: tid for name, tid in tf_token_ids}
    ranked = [name for name, _ in
              sorted(ranking_scores.items(), key=lambda x: x[1], reverse=True)]

    # GT may have dropped out of tf_token_ids through the norm filter, so look it up in the vocab
    gt_in_vocab = [g for g in gt_tfs if g in vocab_stoi]
    n_gt = len(gt_in_vocab)

    conditions = {}
    conditions["source_only"] = []
    for N in topn_list:
        conditions[f"pred_top{N}"] = [(s, sym2tid[s]) for s in ranked[:N]]
    if n_gt > 0:
        conditions[f"pred_top_gt({n_gt})"] = [
            (s, sym2tid[s]) for s in ranked[:n_gt]
        ]
        conditions["gt"] = [(g, vocab_stoi[g]) for g in gt_in_vocab]

    result = {}
    for cond, tfs in conditions.items():
        vals = [
            _cos_sim_inserted_scgpt(
                model, cell, target_centroid, tfs, vocab,
                high_val, max_seq_len, n_input_bins, device,
            )
            for cell in source_cells
        ]
        result[cond] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    # random_k: sample n_random times per cell and average
    rng = np.random.default_rng(seed)
    n_tf = len(tf_token_ids)
    rand_vals = []
    for cell in source_cells:
        cell_vals = []
        for _ in range(n_random):
            idx = rng.choice(n_tf, size=k_adv, replace=False)
            tfs = [tf_token_ids[i] for i in idx]
            cell_vals.append(
                _cos_sim_inserted_scgpt(
                    model, cell, target_centroid, tfs, vocab,
                    high_val, max_seq_len, n_input_bins, device,
                )
            )
        rand_vals.append(np.mean(cell_vals))
    result["random_k"] = {
        "mean": float(np.mean(rand_vals)),
        "std": float(np.std(rand_vals)),
    }
    return result


def _print_verification(cos_res):
    """Print the validation results one per line."""
    order = ["source_only", "random_k"]
    order += [k for k in cos_res if k.startswith("pred_top") and "gt" not in k]
    order += [k for k in cos_res if "pred_top_gt" in k]
    order += ["gt"]
    for k in order:
        if k in cos_res:
            r = cos_res[k]
            print(f"    {k:16s}: cos_sim={r['mean']:.4f} ± {r['std']:.4f}")


# ============================================================
# Main: evaluation across 4 pathways
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="GPR/I-GPR multi-pathway evaluation for scGPT"
    )
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--pathway_config", type=str, required=True,
                        help="Path to pathway_config_geneformer.json")
    parser.add_argument("--tf_list", type=str, required=True)
    parser.add_argument("--pathways", nargs="+",
                        default=["all"])
    parser.add_argument("--k_adv", type=int, default=4)
    parser.add_argument("--n_cells", type=int, default=50)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--max_seq_len", type=int, default=1200)
    parser.add_argument("--n_target_cells", type=int, default=200)
    parser.add_argument("--igpr_steps", nargs="*", type=int, default=[],
                        help="I-GPR step counts (e.g. 5 10 20 50). Empty=GPR only.")
    parser.add_argument("--top_k_grad", type=int, default=256,
                        help="Gradient candidates per position (GCG)")
    parser.add_argument("--batch_candidates", type=int, default=128,
                        help="Candidates to evaluate per GCG step")
    parser.add_argument("--binning_mode", type=str, default="global",
                        choices=["global", "per_cell", "none"])
    parser.add_argument("--high_value_bin", type=int, default=50)
    parser.add_argument("--high_value_cont", type=float, default=10.0)
    parser.add_argument("--use_continuous_value", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./gpr_scgpt_results")
    parser.add_argument("--min_emb_norm", type=float, default=0.01,
                        help="Filter TFs with gene embedding norm <= this value")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss_mode", type=str, default="angular",
                        choices=["angular", "cosine"],
                        help="Loss function: angular (recommended) or cosine")
    # (B) validation pass
    parser.add_argument("--verify", action="store_true",
                        help="enable the validation that actually inserts the top predicted TFs and measures cos_sim")
    parser.add_argument("--verify_topn", nargs="+", type=int,
                        default=[1, 2, 4, 8],
                        help="list of the numbers of top TFs inserted in the validation")
    parser.add_argument("--n_random", type=int, default=5,
                        help="number of samples of the random_k baseline")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # model and TFs
    model, vocab, model_config = load_scgpt_model(
        args.model_dir, device, use_continuous_value=args.use_continuous_value
    )
    vocab_stoi = vocab.get_stoi()
    n_input_bins = model_config["n_input_bins"]

    with open(args.tf_list) as f:
        tf_symbols = [line.strip() for line in f if line.strip()]
    tf_token_ids = [(tf, vocab_stoi[tf]) for tf in tf_symbols if tf in vocab_stoi]
    print(f"TFs in vocab: {len(tf_token_ids)}/{len(tf_symbols)}")

    # filter TFs with zero embedding
    with torch.no_grad():
        all_tf_ids = torch.tensor([tid for _, tid in tf_token_ids],
                                  dtype=torch.long, device=device)
        all_tf_embs = model.encoder(all_tf_ids)
        norms = all_tf_embs.norm(dim=-1)
        valid_mask = norms > args.min_emb_norm
        n_filtered = (~valid_mask).sum().item()
    if n_filtered > 0:
        tf_token_ids = [tf_token_ids[i] for i in range(len(tf_token_ids))
                        if valid_mask[i].item()]
        print(f"Filtered {n_filtered} TFs with norm <= {args.min_emb_norm}, "
              f"remaining: {len(tf_token_ids)}")

    tf_names = [name for name, _ in tf_token_ids]

    # data
    # load from pathway_config_geneformer.json
    pw_config, h5ad_paths = load_pathway_config(args.pathway_config, model="scgpt")
    pw_ids = get_pathway_ids(pw_config, args.pathways if args.pathways != ["all"] else None)
    print(f"Pathways to run: {pw_ids}")

    # high_value: determined by binning_mode
    if args.binning_mode == "none":
        high_value = args.high_value_cont
    else:
        high_value = args.high_value_bin
    print(f"binning_mode={args.binning_mode}, high_value={high_value}")

    # evaluate on all pathways
    all_results = {}
    summary_lines = []

    for pw_name in pw_ids:
        pw = pw_config[pw_name]
        adata = sc.read_h5ad(h5ad_paths[pw_name])
        print(f"\n{'='*60}")
        print(f"Pathway: {pw_name}  ({pw['source_type']} -> {pw['target_type']})")
        print(f"{'='*60}")

        # data preparation
        source_cells = prepare_cell_data(
            adata, pw["source_type"], vocab,
            n_cells=args.n_cells, seed=args.seed,
            n_input_bins=n_input_bins, binning_mode=args.binning_mode,
        )
        target_cells = prepare_cell_data(
            adata, pw["target_type"], vocab,
            n_cells=args.n_target_cells, seed=args.seed,
            n_input_bins=n_input_bins, binning_mode=args.binning_mode,
        )
        target_centroid = compute_target_centroid(
            model, target_cells, vocab, device,
            max_seq_len=args.max_seq_len, n_input_bins=n_input_bins,
        )

        pw_results = {}

        # ---- GPR (step 0) ----
        print(f"\nRunning GPR (step 0, loss={args.loss_mode})...")
        from gpr_scgpt import run_gpr
        mean_scores, all_cell_scores = run_gpr(
            model, source_cells, target_centroid, tf_token_ids, vocab,
            device, k_adv=args.k_adv, n_init=args.n_init,
            max_seq_len=args.max_seq_len, n_input_bins=n_input_bins,
            high_value=high_value, loss_mode=args.loss_mode,
        )
        auc, ranks = evaluate_auc(mean_scores, pw["gt_tfs"], tf_names)
        pw_results["step_0"] = {
            "auc": float(auc), "ranks": ranks,
            # Save all scores for post-hoc analysis (e.g. evaluating pathway specificity).
            # Without this, every added metric would require a rerun.
            "scores": {k: round(float(v), 6) for k, v in mean_scores.items()},
        }
        line = f"{pw_name:10s}  GPR(step=0):  AUC={auc:.3f}"
        print(line)
        summary_lines.append(line)

        # (B) Validation: actually insert the top predicted TFs and measure cos_sim
        if args.verify:
            print("  [verify] GPR ranking cos_sim:")
            cos_res = run_verification(
                model, source_cells, target_centroid, mean_scores,
                tf_token_ids, pw["gt_tfs"], vocab_stoi, vocab, device,
                high_val=high_value, max_seq_len=args.max_seq_len,
                n_input_bins=n_input_bins, k_adv=args.k_adv,
                topn_list=tuple(args.verify_topn),
                n_random=args.n_random, seed=args.seed,
            )
            pw_results["step_0"]["cos_sim"] = cos_res
            _print_verification(cos_res)

        # ---- I-GPR (for each number of steps) ----
        for n_steps in args.igpr_steps:
            print(f"\nRunning I-GPR (step={n_steps}, loss={args.loss_mode})...")
            mean_scores_i, all_cell_scores_i, traj_mean, n_excluded = run_igpr(
                model, source_cells, target_centroid, tf_token_ids, vocab,
                device, k_adv=args.k_adv, n_init=args.n_init,
                max_seq_len=args.max_seq_len, n_input_bins=n_input_bins,
                n_steps=n_steps, top_k_grad=args.top_k_grad,
                batch_candidates=args.batch_candidates,
                high_value=high_value, loss_mode=args.loss_mode,
            )
            auc_i, ranks_i = evaluate_auc(
                mean_scores_i, pw["gt_tfs"], tf_names
            )
            pw_results[f"step_{n_steps}"] = {
                "auc": float(auc_i), "ranks": ranks_i,
                "scores": {k: round(float(v), 6)
                           for k, v in mean_scores_i.items()},
                # (A) cos_sim trajectory during optimization (on entering each step x n_steps + final)
                "cos_sim_trajectory": traj_mean,
                "cos_sim_traj_excluded": n_excluded,
            }
            line = f"{pw_name:10s}  I-GPR(step={n_steps:3d}): AUC={auc_i:.3f}"
            print(line)
            summary_lines.append(line)

            # (A) display the key points of the trajectory
            if traj_mean is not None:
                print(f"  [traj] cos_sim: init={traj_mean[0]:.4f} "
                      f"-> final={traj_mean[-1]:.4f} "
                      f"(Δ={traj_mean[-1] - traj_mean[0]:+.4f}, "
                      f"excluded={n_excluded})")

            # (B) Validation: actually insert the top predicted TFs and measure cos_sim
            if args.verify:
                print(f"  [verify] I-GPR(step={n_steps}) ranking cos_sim:")
                cos_res = run_verification(
                    model, source_cells, target_centroid, mean_scores_i,
                    tf_token_ids, pw["gt_tfs"], vocab_stoi, vocab, device,
                    high_val=high_value, max_seq_len=args.max_seq_len,
                    n_input_bins=n_input_bins, k_adv=args.k_adv,
                    topn_list=tuple(args.verify_topn),
                    n_random=args.n_random, seed=args.seed,
                )
                pw_results[f"step_{n_steps}"]["cos_sim"] = cos_res
                _print_verification(cos_res)

        all_results[pw_name] = pw_results

    # summary display
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for line in summary_lines:
        print(line)

    # comparison table with Geneformer V1
    gf_v1 = {
        "iCM": {"GPR": 0.858, "I-GPR_best": 0.905},
        "iHep": {"GPR": 0.878, "I-GPR_best": 0.917},
        "B_to_Mac": {"GPR": 0.872, "I-GPR_best": 0.872},
        "Muscle": {"GPR": 0.652, "I-GPR_best": 0.830},
    }
    # Note: the reference values are from the old 4-pathway setting and their keys do not match P01-P18
    print(f"\nComparison with Geneformer V1 [STALE: reference values of the old 4-pathway setting]:")
    print(f"{'Pathway':10s} {'scGPT GPR':>10s} {'GF-V1 GPR':>10s} {'GF-V1 best':>10s}")
    for pw_name in all_results:
        scgpt_auc = all_results[pw_name]["step_0"]["auc"]
        gf_gpr = gf_v1.get(pw_name, {}).get("GPR", float("nan"))
        gf_best = gf_v1.get(pw_name, {}).get("I-GPR_best", float("nan"))
        print(f"{pw_name:10s} {scgpt_auc:10.3f} {gf_gpr:10.3f} {gf_best:10.3f}")

    # save
    # The config is stored alongside. If the settings cannot be traced later, it becomes
    # impossible to tell which conditions a result belongs to.
    save_data = {
        "config": {
            "model": "scGPT",
            "k_adv": args.k_adv,
            "n_cells": args.n_cells,
            "n_init": args.n_init,
            "n_target_cells": args.n_target_cells,
            "max_seq_len": args.max_seq_len,
            "igpr_steps": args.igpr_steps,
            "top_k_grad": args.top_k_grad,
            "batch_candidates": args.batch_candidates,
            "binning_mode": args.binning_mode,
            "loss_mode": args.loss_mode,
            "n_tfs": len(tf_names),
            "pathways": list(all_results),
            "seed": args.seed,
        },
        "results": all_results,
    }
    outfile = os.path.join(
        args.output_dir,
        f"multi_pathway_scgpt_k{args.k_adv}_n{args.n_cells}.json",
    )
    tmp = outfile + ".tmp"
    with open(tmp, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, outfile)
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()