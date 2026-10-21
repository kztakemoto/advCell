#!/usr/bin/env python3
"""
GPR/I-GPR evaluation across 4 pathways for Geneformer V1
============================================
Evaluate all 4 pathways (iCM, iHep, B->Mphi, Muscle) at once.
I-GPR computes the GPR gradient-projection score after running N steps of GCG discrete optimization.

Usage:
    # GPR (step 0) only
    python gpr_geneformer_multi_pathway.py \
        --model_dir /data/takemoto/advCell/Geneformer/Geneformer-V1-10M \
        --dataset_path /data/takemoto/advCell/geneformer_cardiac/tokenized_v1/cardiac_v1.dataset \
        --dict_dir /data/takemoto/advCell/Geneformer/geneformer/gene_dictionaries_30m \
        --tf_list /data/takemoto/advCell/tf_list_human.txt \
        --n_cells 50 --k_adv 4

    # I-GPR (with iterations)
    python gpr_geneformer_multi_pathway.py \
        --model_dir ./Geneformer/Geneformer-V1-10M \
        --dataset_path ./geneformer_cardiac/tokenized_v1/cardiac_v1.dataset \
        --dict_dir ./Geneformer/geneformer/gene_dictionaries_30m \
        --tf_list ./tf_list_human.txt \
        --n_cells 50 --k_adv 4 --n_init 5 \
        --igpr_steps 5 10 20 
        --verify
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# import the common functions from gpr_geneformer.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from advcell_pathway_loader import load_pathway_config, get_pathway_ids
from advcell_aggregate import (
    GammaAccumulator,
    geometry_diagnostics,
    format_geometry,
    normalize_by_norm,
    auc_vs_other_gt,
)
from advcell_overexpress import (
    build_inputs_embeds,
    masked_mean_pool,
    overexpress_batch,
    overexpress_single,
    set_overexpress_mode,
)
from gpr_geneformer import (
    MAX_SEQ_LEN,
    PAD_TOKEN_ID,
    SPECIAL_TOKEN_IDS,
    load_geneformer_model,
    load_dicts,
    load_tf_list,
    prepare_cell_data,
    compute_target_centroid,
    compute_loss,
    forward_cell_embedding,
    set_emb_layer,
    get_emb_layer,
    gpr_score_single_cell,
    run_gpr,
    evaluate_auc,
)


# ============================================================
# I-GPR: GPR scoring after GCG discrete optimization (1 cell x 1 initialization)
# ============================================================
def _gpr_scores_at_position(model, adv_ids, orig_ids_t, target_centroid,
                            tf_token_ids, k_adv, max_seq_len, device,
                            loss_mode="angular"):
    """Compute the GPR gradient-projection score at the current adversarial token positions.

    overexpress is the pop-and-insert following the official InSilicoPerturber.

    Returns:
        scores: dict {symbol: float}
        cos_sim: float
    """
    E = model.embeddings.word_embeddings.weight

    ids, attn, _ = overexpress_single(orig_ids_t, adv_ids, max_seq_len)
    adv_emb = E[adv_ids].detach().clone().requires_grad_(True)
    combined = build_inputs_embeds(E, ids, k_adv, adv_emb=adv_emb)

    cell_emb = forward_cell_embedding(
        model, attn, inputs_embeds=combined
    ).squeeze(0)

    # always the cosine basis (gamma weighting is done on the aggregation side)
    loss, cos_sim_val = compute_loss(cell_emb, target_centroid,
                                     loss_mode="cosine")
    loss.backward()

    g = adv_emb.grad  # (k, D)
    all_tf_ids_tensor = torch.tensor(
        [tid for _, tid in tf_token_ids], dtype=torch.long, device=device,
    )
    with torch.no_grad():
        tf_embs = E[all_tf_ids_tensor]
        neg_g = -g
        proj = neg_g @ tf_embs.T  # (k, n_tf)
        scores_tensor = proj.mean(dim=0)

    scores = {
        name: float(scores_tensor[i].item())
        for i, (name, _) in enumerate(tf_token_ids)
    }
    return scores, cos_sim_val


def igpr_score_single_cell(
    model, cell, target_centroid, tf_token_ids,
    device, k_adv=4, max_seq_len=MAX_SEQ_LEN,
    n_steps=20, top_k_grad=256, batch_candidates=128,
    rng=None, loss_mode="angular",
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
        n_removed_traj: list[int]  -- number of tokens removed from the original sequence
              at each step. For diagnosing whether GCG exploits the duplication artifact.
    """
    if rng is None:
        rng = np.random.default_rng()

    E = model.embeddings.word_embeddings.weight  # (V, D)
    n_tf = len(tf_token_ids)

    traj = []              # trajectory of cos_sim
    n_removed_traj = []    # number of removed tokens at each step
    completed = True

    # original token sequence (truncation is done on the overexpress side)
    orig_ids_t = torch.tensor(
        cell["input_ids"][:max_seq_len], dtype=torch.long, device=device,
    )

    # tensor of TF token IDs (to restrict candidates)
    all_tf_ids_tensor = torch.tensor(
        [tid for _, tid in tf_token_ids], dtype=torch.long, device=device,
    )
    # mask restricting candidates to TFs only
    V = E.shape[0]
    ban_mask = torch.ones(V, device=device, dtype=torch.bool)
    ban_mask[all_tf_ids_tensor] = False

    # initial random TFs
    adv_indices = rng.choice(n_tf, size=k_adv, replace=False)
    adv = torch.tensor(
        [tf_token_ids[i][1] for i in adv_indices],
        dtype=torch.long, device=device,
    )

    # ----- GCG steps (following Zou et al. 2023) -----
    for step in range(n_steps):
        # 1. forward + backward
        ids, attn, n_rem = overexpress_single(orig_ids_t, adv, max_seq_len)
        n_removed_traj.append(int(n_rem[0].item()))

        adv_emb = E[adv].detach().clone().requires_grad_(True)
        combined = build_inputs_embeds(E, ids, k_adv, adv_emb=adv_emb)

        cell_emb = forward_cell_embedding(
            model, attn, inputs_embeds=combined
        ).squeeze(0)

        cos_sim = F.cosine_similarity(
            cell_emb.unsqueeze(0), target_centroid.unsqueeze(0)
        )
        cur_sim = cos_sim.item()
        traj.append(cur_sim)  # cos_sim on entering this step

        loss, _ = compute_loss(cell_emb, target_centroid, loss_mode="cosine")
        loss.backward()

        g = adv_emb.grad  # (k, D)
        if g is None:
            completed = False
            break

        # 2. top-k candidate tokens based on the gradient
        token_scores = -torch.matmul(g, E.T)  # (k, V)
        token_scores[:, list(SPECIAL_TOKEN_IDS)] = float("-inf")
        token_scores[:, ban_mask] = float("-inf")
        _, topk_ids = token_scores.topk(top_k_grad, dim=1)  # (k, top_k_grad)

        # 3. generate candidates with a one-position swap
        n_eval = min(batch_candidates, k_adv * top_k_grad)
        positions = rng.integers(0, k_adv, (n_eval,))
        cand_idx = rng.integers(0, top_k_grad, (n_eval,))

        cand_tokens = adv.unsqueeze(0).expand(n_eval, -1).clone()  # (n_eval, k)
        for b in range(n_eval):
            cand_tokens[b, positions[b]] = topk_ids[positions[b], cand_idx[b]]

        # remove duplicates + keep only candidates in which all k positions have distinct tokens
        cand_tokens = torch.unique(cand_tokens, dim=0)
        valid = torch.tensor(
            [len(set(row.tolist())) == k_adv for row in cand_tokens],
            dtype=torch.bool, device=cand_tokens.device,
        )
        if not valid.any():
            # No valid candidates. Do not silently evaluate invalid ones; skip this step.
            continue
        cand_tokens = cand_tokens[valid]

        # 4. batch forward evaluation
        # no gradient needed, so pass input_ids directly
        # (with inputs_embeds a (B, T, D) float tensor would have to be allocated)
        with torch.no_grad():
            b_ids, b_attn, _ = overexpress_batch(
                orig_ids_t, cand_tokens, max_seq_len
            )
            cell_embs = forward_cell_embedding(
                model, b_attn, input_ids=b_ids
            )
            sims = F.cosine_similarity(
                cell_embs, target_centroid.unsqueeze(0), dim=1
            )

        # 5. greedy acceptance: adopt only on improvement
        best_idx = sims.argmax().item()
        if sims[best_idx].item() > cur_sim:
            adv = cand_tokens[best_idx].clone()

    # ----- final GPR scoring -----
    scores, final_sim = _gpr_scores_at_position(
        model, adv, orig_ids_t, target_centroid,
        tf_token_ids, k_adv, max_seq_len, device,
        loss_mode=loss_mode,
    )
    traj.append(final_sim)  # cos_sim of the finally adopted state
    return {
        "scores": scores,          # gradient-projection score on the cosine basis
        "cos_sim": final_sim,      # cos_sim at the point where the gradient was taken (for gamma weights)
        "traj": traj,
        "completed": completed,
        "n_removed_traj": n_removed_traj,
    }


# ============================================================
# I-GPR run (all cells x n_init)
# ============================================================
def run_igpr(model, source_cells, target_centroid, tf_token_ids,
             device, k_adv=4, n_init=5, max_seq_len=MAX_SEQ_LEN,
             n_steps=20, top_k_grad=256, batch_candidates=128,
             loss_mode="angular", gammas=(0.0, 1.0)):
    """Run I-GPR on all source cells.

    The GCG trajectory is invariant as long as the loss is a monotone function of cos_sim,
    so run once on the cosine basis and reconstruct the gamma-weighted means from the same trajectory.

    Returns:
        dict:
          mean_scores      mean over all cells at the gamma corresponding to loss_mode
          all_cell_scores  per-cell means of the same
          gamma_results    {"0": {...}, "1": {...}, ...}
          cos_stats        distribution of cos_sim and lambda
          traj_mean        mean of the cos_sim trajectories of the (cell, init) that completed
          n_excluded       number excluded from the trajectory aggregate because of a break midway
          n_removed_mean   mean number of removed tokens at each step
    """
    tf_names = [n for n, _ in tf_token_ids]
    gset = sorted({float(g) for g in gammas} | {0.0, 1.0})
    acc = GammaAccumulator(tf_names, gset)

    traj_list, nrem_list = [], []
    n_excluded = 0
    expected_len = n_steps + 1

    for cell_idx, cell in enumerate(tqdm(
        source_cells, desc=f"I-GPR (step={n_steps})"
    )):
        acc.start_cell()
        for init_i in range(n_init):
            rng = np.random.default_rng(seed=cell_idx * 1000 + init_i)
            r = igpr_score_single_cell(
                model, cell, target_centroid, tf_token_ids,
                device, k_adv=k_adv, max_seq_len=max_seq_len,
                n_steps=n_steps, top_k_grad=top_k_grad,
                batch_candidates=batch_candidates, rng=rng,
                loss_mode=loss_mode,
            )
            acc.add(r["scores"], r["cos_sim"])
            if r["completed"] and len(r["traj"]) == expected_len:
                traj_list.append(r["traj"])
            else:
                n_excluded += 1
            if len(r["n_removed_traj"]) == n_steps:
                nrem_list.append(r["n_removed_traj"])
        acc.end_cell()

    gamma_results = acc.finalize()
    primary = "1" if loss_mode == "angular" else "0"

    traj_mean = (np.array(traj_list).mean(axis=0).tolist()
                 if traj_list else None)
    n_removed_mean = (np.array(nrem_list, dtype=float).mean(axis=0).tolist()
                      if nrem_list else None)

    return {
        "mean_scores": gamma_results[primary]["mean"],
        "all_cell_scores": gamma_results[primary]["per_cell"],
        "gamma_results": gamma_results,
        "cos_stats": acc.cos_sim_stats(),
        "traj_mean": traj_mean,
        "n_excluded": n_excluded,
        "n_removed_mean": n_removed_mean,
    }


# ============================================================
# (B) Validation: actually insert the top predicted TFs and measure cos_sim
# ============================================================
def _cos_sim_inserted_gf(model, cell, target_centroid, insert_ids,
                         device, max_seq_len=MAX_SEQ_LEN):
    """Insert the given tokens in the same way as the attack (official pop-and-insert) and
    return cos_sim (no gradient).

    insert_ids: list[int]  -- TF token ids to insert (an empty list = source alone).
    """
    orig_ids_t = torch.tensor(
        cell["input_ids"][:max_seq_len], dtype=torch.long, device=device,
    )
    ins_t = torch.tensor(insert_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        ids, attn, _ = overexpress_single(orig_ids_t, ins_t, max_seq_len)
        cell_emb = forward_cell_embedding(
            model, attn, input_ids=ids
        ).squeeze(0)
        cos = F.cosine_similarity(
            cell_emb.unsqueeze(0), target_centroid.unsqueeze(0)
        ).item()
    return cos


def run_verification(model, source_cells, target_centroid, ranking_scores,
                     tf_token_ids, gt_tfs, device, k_adv=4,
                     max_seq_len=MAX_SEQ_LEN, topn_list=(1, 2, 4, 8),
                     n_random=5, seed=42):
    """Aggregate, for each condition, the cos_sim when the top-ranked TFs are actually inserted.

    Conditions: source_only / random_k / gt / pred_top{N} / pred_top_gt
    For each condition compute the cos_sim of all source cells and return mean+-std.
    """
    sym2tid = {name: tid for name, tid in tf_token_ids}
    ranked = [name for name, _ in
              sorted(ranking_scores.items(), key=lambda x: x[1], reverse=True)]

    gt_in_vocab = [g for g in gt_tfs if g in sym2tid]
    n_gt = len(gt_in_vocab)

    # prepare the list of token ids to insert for each condition
    conditions = {}
    conditions["source_only"] = []          # no insertion
    for N in topn_list:
        conditions[f"pred_top{N}"] = [sym2tid[s] for s in ranked[:N]]
    if n_gt > 0:
        conditions[f"pred_top_gt({n_gt})"] = [sym2tid[s] for s in ranked[:n_gt]]
        conditions["gt"] = [sym2tid[g] for g in gt_in_vocab]

    result = {}

    # deterministic conditions (source_only, pred_*, gt)
    for cond, ids in conditions.items():
        vals = [
            _cos_sim_inserted_gf(model, cell, target_centroid, ids,
                                 device, max_seq_len)
            for cell in source_cells
        ]
        result[cond] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
        }

    # Random control: match the number of insertions to each pred condition.
    # The sequence length itself moves the mean pool, so comparing conditions with different
    # numbers of insertions mixes in a length effect. Prepare a corresponding random for each N.
    n_tf = len(tf_token_ids)
    rand_sizes = {("random_k", k_adv)}
    for N in topn_list:
        rand_sizes.add((f"random{N}", N))
    if n_gt > 0:
        rand_sizes.add((f"random_gt({n_gt})", n_gt))

    for cond_name, size in sorted(rand_sizes, key=lambda x: x[1]):
        if size <= 0 or size > n_tf:
            continue
        rng = np.random.default_rng(seed)   # draw from the same seed for each condition
        rand_vals = []
        for cell in source_cells:
            cell_vals = []
            for _ in range(n_random):
                idx = rng.choice(n_tf, size=size, replace=False)
                ids = [tf_token_ids[i][1] for i in idx]
                cell_vals.append(
                    _cos_sim_inserted_gf(model, cell, target_centroid, ids,
                                         device, max_seq_len)
                )
            rand_vals.append(np.mean(cell_vals))
        result[cond_name] = {
            "mean": float(np.mean(rand_vals)),
            "std": float(np.std(rand_vals)),
            "n_inserted": int(size),
        }

    # recording the number of insertions makes later analysis need no matching
    for N in topn_list:
        if f"pred_top{N}" in result:
            result[f"pred_top{N}"]["n_inserted"] = int(N)
    if n_gt > 0:
        result[f"pred_top_gt({n_gt})"]["n_inserted"] = int(n_gt)
        result["gt"]["n_inserted"] = int(n_gt)
    result["source_only"]["n_inserted"] = 0

    return result


def _print_verification(cos_res):
    """Print the validation results lined up by number of insertions.

    Putting the same numbers of insertions vertically side by side lets one follow, by eye,
    the comparison "prediction vs random vs GT" free of the sequence-length effect.
    """
    base = cos_res.get("source_only", {}).get("mean")

    def n_ins(k):
        return cos_res[k].get("n_inserted", -1)

    groups = {}
    for k in cos_res:
        if k == "source_only":
            continue
        groups.setdefault(n_ins(k), []).append(k)

    if base is not None:
        r = cos_res["source_only"]
        print(f"    {'source_only':18s} (n=0): cos_sim={r['mean']:.4f} "
              f"± {r['std']:.4f}")

    def sort_key(k):
        return (0 if k.startswith("random") else
                1 if k.startswith("pred") else 2, k)

    for n in sorted(groups):
        for k in sorted(groups[n], key=sort_key):
            r = cos_res[k]
            d = f"  Δ={r['mean'] - base:+.5f}" if base is not None else ""
            print(f"    {k:18s} (n={n}): cos_sim={r['mean']:.4f} "
                  f"± {r['std']:.4f}{d}")


# ============================================================
# Main: evaluation across 4 pathways
# ============================================================
# ============================================================
# Script version
# ============================================================
# Printed at startup. Before launching a long run, one can check by eye that it is the intended version.
SCRIPT_VERSION = "2026-07-28"
SCRIPT_FEATURES = [
    "overexpress=pop_and_insert (follows the official ISP)",
    "emb_layer switch (0=last layer / -1=second-to-last)",
    "gamma aggregation (reconstruct cosine/angular from one run)",
    "geometric diagnostics (separation and anisotropy per layer)",
    "auc_cosproj (cosine projection against norm bias)",
    "auc_vs_other_gt (pathway specificity)",
    "save_scores (store whole scores for post-hoc analysis)",
    "random_N (random control with matched numbers of insertions in the validation pass)",
    "per-pathway incremental saving",
]


def print_version_banner():
    print("=" * 60)
    print(f"  gpr_geneformer_multi_pathway.py  version {SCRIPT_VERSION}")
    print("=" * 60)
    for f in SCRIPT_FEATURES:
        print(f"    - {f}")
    print("=" * 60)


def _eval_all(scores, gt_tfs, tf_names, tf_norms, other_gt):
    """For one score dictionary, return the ordinary AUC, the cosine-projection AUC and the pathway-specific AUC."""
    auc, ranks = evaluate_auc(scores, gt_tfs, tf_names)
    cos_scores = normalize_by_norm(scores, tf_norms)
    auc_cos, _ = evaluate_auc(cos_scores, gt_tfs, tf_names)
    return {
        "auc": float(auc),
        "ranks": ranks,
        "auc_cosproj": float(auc_cos),
        "auc_vs_other_gt": float(auc_vs_other_gt(scores, gt_tfs, other_gt)),
        "auc_vs_other_gt_cosproj": float(
            auc_vs_other_gt(cos_scores, gt_tfs, other_gt)),
    }


def _save_results(args, all_results, outfile, n_tfs=None):
    """Write the results to JSON. Called after each pathway,
    so the results up to that point always survive an exception midway."""
    save_data = {
        "config": {
            "model": "Geneformer_V1_10M",
            "k_adv": args.k_adv,
            "n_cells": args.n_cells,
            "n_init": args.n_init,
            "n_target_cells": args.n_target_cells,
            "max_seq_len": args.max_seq_len,
            "igpr_steps": args.igpr_steps,
            "top_k_grad": args.top_k_grad,
            "batch_candidates": args.batch_candidates,
            "n_tfs": n_tfs,
            "seed": args.seed,
            "verify": args.verify,
            "verify_topn": args.verify_topn,
            "n_random": args.n_random,
            "loss_mode": args.loss_mode,
            # pop_and_insert = follows the official InSilicoPerturber.
            # prepend_only  = prepend without removal (duplicate tokens; for A/B).
            "overexpress_mode": args.overexpress_mode,
            "emb_layer": args.emb_layer,
            "gammas": args.gammas,
        },
        "results": all_results,
    }
    tmp = outfile + ".tmp"
    with open(tmp, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, outfile)   # do not leave a half-written file


def main():
    parser = argparse.ArgumentParser(
        description="GPR/I-GPR multi-pathway evaluation for Geneformer V1"
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Geneformer V1 checkpoint directory")
    parser.add_argument("--pathway_config", type=str, required=True,
                        help="Path to pathway_config_geneformer.json")
    parser.add_argument("--dict_dir", type=str, required=True,
                        help="Gene dictionaries directory")
    parser.add_argument("--tf_list", type=str, required=True,
                        help="TF list file")
    parser.add_argument("--pathways", nargs="+",
                        default=["all"])
    parser.add_argument("--k_adv", type=int, default=4)
    parser.add_argument("--n_cells", type=int, default=50)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--n_target_cells", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--igpr_steps", nargs="*", type=int, default=[],
                        help="I-GPR step counts (e.g. 5 10 20 50 100 200). "
                             "Empty=GPR only.")
    parser.add_argument("--top_k_grad", type=int, default=256,
                        help="Gradient candidates per position (GCG)")
    parser.add_argument("--batch_candidates", type=int, default=128,
                        help="Candidates to evaluate per GCG step")
    parser.add_argument("--output_dir", type=str,
                        default="./gpr_geneformer_results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--loss_mode", type=str, default="angular",
                        choices=["angular", "cosine"],
                        help="Loss function: angular (recommended) or cosine")
    parser.add_argument("--save_scores", action="store_true", default=True,
                        help="save the whole aggregated score vector to JSON. "
                             "Post-hoc analysis is possible without rerunning.")
    parser.add_argument("--no_save_scores", dest="save_scores",
                        action="store_false")
    parser.add_argument("--gammas", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 1.5, 2.0],
                        help="gamma of the aggregation weight w=(1-c^2)^(-gamma/2). "
                             "gamma=0 is cosine, gamma=1 is angular. "
                             "All are reconstructed from a single run.")
    parser.add_argument("--geometry_diag", action="store_true", default=True,
                        help="run the geometric diagnostics per layer (separation, anisotropy)")
    parser.add_argument("--no_geometry_diag", dest="geometry_diag",
                        action="store_false")
    parser.add_argument("--emb_layer", type=int, default=0, choices=[0, -1],
                        help="layer from which embeddings are taken. 0=last layer (current), "
                             "-1=second to last (default of the official EmbExtractor, follows the paper). "
                             "Aligned automatically between the centroid and the perturbed embedding.")
    parser.add_argument("--overexpress_mode", type=str, default="pop_and_insert",
                        choices=["pop_and_insert", "prepend_only"],
                        help="pop_and_insert: follows the official ISP (default). "
                             "prepend_only: prepend without removal (duplicate tokens arise). "
                             "For A/B comparison.")
    # (B) validation pass
    parser.add_argument("--verify", action="store_true",
                        help="enable the validation that actually inserts the top predicted TFs and measures cos_sim")
    parser.add_argument("--verify_topn", nargs="+", type=int,
                        default=[1, 2, 4, 8],
                        help="list of the numbers of top TFs inserted in the validation")
    parser.add_argument("--n_random", type=int, default=5,
                        help="number of samples of the random_k baseline")
    args = parser.parse_args()

    print_version_banner()
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

    # model, dictionary, TFs
    print("Loading Geneformer V1 model...")
    model, model_config = load_geneformer_model(args.model_dir, device)

    # load from pathway_config
    pw_config, dataset_paths = load_pathway_config(args.pathway_config, model="geneformer")
    pw_ids = get_pathway_ids(pw_config, None if args.pathways == ["all"] else args.pathways)
    print(f"Pathways to run: {pw_ids}")

    print(f"Loading dictionaries: {args.dict_dir}")
    token_dict, name_dict, id_to_sym, sym_to_id = load_dicts(args.dict_dir)

    tf_token_ids = load_tf_list(args.tf_list, token_dict, name_dict)

    # Countermeasure to norm bias: s_j = -g.E[j] inflates in proportion to ||E[j]||.
    # The norm-only baseline, which just ranks ||E[j]||, gives an AUC of 0.76,
    # so evaluate also with the cosine projection s_j/||E[j]|| (zero extra computation).
    _E = model.embeddings.word_embeddings.weight.detach()
    tf_norms = {name: float(_E[tid].norm().item())
                for name, tid in tf_token_ids}
    print(f"TFs in vocab: {len(tf_token_ids)}")
    tf_names = [name for name, _ in tf_token_ids]

    # evaluate on all pathways
    outfile = os.path.join(
        args.output_dir,
        f"multi_pathway_geneformer_k{args.k_adv}_n{args.n_cells}.json",
    )
    # "GT TFs of other pathways" used for the evaluation of pathway specificity
    gt_union = {}
    for _p in pw_ids:
        gt_union[_p] = set(pw_config[_p]["gt_tfs"])

    all_results = {}
    summary_lines = []

    for pw_name in pw_ids:
        pw = pw_config[pw_name]
        dataset = load_from_disk(dataset_paths[pw_name])
        print(f"  Loaded dataset: {dataset_paths[pw_name]} ({len(dataset)} cells)")
        print(f"\n{'='*60}")
        print(f"Pathway: {pw_name}  ({pw['source_type']} -> {pw['target_type']})")
        print(f"GT TFs: {pw['gt_tfs']}")
        print(f"{'='*60}")

        # data preparation
        source_cells = prepare_cell_data(
            dataset, pw["source_type"],
            n_cells=args.n_cells, seed=args.seed,
        )
        target_centroid = compute_target_centroid(
            model, dataset, pw["target_type"],
            n_cells=args.n_target_cells, device=device,
            max_seq_len=args.max_seq_len,
        )

        pw_results = {}

        # ---- geometric diagnostics (per layer; cheap since no gradient is needed) ----
        if args.geometry_diag:
            print("\nGeometry diagnostics (both layers)...")
            target_cells = prepare_cell_data(
                dataset, pw["target_type"],
                n_cells=min(args.n_target_cells, 100), seed=args.seed,
            )
            saved_layer = get_emb_layer()
            geo = {}
            for lay, lname in ((0, "last"), (-1, "second_to_last")):
                set_emb_layer(lay)

                def _embed(cell, _ml=args.max_seq_len):
                    ids = torch.tensor(
                        cell["input_ids"][:_ml], dtype=torch.long, device=device
                    ).unsqueeze(0)
                    attn = (ids != PAD_TOKEN_ID).long()
                    return forward_cell_embedding(
                        model, attn, input_ids=ids
                    ).squeeze(0)

                geo[lname] = geometry_diagnostics(
                    _embed, source_cells, target_cells
                )
            set_emb_layer(saved_layer)
            pw_results["geometry"] = geo
            print(format_geometry(geo))

        # ---- GPR (step 0) ----
        print(f"\nRunning GPR (step 0, loss={args.loss_mode})...")
        gpr_nrem = []
        mean_scores, all_cell_scores, gpr_gamma, gpr_cos = run_gpr(
            model, source_cells, target_centroid, tf_token_ids,
            device, k_adv=args.k_adv, n_init=args.n_init,
            max_seq_len=args.max_seq_len, loss_mode=args.loss_mode,
            stats_out=gpr_nrem, gammas=args.gammas,
        )
        other_gt = sorted(set().union(
            *[g for q, g in gt_union.items() if q != pw_name]
        )) if len(gt_union) > 1 else []
        ev = _eval_all(mean_scores, pw["gt_tfs"], tf_names, tf_norms, other_gt)
        auc, ranks = ev["auc"], ev["ranks"]

        # per-cell AUC
        per_cell_aucs = []
        for cs in all_cell_scores:
            a, _ = evaluate_auc(cs, pw["gt_tfs"], tf_names)
            per_cell_aucs.append(a)
        valid_aucs = [a for a in per_cell_aucs if not np.isnan(a)]

        nrem_mean = float(np.mean(gpr_nrem)) if gpr_nrem else None
        pw_results["step_0"] = {
            "auc": float(auc),
            "ranks": ranks,
            "per_cell_auc_mean": float(np.mean(valid_aucs)) if valid_aucs else None,
            "per_cell_auc_std": float(np.std(valid_aucs)) if valid_aucs else None,
            "n_removed_mean": nrem_mean,
            "cos_stats": gpr_cos,
            "auc_cosproj": ev["auc_cosproj"],
            "auc_vs_other_gt": ev["auc_vs_other_gt"],
            "auc_vs_other_gt_cosproj": ev["auc_vs_other_gt_cosproj"],
            "gamma_auc": {
                g: float(evaluate_auc(v["mean"], pw["gt_tfs"], tf_names)[0])
                for g, v in gpr_gamma.items()
            },
            "mean_scores": ({n: round(float(x), 6)
                             for n, x in mean_scores.items()}
                            if args.save_scores else None),
        }
        line = (f"{pw_name:10s}  GPR(step=0):  AUC={auc:.3f}"
                f"  cosproj={ev['auc_cosproj']:.3f}"
                f"  vs_otherGT={ev['auc_vs_other_gt']:.3f}")
        print(line)
        summary_lines.append(line)
        if nrem_mean is not None:
            print(f"  [overexpress] n_removed={nrem_mean:.3f}/{args.k_adv} "
                  f"(fraction of inserted TFs that were already in the original sequence={nrem_mean/args.k_adv:.1%})")

        # (B) Validation: actually insert the top predicted TFs and measure cos_sim
        if args.verify:
            print("  [verify] GPR ranking cos_sim:")
            cos_res = run_verification(
                model, source_cells, target_centroid, mean_scores,
                tf_token_ids, pw["gt_tfs"], device,
                k_adv=args.k_adv, max_seq_len=args.max_seq_len,
                topn_list=tuple(args.verify_topn),
                n_random=args.n_random, seed=args.seed,
            )
            pw_results["step_0"]["cos_sim"] = cos_res
            _print_verification(cos_res)

        # show Top-10
        ranked = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
        gt_set = set(pw["gt_tfs"])
        print("  Top-10:", end="")
        for i, (name, score) in enumerate(ranked[:10]):
            mark = "*" if name in gt_set else ""
            print(f"  {name}{mark}", end="")
        print()

        # ---- I-GPR (for each number of steps) ----
        for n_steps in args.igpr_steps:
            print(f"\nRunning I-GPR (step={n_steps}, loss={args.loss_mode})...")
            R = run_igpr(
                model, source_cells, target_centroid, tf_token_ids,
                device, k_adv=args.k_adv, n_init=args.n_init,
                max_seq_len=args.max_seq_len,
                n_steps=n_steps, top_k_grad=args.top_k_grad,
                batch_candidates=args.batch_candidates,
                loss_mode=args.loss_mode, gammas=args.gammas,
            )
            mean_scores_i = R["mean_scores"]
            all_cell_scores_i = R["all_cell_scores"]
            traj_mean = R["traj_mean"]
            n_excluded = R["n_excluded"]
            nrem_traj = R["n_removed_mean"]
            ev_i = _eval_all(mean_scores_i, pw["gt_tfs"], tf_names,
                             tf_norms, other_gt)
            auc_i, ranks_i = ev_i["auc"], ev_i["ranks"]

            per_cell_aucs_i = []
            for cs in all_cell_scores_i:
                a, _ = evaluate_auc(cs, pw["gt_tfs"], tf_names)
                per_cell_aucs_i.append(a)
            valid_aucs_i = [a for a in per_cell_aucs_i if not np.isnan(a)]

            pw_results[f"step_{n_steps}"] = {
                "auc": float(auc_i),
                "ranks": ranks_i,
                "per_cell_auc_mean": (
                    float(np.mean(valid_aucs_i)) if valid_aucs_i else None
                ),
                "per_cell_auc_std": (
                    float(np.std(valid_aucs_i)) if valid_aucs_i else None
                ),
                # (A) cos_sim trajectory during optimization (on entering each step x n_steps + final)
                "cos_sim_trajectory": traj_mean,
                "cos_sim_traj_excluded": n_excluded,
                # mean number of tokens removed from the original sequence at each step
                "n_removed_trajectory": nrem_traj,
                "cos_stats": R["cos_stats"],
                "auc_cosproj": ev_i["auc_cosproj"],
                "auc_vs_other_gt": ev_i["auc_vs_other_gt"],
                "auc_vs_other_gt_cosproj": ev_i["auc_vs_other_gt_cosproj"],
                "gamma_auc": {
                    g: float(evaluate_auc(v["mean"], pw["gt_tfs"], tf_names)[0])
                    for g, v in R["gamma_results"].items()
                },
                "mean_scores": ({n: round(float(x), 6)
                                 for n, x in mean_scores_i.items()}
                                if args.save_scores else None),
            }
            line = (f"{pw_name:10s}  I-GPR(step={n_steps:3d}): AUC={auc_i:.3f}"
                    f"  cosproj={ev_i['auc_cosproj']:.3f}"
                    f"  vs_otherGT={ev_i['auc_vs_other_gt']:.3f}")
            print(line)
            summary_lines.append(line)

            # (A) display the key points of the trajectory
            if traj_mean is not None:
                print(f"  [traj] cos_sim: init={traj_mean[0]:.4f} "
                      f"-> final={traj_mean[-1]:.4f} "
                      f"(Δ={traj_mean[-1] - traj_mean[0]:+.4f}, "
                      f"excluded={n_excluded})")

            # diagnosis of exploitation of the duplication artifact
            if nrem_traj is not None:
                print(f"  [overexpress] n_removed: init={nrem_traj[0]:.3f} "
                      f"-> final={nrem_traj[-1]:.3f} /{args.k_adv} "
                      f"(Δ={nrem_traj[-1] - nrem_traj[0]:+.3f})")

            # (B) Validation: actually insert the top predicted TFs and measure cos_sim
            if args.verify:
                print(f"  [verify] I-GPR(step={n_steps}) ranking cos_sim:")
                cos_res = run_verification(
                    model, source_cells, target_centroid, mean_scores_i,
                    tf_token_ids, pw["gt_tfs"], device,
                    k_adv=args.k_adv, max_seq_len=args.max_seq_len,
                    topn_list=tuple(args.verify_topn),
                    n_random=args.n_random, seed=args.seed,
                )
                pw_results[f"step_{n_steps}"]["cos_sim"] = cos_res
                _print_verification(cos_res)

        all_results[pw_name] = pw_results

        # Save after each pathway finishes.
        # Even if a later step fails, the results so far always remain on disk.
        _save_results(args, all_results, outfile, len(tf_token_ids))
        print(f"  [saved] {outfile} ({len(all_results)} pathways)")

    # summary display
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for line in summary_lines:
        print(line)

    # ---- Norm bias and pathway specificity ----
    print(f"\n{'='*76}")
    print("  Norm bias and pathway specificity")
    print("  auc        : ordinary (own-pathway GT vs all TFs)")
    print("  cosproj    : cosine projection, the score divided by ||E[j]||")
    print("  vs_otherGT : negatives replaced by the GT of other pathways (measures pathway specificity only)")
    print(f"{'='*76}")
    for key in sorted({k for r in all_results.values() for k in r
                       if k.startswith("step_")},
                      key=lambda k: int(k.split("_")[1])):
        name = "GPR" if key == "step_0" else f"I-GPR({key.split('_')[1]})"
        rows = [(p_, r[key]) for p_, r in sorted(all_results.items())
                if key in r and "auc_cosproj" in r[key]]
        if not rows:
            continue
        print(f"\n  [{name}]")
        print(f"    {'Pathway':10s} {'auc':>8s} {'cosproj':>9s} "
              f"{'Delta':>8s} {'vs_otherGT':>11s} {'same cosproj':>10s}")
        print("    " + "-" * 60)
        for p_, r in rows:
            print(f"    {p_:10s} {r['auc']:8.3f} {r['auc_cosproj']:9.3f} "
                  f"{r['auc_cosproj'] - r['auc']:+8.3f} "
                  f"{r['auc_vs_other_gt']:11.3f} "
                  f"{r['auc_vs_other_gt_cosproj']:10.3f}")
        print("    " + "-" * 60)
        for lab, f in (("mean", np.mean),):
            print(f"    {lab:10s} "
                  f"{f([r['auc'] for _, r in rows]):8.3f} "
                  f"{f([r['auc_cosproj'] for _, r in rows]):9.3f} "
                  f"{f([r['auc_cosproj'] - r['auc'] for _, r in rows]):+8.3f} "
                  f"{f([r['auc_vs_other_gt'] for _, r in rows]):11.3f} "
                  f"{f([r['auc_vs_other_gt_cosproj'] for _, r in rows]):10.3f}")
        m_cos = np.mean([r["auc_cosproj"] for _, r in rows])
        m_raw = np.mean([r["auc"] for _, r in rows])
        m_spec = np.nanmean([r["auc_vs_other_gt"] for _, r in rows])
        if abs(m_cos - m_raw) < 0.02:
            print("    -> The AUC is preserved with the cosine projection too. "
                  "The norm does not drive the result.")
        elif m_cos < m_raw - 0.02:
            print(f"    -> Down by {m_raw - m_cos:.3f} with the cosine projection. "
                  "The norm contributes to the result.")
        else:
            print(f"    -> Up by {m_cos - m_raw:.3f} with the cosine projection. "
                  "The norm was a harmful confounder.")
        if m_spec == m_spec:
            if m_spec < 0.55:
                print("    -> vs_otherGT is near 0.5. The GT of the own pathway cannot be")
                print("       told from that of other pathways. Pathway specificity is poor.")
            else:
                print(f"    -> vs_otherGT = {m_spec:.3f}. "
                      "There is a pathway-specific signal.")

    # ---- gamma sweep (loss ablation) ----
    gam_rows = {}
    for pw_name, pw_res in all_results.items():
        for key, val in pw_res.items():
            if not key.startswith("step_"):
                continue
            ga = val.get("gamma_auc")
            if not ga:
                continue
            gam_rows.setdefault(key, {})[pw_name] = ga

    if gam_rows:
        print(f"\n{'='*72}")
        print("  gamma sweep  w=(1-c^2)^(-gamma/2)   gamma=0:cosine  gamma=1:angular")
        print("  Note: reconstructed from the same GCG trajectory. The search does not depend on gamma.")
        print(f"{'='*72}")
        for key in sorted(gam_rows, key=lambda k: int(k.split("_")[1])):
            per_pw = gam_rows[key]
            gs = sorted(next(iter(per_pw.values())).keys(), key=float)
            name = "GPR" if key == "step_0" else f"I-GPR({key.split('_')[1]})"
            print(f"\n  [{name}]")
            print("    " + f"{'Pathway':10s}" +
                  "".join(f"{'γ='+g:>10s}" for g in gs))
            print("    " + "-" * (10 + 10 * len(gs)))
            for pw_name in sorted(per_pw):
                row = per_pw[pw_name]
                print("    " + f"{pw_name:10s}" +
                      "".join(f"{row[g]:10.4f}" for g in gs))
            means = {g: float(np.mean([per_pw[p_][g] for p_ in per_pw]))
                     for g in gs}
            print("    " + "-" * (10 + 10 * len(gs)))
            print("    " + f"{'mean':10s}" +
                  "".join(f"{means[g]:10.4f}" for g in gs))
            best = max(means, key=lambda g: means[g])
            spread = max(means.values()) - min(means.values())
            print(f"    best gamma={best} (mean {means[best]:.4f}), "
                  f"spread across gamma {spread:.4f}")
            if spread < 0.005:
                print("    -> The choice of gamma has almost no effect on the result.")
                print("       The difference between angular / cosine is only the aggregation weight,")
                print("       and it can be reported that the effect is negligible.")

    _save_results(args, all_results, outfile, len(tf_token_ids))
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
