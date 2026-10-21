#!/usr/bin/env python3
"""
Test of the additivity of the objective function
======================
Combinatorial search (GCG/PGD) is meaningful only if the objective is non-additive.
If it is additive, "pick the top k single effects" gives the optimum and no search is needed.

Ahlmann-Eltze et al. (Nat Methods 2025) showed that, in perturbation-effect prediction,
deep learning models do not outperform a simple additive model.
This script tests the same question in the context of search with the embedding shift as the objective.

What is measured:
  1. Delta(j) = cos_sim(cell+j) - cos_sim(cell) when each TF is inserted alone
     (forward only; no gradient needed)
  2. For a set S, compare the measured Delta(S) with the additive prediction sum_{j in S} Delta(j)
     - the set of the top k by additivity
     - a random set of k
  3. auc / spec of the additive ranking (ordered by Delta(j))
     -> may be stronger than the gradient-projection ranking

Decision:
  measured ~ additive prediction  -> the objective is additive. Combinatorial search is unnecessary.
  measured >  additive prediction  -> non-additive. There is room for search to exploit.

Usage:
  python analyze_additivity.py \\
      --model_dir ./Geneformer/Geneformer-V1-10M \\
      --pathway_config ./data/pathway_config_geneformer.json \\
      --dict_dir ./Geneformer/geneformer/gene_dictionaries_30m \\
      --tf_list ./tf_list_human.txt \\
      --pathways P01 P02 P03 P04 \\
      --n_cells 10 --ks 2 4 8 --n_random 200 \\
      --output_dir ./additivity_results
"""

import argparse
import json
import os
import statistics as st
import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advcell_pathway_loader import load_pathway_config, get_pathway_ids
from advcell_overexpress import overexpress_batch, set_overexpress_mode
from gpr_geneformer import (
    MAX_SEQ_LEN,
    load_geneformer_model,
    load_dicts,
    load_tf_list,
    prepare_cell_data,
    compute_target_centroid,
    forward_cell_embedding,
    set_emb_layer,
    evaluate_auc,
)


@torch.no_grad()
def cos_sim_for_sets(model, orig_ids_t, sets, target_centroid,
                     max_seq_len, batch_size=64):
    """Compute cos_sim for many TF sets at once.

    Args:
        sets: (N, k) long tensor. Each row is the set of TF token ids to insert.
              Treat k=0 (the empty set) separately as the baseline.
    Returns:
        (N,) tensor
    """
    out = []
    for i in range(0, sets.shape[0], batch_size):
        chunk = sets[i:i + batch_size]
        ids, attn, _ = overexpress_batch(orig_ids_t, chunk, max_seq_len)
        emb = forward_cell_embedding(model, attn, input_ids=ids)
        out.append(F.cosine_similarity(
            emb, target_centroid.unsqueeze(0), dim=1
        ))
    return torch.cat(out)


@torch.no_grad()
def baseline_cos(model, orig_ids_t, target_centroid, max_seq_len):
    ids = orig_ids_t[:max_seq_len].unsqueeze(0)
    attn = torch.ones_like(ids)
    emb = forward_cell_embedding(model, attn, input_ids=ids)
    return float(F.cosine_similarity(
        emb, target_centroid.unsqueeze(0), dim=1
    ).item())


@torch.no_grad()
def greedy_forward_selection(model, orig_ids_t, tf_ids, target_centroid,
                             k, max_seq_len, batch_size, base):
    """Greedy forward selection. One at a time, add the TF that raises cos_sim the most.

    Cost = k x n_TF forwards. Higher than GCG (n_steps x batch).
    A stronger and practically natural baseline than the additive model.

    Returns: (list of chosen token ids, cos_sim at each stage)
    """
    n_tf = tf_ids.numel()
    chosen, traj = [], []
    for _ in range(k):
        remain = [i for i in range(n_tf) if i not in chosen]
        cand = torch.tensor(
            [chosen + [i] for i in remain], dtype=torch.long,
            device=tf_ids.device
        )
        cs = cos_sim_for_sets(model, orig_ids_t, tf_ids[cand],
                              target_centroid, max_seq_len, batch_size)
        best = int(torch.argmax(cs).item())
        chosen.append(remain[best])
        traj.append(float(cs[best].item()) - base)
    return chosen, traj

def analyze_pathway(model, dataset, pw, tf_token_ids, device, args, rng):
    tf_ids = torch.tensor([t for _, t in tf_token_ids],
                          dtype=torch.long, device=device)
    tf_names = [n for n, _ in tf_token_ids]
    n_tf = len(tf_names)

    source_cells = prepare_cell_data(
        dataset, pw["source_type"], n_cells=args.n_cells, seed=args.seed
    )
    target_centroid = compute_target_centroid(
        model, dataset, pw["target_type"], n_cells=args.n_target_cells,
        device=device, max_seq_len=args.max_seq_len,
    )

    # ---- 1. Single effect Delta(j) of every TF in each cell ----
    deltas = np.zeros((len(source_cells), n_tf), dtype=np.float64)
    base_cos = []
    for ci, cell in enumerate(source_cells):
        orig = torch.tensor(cell["input_ids"][:args.max_seq_len],
                            dtype=torch.long, device=device)
        b = baseline_cos(model, orig, target_centroid, args.max_seq_len)
        base_cos.append(b)
        singles = tf_ids.view(-1, 1)                   # (n_tf, 1)
        cs = cos_sim_for_sets(model, orig, singles, target_centroid,
                              args.max_seq_len, args.batch_size)
        deltas[ci] = (cs - b).cpu().numpy()

    mean_delta = deltas.mean(axis=0)                   # (n_tf,)
    order = np.argsort(-mean_delta)

    # ---- 2. Measured vs additive prediction of sets ----
    set_rows = []
    for k in args.ks:
        if k > n_tf:
            continue
        # (a) top k by additivity
        top_idx = order[:k]
        # (b) random k
        rand_idx = [rng.choice(n_tf, size=k, replace=False)
                    for _ in range(args.n_random)]

        cand = [("additive_top", top_idx)] + \
               [("random", ri) for ri in rand_idx]

        idx_mat = torch.tensor(np.stack([c[1] for c in cand]),
                               dtype=torch.long, device=device)
        tok_mat = tf_ids[idx_mat]                      # (N, k)

        for ci, cell in enumerate(source_cells):
            orig = torch.tensor(cell["input_ids"][:args.max_seq_len],
                                dtype=torch.long, device=device)
            actual = cos_sim_for_sets(
                model, orig, tok_mat, target_centroid,
                args.max_seq_len, args.batch_size
            ).cpu().numpy() - base_cos[ci]
            pred = np.stack([deltas[ci][c[1]].sum() for c in cand])
            for (kind, _), a, p in zip(cand, actual, pred):
                set_rows.append({"k": k, "kind": kind, "cell": ci,
                                 "actual": float(a), "additive": float(p)})

    # ---- 2b. Greedy forward selection ----
    greedy_rows = []
    for k in args.greedy_ks:
        if k <= 0 or k > n_tf:
            continue
        for ci, cell in enumerate(source_cells):
            orig = torch.tensor(cell["input_ids"][:args.max_seq_len],
                                dtype=torch.long, device=device)
            idx, traj = greedy_forward_selection(
                model, orig, tf_ids, target_centroid, k,
                args.max_seq_len, args.batch_size, base_cos[ci]
            )
            greedy_rows.append({
                "k": k, "cell": ci, "actual": traj[-1],
                "tfs": [tf_names[i] for i in idx],
                "additive": float(deltas[ci][idx].sum()),
            })

    # ---- 3. Evaluation of the additive ranking ----
    add_scores = {n: float(v) for n, v in zip(tf_names, mean_delta)}
    auc, ranks = evaluate_auc(add_scores, pw["gt_tfs"], tf_names)

    return {
        "n_cells": len(source_cells),
        "base_cos_mean": float(np.mean(base_cos)),
        "delta_mean": float(mean_delta.mean()),
        "delta_max": float(mean_delta.max()),
        "delta_min": float(mean_delta.min()),
        "additive_ranking_auc": float(auc),
        "additive_ranking_ranks": ranks,
        "additive_ranking_scores": {n: round(v, 8)
                                    for n, v in add_scores.items()},
        "set_rows": set_rows,
        "greedy_rows": greedy_rows,
    }


def _fit(a, p):
    """Least squares of actual ~ slope * additive + intercept.

    If additive: slope = 1, intercept = 0, residual ~ 0.
    The ratio mean(a)/mean(p) is not used because it diverges when the denominator is near zero.
    """
    n = len(a)
    if n < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")
    ma, mp = st.mean(a), st.mean(p)
    sxy = sum((x - mp) * (y - ma) for x, y in zip(p, a))
    sxx = sum((x - mp) ** 2 for x in p)
    slope = sxy / sxx if sxx else float("nan")
    intercept = ma - slope * mp if slope == slope else float("nan")
    syy = sum((y - ma) ** 2 for y in a)
    r = sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float("nan")
    resid = st.mean([x - y for x, y in zip(a, p)])   # measured - additive prediction
    return slope, intercept, r, resid


def report(name, rows):
    """Summarize the additivity of one pathway."""
    print(f"\n  {'k':>3s} {'condition':>14s} {'n':>5s} {'measured':>11s} "
          f"{'additive':>11s} {'meas-pred':>11s} {'slope':>7s} {'corr':>7s}")
    print("  " + "-" * 76)
    out = []
    for k in sorted({r["k"] for r in rows}):
        for kind in ("additive_top", "random"):
            sub = [r for r in rows if r["k"] == k and r["kind"] == kind]
            if not sub:
                continue
            a = [r["actual"] for r in sub]
            p = [r["additive"] for r in sub]
            slope, icpt, r_p, resid = _fit(a, p)
            print(f"  {k:3d} {kind:>14s} {len(sub):5d} {st.mean(a):11.6f} "
                  f"{st.mean(p):11.6f} {resid:+11.6f} {slope:7.3f} "
                  f"{r_p:7.3f}")
            out.append({"k": k, "kind": kind, "n": len(sub),
                        "actual": st.mean(a), "additive": st.mean(p),
                        "resid": resid, "slope": slope, "intercept": icpt,
                        "pearson": r_p})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--pathway_config", required=True)
    ap.add_argument("--dict_dir", required=True)
    ap.add_argument("--tf_list", required=True)
    ap.add_argument("--pathways", nargs="+", default=["all"])
    ap.add_argument("--n_cells", type=int, default=10)
    ap.add_argument("--n_target_cells", type=int, default=200)
    ap.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--n_random", type=int, default=200)
    ap.add_argument("--greedy_ks", nargs="*", type=int, default=[],
                    help="List of k for which to run greedy forward selection (e.g. 4). "
                         "Disabled by default because the cost is high: k x n_TF x n_cells forwards.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--emb_layer", type=int, default=0, choices=[0, -1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default="./additivity_results")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_overexpress_mode("pop_and_insert")
    set_emb_layer(args.emb_layer)

    model, _ = load_geneformer_model(args.model_dir, device)
    token_dict, name_dict, _, _ = load_dicts(args.dict_dir)
    tf_token_ids = load_tf_list(args.tf_list, token_dict, name_dict)
    print(f"TFs in vocab: {len(tf_token_ids)}")

    pw_config, data_paths = load_pathway_config(
        args.pathway_config, model="geneformer"
    )
    pw_ids = get_pathway_ids(
        pw_config, None if args.pathways == ["all"] else args.pathways
    )
    print(f"Pathways: {pw_ids}")

    rng = np.random.default_rng(args.seed)
    all_res, summary = {}, []
    outfile = os.path.join(args.output_dir, "additivity.json")

    def _save():
        tmp = outfile + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"config": vars(args), "results": all_res},
                      f, indent=2, ensure_ascii=False)
        os.replace(tmp, outfile)

    for pw_name in pw_ids:
        pw = pw_config[pw_name]
        print(f"\n{'='*72}")
        print(f"Pathway: {pw_name}  ({pw['source_type']} -> {pw['target_type']})")
        print(f"{'='*72}")
        dataset = load_from_disk(data_paths[pw_name])
        res = analyze_pathway(model, dataset, pw, tf_token_ids,
                              device, args, rng)
        print(f"  baseline cos_sim = {res['base_cos_mean']:.5f}")
        print(f"  Single effect Delta(j): mean={res['delta_mean']:+.6f} "
              f"max={res['delta_max']:+.6f} min={res['delta_min']:+.6f}")
        print(f"  AUC of the additive ranking = {res['additive_ranking_auc']:.3f}")
        rows = report(pw_name, res["set_rows"])
        res["summary"] = rows
        if res.get("greedy_rows"):
            for k in sorted({g["k"] for g in res["greedy_rows"]}):
                gg = [g for g in res["greedy_rows"] if g["k"] == k]
                at = [x for x in rows
                      if x["k"] == k and x["kind"] == "additive_top"]
                msg = (f"  greedy(k={k}): measured {st.mean([g['actual'] for g in gg]):+.6f}")
                if at:
                    msg += f"   additive top{k}: {at[0]['actual']:+.6f}"
                print(msg)

        all_res[pw_name] = res
        summary.append((pw_name, rows))
        _save()
        print(f"  [saved] {outfile} ({len(all_res)} pathways)")
        del dataset

    # ---- Verdict over all pathways ----
    print(f"\n{'='*72}")
    print("  Verdict: is the objective additive?")
    print(f"{'='*72}")
    print("  If additive, slope = 1.00 and measured - predicted = 0\n")
    for kind in ("additive_top", "random"):
        for k in args.ks:
            sub = [r for _, rows in summary for r in rows
                   if r["k"] == k and r["kind"] == kind]
            sl = [r["slope"] for r in sub if r["slope"] == r["slope"]]
            rs = [r["resid"] for r in sub if r["resid"] == r["resid"]]
            ac = [r["actual"] for r in sub]
            if not rs:
                continue
            print(f"  k={k:2d} {kind:>14s}: measured - predicted = {st.mean(rs):+.6f}"
                  f"  (of the measured Delta {st.mean(ac):+.6f}: "
                  f"{abs(st.mean(rs)) / abs(st.mean(ac)) * 100 if st.mean(ac) else float('nan'):.0f}%)"
                  + (f"   slope = {st.mean(sl):.3f}" if sl else ""))

    # The verdict is based on the regression over random sets (wide distribution, stable estimate)
    sub = [r for _, rows in summary for r in rows if r["kind"] == "random"]
    sl = [r["slope"] for r in sub if r["slope"] == r["slope"]]
    rs = [r["resid"] for r in sub if r["resid"] == r["resid"]]
    ac = [r["actual"] for r in sub]
    if sl and rs:
        slope, resid = st.mean(sl), st.mean(rs)
        frac = abs(resid) / abs(st.mean(ac)) if st.mean(ac) else float("nan")
        print()
        print(f"  Regression on random sets: slope {slope:.3f}, "
              f"measured - predicted {resid:+.6f} ({frac:.0%} of the measured Delta)")
        if 0.9 <= slope <= 1.1 and frac < 0.1:
            print("  -> The objective is nearly additive.")
            print("     Combinations bring no gain; the top k single effects suffice.")
            print("     The premise of the algorithm paper does not hold.")
        elif slope < 0.9 or resid < 0:
            print("  -> The measured effect falls below the additive prediction. Effects are saturated,")
            print("     and the combined effect is smaller than additive (interference).")
            print("     There is room for search, but the claim is not 'discovery of synergy'")
            print("     but 'optimal allocation under saturation'.")
        else:
            print("  -> The measured effect exceeds the additive prediction. Non-additivity exists,")
            print("     and there is room for search to exploit.")
        print("\n  Note: in Geneformer's rank value encoding, inserting a token shifts")
        print("        the later ranks and truncates the tail. This analysis cannot separate")
        print("        this mechanical non-additivity from learned biological interactions.")

    _save()
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
