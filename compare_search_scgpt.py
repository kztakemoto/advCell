#!/usr/bin/env python3
"""
Matched-budget comparison of search algorithms -- scGPT version
=========================================
Uses the implementation in search_algorithms.py as it is. Only two callbacks are model-specific:
"evaluate a set" and "return the top gradient candidates", so the algorithms are
identical to the Geneformer version down to the last line.

Structural differences between scGPT and Geneformer:
  Geneformer  rank value encoding. Overexpression inserts the token at the front,
              shifting the later ranks and truncating the tail. Depends on the insertion order.
  scGPT       gene token + expression value. Overexpression only rewrites the expression value
              at the existing position to high_value. Positions do not shift and nothing is truncated.
              It should not depend on the insertion order, which order_sensitivity can confirm.

Therefore, if sub-additivity is also observed in scGPT, it is not an
artifact of Geneformer's rank encoding.

Usage:
  python compare_search_scgpt.py \\
      --model_dir ./scGPT/scGPT_human \\
      --pathway_config ./data/pathway_config_geneformer.json \\
      --tf_list ./tf_list_human.txt \\
      --pathways P01 P03 P07 P12 P14 \\
      --n_cells 20 --k_adv 4 --budget 2580 --n_restarts 3 \\
      --greedy --order_sets 20 \\
      --output_dir ./search_scgpt_k4
"""

import argparse
import json
import os
import statistics as st
import sys

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advcell_pathway_loader import load_pathway_config, get_pathway_ids
from gpr_scgpt import (
    load_scgpt_model, prepare_cell_data, compute_target_centroid,
)
import search_algorithms as SA


# ------------------------------------------------------------
class ScgptEvaluator:
    """For one cell, provide TF set -> objective value / top gradient candidates."""

    def __init__(self, model, vocab, cell, tf_gene_ids, target_centroid,
                 device, max_seq_len, high_value, batch_size):
        self.model = model
        self.device = device
        self.max_seq_len = max_seq_len
        self.high_value = high_value
        self.bs = batch_size
        self.centroid = target_centroid
        self.pad_id = vocab["<pad>"]

        # the key of the return value of prepare_cell_data is "gene_ids"
        gene_ids = list(cell["gene_ids"])[:max_seq_len]
        values = list(cell["values"])[:max_seq_len]
        self.n_orig = len(gene_ids)
        self.tf_gene_ids = list(tf_gene_ids)
        self.n_tf = len(self.tf_gene_ids)

        # Resolve in advance whether each TF is already present.
        # present -> rewrite the value at that position
        # absent  -> append at the end (scGPT does not shift positions)
        pos_of = {g: i for i, g in enumerate(gene_ids)}
        self.tf_pos = []          # position of a present TF; None if absent
        for g in self.tf_gene_ids:
            self.tf_pos.append(pos_of.get(g))

        self.base_genes = gene_ids
        self.base_values = values

        with torch.no_grad():
            self.tf_embs = model.encoder(
                torch.tensor(self.tf_gene_ids, dtype=torch.long, device=device)
            )                                   # (n_tf, d)

    # -- internal: build (src, val, mask) from a set --
    def _build(self, sets):
        """sets: (N, k) TF indices -> padded tensors

        TFs absent from the cell are appended at the end. If the sequence has reached max_seq_len,
        writing all of them to the single last position makes the k of them collide and only the
        last one of the permutation remains (the search degenerates to effectively k=1). To avoid this,
        the base sequence is truncated by k slots in advance to secure the seats.
        """
        N, k = sets.shape
        # always secure seats for k
        keep = max(0, min(self.n_orig, self.max_seq_len - k))
        base_g = torch.tensor(self.base_genes[:keep], dtype=torch.long)
        base_v = torch.tensor(self.base_values[:keep], dtype=torch.float32)
        T = keep + k

        src = torch.full((N, T), self.pad_id, dtype=torch.long)
        val = torch.zeros((N, T), dtype=torch.float32)
        src[:, :keep] = base_g
        val[:, :keep] = base_v

        lengths = torch.full((N,), keep, dtype=torch.long)
        for r in range(N):
            tail = int(lengths[r].item())
            for idx in sets[r].tolist():
                p = self.tf_pos[idx]
                if p is not None and p < keep:
                    # present gene -> rewrite the expression value at that position
                    val[r, p] = self.high_value
                else:
                    # absent (or dropped by the truncation) -> add to a dedicated slot
                    src[r, tail] = self.tf_gene_ids[idx]
                    val[r, tail] = self.high_value
                    tail += 1
            lengths[r] = tail
        ar = torch.arange(T).view(1, T)
        pad_mask = ar >= lengths.view(N, 1)
        src[pad_mask] = self.pad_id
        val[pad_mask] = 0.0
        return (src.to(self.device), val.to(self.device),
                pad_mask.to(self.device))

    def _forward(self, src, val, pad_mask, adv_emb=None, adv_pos=None):
        m = self.model
        if adv_emb is None:
            with torch.no_grad():
                gene_emb = m.encoder(src)
        else:
            with torch.no_grad():
                gene_emb = m.encoder(src)
            gene_emb = gene_emb.detach().clone()
            for i, p in enumerate(adv_pos):
                gene_emb[0, p, :] = adv_emb[i]
        total = gene_emb + m.value_encoder(val.float())
        if hasattr(m, "bn") and m.bn is not None:
            total = m.bn(total.permute(0, 2, 1)).permute(0, 2, 1)
        out = m.transformer_encoder(total, src_key_padding_mask=pad_mask)
        w = (~pad_mask).unsqueeze(-1).float()
        return (out * w).sum(1) / w.sum(1)       # (N, d)

    @torch.no_grad()
    def evaluate(self, sets):
        """Callback for SA: (N,k) -> (N,) cos_sim"""
        outs = []
        for i in range(0, sets.shape[0], self.bs):
            chunk = sets[i:i + self.bs]
            src, val, pm = self._build(chunk)
            emb = self._forward(src, val, pm)
            outs.append(F.cosine_similarity(
                emb, self.centroid.unsqueeze(0), dim=1).cpu())
        return torch.cat(outs)

    def base_value(self):
        src = torch.tensor([self.base_genes[:self.max_seq_len]],
                           dtype=torch.long, device=self.device)
        val = torch.tensor([self.base_values[:self.max_seq_len]],
                           dtype=torch.float32, device=self.device)
        pm = src.eq(self.pad_id)
        with torch.no_grad():
            emb = self._forward(src, val, pm)
        return float(F.cosine_similarity(
            emb, self.centroid.unsqueeze(0), dim=1).item())

    def grad_topk(self, current, top_k):
        """Callback for SA: (k,) -> (k, top_k) TF indices"""
        src, val, pm = self._build(current.view(1, -1))
        # locate the adversarial positions
        adv_pos = []
        for idx in current.tolist():
            g = self.tf_gene_ids[idx]
            hit = (src[0] == g).nonzero(as_tuple=True)[0]
            adv_pos.append(int(hit[0].item()) if len(hit) else 0)

        with torch.no_grad():
            ge = self.model.encoder(src)
        adv_emb = ge[0, adv_pos, :].detach().clone().requires_grad_(True)
        emb = self._forward(src, val, pm, adv_emb=adv_emb, adv_pos=adv_pos)
        loss = -F.cosine_similarity(
            emb, self.centroid.unsqueeze(0), dim=1).mean()
        loss.backward()
        if adv_emb.grad is None:
            return None
        scores = (-adv_emb.grad) @ self.tf_embs.T      # (k, n_tf)
        kk = min(top_k, scores.shape[1])
        return scores.topk(kk, dim=1).indices.cpu()


# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--pathway_config", required=True)
    ap.add_argument("--tf_list", required=True)
    ap.add_argument("--pathways", nargs="+", default=["all"])
    ap.add_argument("--n_cells", type=int, default=20)
    ap.add_argument("--n_target_cells", type=int, default=200)
    ap.add_argument("--max_seq_len", type=int, default=1200)
    ap.add_argument("--binning_mode", default="per_cell",
                    choices=["global", "per_cell", "none"])
    ap.add_argument("--high_value", type=float, default=50.0)
    ap.add_argument("--k_adv", type=int, default=4)
    ap.add_argument("--budget", type=int, default=2580)
    ap.add_argument("--n_cand", type=int, default=128)
    ap.add_argument("--top_k_grad", type=int, default=256)
    ap.add_argument("--n_restarts", type=int, default=3)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--methods", nargs="+", default=None,
                    choices=["random_best", "hill_random", "hill_gradient",
                             "hill_random_faithful", "gcg_faithful"],
                    help="Restrict the stochastic methods to run (default: all)")
    ap.add_argument("--only_greedy", action="store_true",
                    help="Skip the stochastic methods and run only the greedy ones")
    ap.add_argument("--greedy_pool", type=int, default=0,
                    help=">0 also runs greedy methods with the candidate pool narrowed to this number. "
                         "Two kinds: random thinning and gradient top.")
    ap.add_argument("--order_sets", type=int, default=20)
    ap.add_argument("--max_perms", type=int, default=64)
    ap.add_argument("--order_cells", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default="./search_scgpt")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, vocab, _ = load_scgpt_model(args.model_dir, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with open(args.tf_list) as f:
        all_tfs = [ln.strip() for ln in f if ln.strip()]
    tf_names, tf_gene_ids = [], []
    for t in all_tfs:
        if t in vocab:
            tf_names.append(t)
            tf_gene_ids.append(vocab[t])
    n_tf = len(tf_names)
    print(f"TFs in vocab: {n_tf}/{len(all_tfs)}  budget {args.budget}  k={args.k_adv}")

    pw_config, h5ad_paths = load_pathway_config(args.pathway_config,
                                                model="scgpt")
    pw_ids = get_pathway_ids(
        pw_config, None if args.pathways == ["all"] else args.pathways)

    outfile = os.path.join(args.output_dir, "search_comparison.json")
    all_res = {}

    def save():
        tmp = outfile + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"config": vars(args), "model": "scGPT",
                       "results": all_res}, f, indent=2, ensure_ascii=False)
        os.replace(tmp, outfile)

    for pw_name in pw_ids:
        pw = pw_config[pw_name]
        print(f"\n{'='*72}\nPathway: {pw_name}\n{'='*72}", flush=True)
        adata = sc.read_h5ad(h5ad_paths[pw_name])
        src_cells = prepare_cell_data(
            adata, pw["source_type"], vocab, n_cells=args.n_cells,
            seed=args.seed, binning_mode=args.binning_mode)
        tgt_cells = prepare_cell_data(
            adata, pw["target_type"], vocab, n_cells=args.n_target_cells,
            seed=args.seed, binning_mode=args.binning_mode)
        centroid = compute_target_centroid(
            model, tgt_cells, vocab, device, max_seq_len=args.max_seq_len)

        rec = {}
        for ci, cell in enumerate(src_cells):
            ev = ScgptEvaluator(model, vocab, cell, tf_gene_ids, centroid,
                                device, args.max_seq_len, args.high_value,
                                args.batch_size)
            base = ev.base_value()
            eval_fn = ev.evaluate
            grad_fn = lambda cur: ev.grad_topk(cur, args.top_k_grad)

            res = ({} if args.only_greedy else SA.run_all_methods(
                eval_fn, grad_fn, n_tf, args.k_adv, args.budget,
                args.n_cand, base, args.seed, ci, args.n_restarts,
                methods=(args.methods or SA.METHODS)))
            for m, v in res.items():
                d = rec.setdefault(m, {"delta": [], "cost": [],
                                       "delta_restarts": [], "init": [],
                                       "curve": [], "sets_restarts": []})
                d["delta"].append(v["delta"]); d["cost"].append(v["cost"])
                d["delta_restarts"].append(v["delta_restarts"])
                d["init"].append(v["init"]); d["curve"].append(v["curve"])
                # keep the gene symbols in the same nesting as delta_restarts
                d.setdefault("sets_restarts", []).append(
                    [[tf_names[i] for i in (st_ or [])]
                     for st_ in v.get("set_restarts", [])])

            if args.greedy:
                dg, cg, _ = SA.run_greedy(eval_fn, n_tf, args.k_adv, base)
                d = rec.setdefault("greedy", {"delta": [], "cost": []})
                d["delta"].append(dg); d["cost"].append(cg)

            # greedy with a narrowed pool (a control for "why not just use greedy at the same budget")
            if args.greedy_pool > 0:
                rp = np.random.default_rng(args.seed + ci).choice(
                    n_tf, size=min(args.greedy_pool, n_tf), replace=False)
                dg, cg, _ = SA.run_greedy_pooled(eval_fn, rp.tolist(),
                                                 args.k_adv, base)
                d = rec.setdefault("greedy_random_pool",
                                   {"delta": [], "cost": []})
                d["delta"].append(dg); d["cost"].append(cg)

                tk = ev.grad_topk(
                    torch.tensor(rp[:args.k_adv].tolist(), dtype=torch.long),
                    args.greedy_pool)
                if tk is not None:
                    gp = sorted({int(x) for x in tk[0].tolist()})
                    dg, cg, _ = SA.run_greedy_pooled(eval_fn, gp,
                                                     args.k_adv, base)
                    d = rec.setdefault("greedy_grad_pool",
                                       {"delta": [], "cost": []})
                    d["delta"].append(dg); d["cost"].append(cg + 2)

            if (args.order_sets > 0 and ci < args.order_cells
                    and not args.only_greedy):
                sp = SA.order_sensitivity(
                    eval_fn, n_tf, args.k_adv, args.order_sets,
                    np.random.default_rng(args.seed + ci),
                    max_perms=args.max_perms)
                d = rec.setdefault("order_spread", {"delta": [], "cost": []})
                d["delta"].append(st.mean(sp)); d["cost"].append(0)

            if (ci + 1) % 5 == 0:
                print(f"  cell {ci+1}/{len(src_cells)}", flush=True)

        all_res[pw_name] = {
            m: {"delta_mean": st.mean(v["delta"]),
                "delta_sd": st.pstdev(v["delta"]),
                "cost_mean": st.mean(v["cost"]),
                "n_cells": len(v["delta"]),
                "delta_per_cell": [float(x) for x in v["delta"]],
                "delta_restarts": v.get("delta_restarts"),
                "init_mean": st.mean(v["init"]) if v.get("init") else None,
                "curve": v.get("curve"),
                # gene symbols of [cell][restart][k]
                "sets_restarts": v.get("sets_restarts")}
            for m, v in rec.items() if v["delta"]}

        print(f"\n  {'method':16s} {'Δ':>11s} {'cost':>10s}")
        for m, v in all_res[pw_name].items():
            print(f"  {m:16s} {v['delta_mean']:+11.6f} {v['cost_mean']:10.0f}")
        save()
        print(f"  [saved] {outfile}")
        del adata

    print(f"\n{'='*72}\n  Summary over all pathways (scGPT)\n{'='*72}")
    if not args.only_greedy:
        SA.report_comparison(all_res)

    osp = [r["order_spread"]["delta_mean"] for r in all_res.values()
           if "order_spread" in r and "hill_gradient" in r]
    if osp:
        hg = st.mean([r["hill_gradient"]["delta_mean"]
                      for r in all_res.values() if "hill_gradient" in r])
        print(f"\n  Sensitivity to insertion order: {st.mean(osp):+.8f} "
              f"(of the search improvement {hg:+.6f}, "
              f"{st.mean(osp)/hg*100 if hg else float('nan'):.1f}%)")
        print("  Note: scGPT does not shift positions, so this should be close to 0. "
              "If it is not, suspect the implementation.")
    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
