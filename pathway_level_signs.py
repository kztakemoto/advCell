#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pathway-level sign test and a cell-by-cell check of the main table.

Uses only existing run outputs. No GPU or recomputation is needed. Run from the top of advCell/.

What it does:
  1. Recompute the 6 cells of the main table (2 models x k=2,4,8) from delta_per_cell
     and compare them with the manuscript values
  2. The per-cell tests are nested within pathways and are not independent replicates
     (pseudoreplication). Averaging per pathway and counting signs makes the effective
     sample size the number of pathways. The binomial-test p-value is also printed

Usage (adapt the directory names to your setup):
    python pathway_level_signs.py \
        --attack  pw18_gf_k2/search_comparison.json \
                  pw18_gf_k4/search_comparison.json \
                  pw18_gf_k8/search_comparison.json \
        --greedy  pw18_greedy_gf_k2/search_comparison.json \
                  pw18_greedy_gf_k4/search_comparison.json \
                  pw18_greedy_gf_k8/search_comparison.json \
        --label Geneformer

Files are matched through k_adv in the config, so the order need not be aligned.
"""
import argparse
import glob
import json
import math
import os
import statistics as st

ATTACK_METHOD = "gcg_faithful"
GREEDY_METHODS = ["greedy_grad_pool", "greedy_random_pool", "greedy"]


def binom_two_sided(k, n, p=0.5):
    """Two-sided p-value of the sign test."""
    if n == 0:
        return float("nan")

    def pmf(i):
        return math.comb(n, i) * p ** i * (1 - p) ** (n - i)

    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-15))


def load(paths):
    out = {}
    for pat in paths:
        for path in sorted(glob.glob(pat)):
            try:
                d = json.load(open(path, encoding="utf-8"))
            except Exception as e:                       # noqa: BLE001
                print(f"  [skip] {path}: {e}")
                continue
            k = d.get("config", {}).get("k_adv")
            if k is None:
                print(f"  [skip] {path}: no config.k_adv")
                continue
            out.setdefault(int(k), []).append((path, d))
    return out


def per_pathway_means(res, method):
    """pathway -> cell-mean delta of that pathway."""
    out = {}
    for pw, methods in res.items():
        v = methods.get(method)
        if not isinstance(v, dict):
            continue
        dpc = v.get("delta_per_cell")
        if dpc:
            out[pw] = st.mean(dpc)
        elif v.get("delta_mean") is not None:
            out[pw] = v["delta_mean"]
    return out


def mean_cost(res, method):
    """Mean cost of the method. Used to check that the budgets are matched."""
    cs = [m[method]["cost_mean"] for m in res.values()
          if isinstance(m.get(method), dict)
          and m[method].get("cost_mean") is not None]
    return st.mean(cs) if cs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", nargs="+", required=True)
    ap.add_argument("--greedy", nargs="*", default=[])
    ap.add_argument("--label", default="model")
    ap.add_argument("--greedy_method", default=None,
                    help="default: the first found in the order greedy_grad_pool -> greedy_random_pool "
                         "-> greedy")
    args = ap.parse_args()

    A = load(args.attack)
    G = load(args.greedy) if args.greedy else {}

    print("=" * 74)
    print(f"  {args.label}")
    print("=" * 74)

    # ---- 1. Recompute the main table ----
    print("\n[1] Recomputation from delta_per_cell (to check the main table)")
    print(f"  {'k':>2s} {'paths':>5s} {'cells':>5s} {'attack mean':>13s} "
          f"{'rounded':>10s}  file")
    print("  " + "-" * 70)
    for k in sorted(A):
        for path, d in A[k]:
            m = per_pathway_means(d["results"], ATTACK_METHOD)
            if not m:
                print(f"  {k:2d}  {ATTACK_METHOD} not found: {path}")
                continue
            ncell = len(next(iter(d["results"].values()))
                        [ATTACK_METHOD].get("delta_per_cell", []))
            mean = st.mean(m.values())
            print(f"  {k:2d} {len(m):5d} {ncell:5d} {mean:13.8f} "
                  f"{round(mean, 6):10.6f}  {os.path.dirname(path)}")
    print("\n  Compare with the main table of the manuscript cell by cell. If any cell is off,")
    print("  correct both figure_data.py and the main table to the measured values.")

    if not G:
        print("\n[2] No greedy files given, so the sign test is skipped.")
        print("    Pass pw18_greedy_*/search_comparison.json to --greedy.")
        return

    # ---- 2. Pathway-level sign test ----
    print("\n[2] Pathway-level sign test (guarding against pseudoreplication)")
    print("  The unit is the pathway, not the cell. The effective sample size is the number of pathways.")
    print("\n  Rows with different budgets are not matched-budget comparisons. greedy (all-TF greedy)")
    print("  runs at cost k*n, so it cannot be compared with the attack's 1290 / 2580.")
    print("  The matched-budget methods are greedy_grad_pool and greedy_random_pool.\n")
    print(f"  {'k':>2s} {'greedy method':22s} {'paths':>5s} {'attack wins':>11s} "
          f"{'p(sign)':>9s} {'mean diff':>12s} {'budget A/B':>17s} {'matched':>7s}")
    print("  " + "-" * 96)
    summary = {}
    for k in sorted(set(A) & set(G)):
        _, da = A[k][0]
        _, dg = G[k][0]
        ma = per_pathway_means(da["results"], ATTACK_METHOD)
        cands = ([args.greedy_method] if args.greedy_method
                 else GREEDY_METHODS)
        for gm in cands:
            mg = per_pathway_means(dg["results"], gm)
            common = sorted(set(ma) & set(mg))
            if not common:
                continue
            diffs = [ma[p] - mg[p] for p in common]
            wins = sum(1 for x in diffs if x > 0)
            p = binom_two_sided(max(wins, len(diffs) - wins), len(diffs))
            summary[f"k{k}_{gm}"] = {"n_pathways": len(common),
                                     "attack_wins": wins,
                                     "p_sign": p,
                                     "mean_diff": st.mean(diffs)}
            ca = mean_cost(da["results"], ATTACK_METHOD)
            cg = mean_cost(dg["results"], gm)
            matched = (abs(ca - cg) / max(ca, cg) < 0.05
                       if ca == ca and cg == cg else False)
            summary[f"k{k}_{gm}"]["cost_attack"] = ca
            summary[f"k{k}_{gm}"]["cost_greedy"] = cg
            summary[f"k{k}_{gm}"]["matched_budget"] = bool(matched)
            print(f"  {k:2d} {gm:22s} {len(common):5d} "
                  f"{f'{wins}/{len(common)}':>11s} {p:9.4g} "
                  f"{st.mean(diffs):+12.6f} "
                  f"{f'{ca:.0f}/{cg:.0f}':>17s} "
                  f"{('yes' if matched else 'no'):>7s}")
            if not matched:
                print("       ^ budgets differ, so this row is not a matched-budget comparison")
            losers = [p_ for p_, x in zip(common, diffs) if x <= 0]
            if losers and matched:
                print(f"     pathways lost: {', '.join(losers)}")

    print("\n  One sentence for the main text or Limitations:")
    print("    Cells are nested within pathways, so the paired tests over")
    print("    cells are not independent replicates. Aggregating to the")
    print("    pathway level, the attack exceeds greedy in N of 18 pathways")
    print("    at k = ... (sign test p = ...).")

    json.dump(summary, open("pathway_level_signs.json", "w"),
              ensure_ascii=False, indent=1)
    print("\n  [saved] pathway_level_signs.json")


if __name__ == "__main__":
    main()
