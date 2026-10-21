#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interaction analysis of the 2x2 design (candidate source x acceptance rule).

The 4 conditions are split over separate search_comparison.json directories,
so they are joined by key (pathway, cell) and a cell-level paired test is performed.

    a = uniform  x improve-only   (hill_random)
    b = gradient x improve-only   (hill_gradient)
    c = uniform  x unconditional  (hill_random_faithful)
    d = gradient x unconditional  (gcg_faithful)

Contrasts printed:
    simple effect of gradient       : (b-a), (d-c)
    simple effect of acceptance rule: (c-a), (d-b)
    main effects                    : ((b-a)+(d-c))/2, ((c-a)+(d-b))/2
    interaction                     : (d-b) - (c-a) = a - b - c + d

Example:
    python analyze_2x2_interaction.py \
        --json ./budget_match_gf_k8_hill_random/search_comparison.json \
        --json ./budget_match_gf_k8_hill_gradient/search_comparison.json \
        --json ./budget_match_gf_k8_hill_random_faithful/search_comparison.json \
        --json ./budget_match_gf_k8_gcg_faithful/search_comparison.json \
        --out ./interaction_2x2_gf_k8.json

With --agg best, the same analysis is run on the best-of-R of delta_restarts.
"""

import argparse
import json
import os
import sys
from collections import OrderedDict

import numpy as np

try:
    from scipy import stats as sstats
except ImportError:  # without scipy, report up to the t value
    sstats = None
    sys.stderr.write("[warn] scipy not available, so p-values are not computed\n")


# ---------------------------------------------------------------- loading

def load_runs(paths):
    """Read several search_comparison.json files and flatten them to method -> pathway -> record."""
    methods = OrderedDict()
    configs = {}
    for p in paths:
        if not os.path.exists(p):
            sys.exit("[error] file not found: %s" % p)
        with open(p) as f:
            d = json.load(f)
        cfg = d.get("config", {})
        configs[p] = cfg
        for pw, per_method in d.get("results", {}).items():
            for m, rec in per_method.items():
                methods.setdefault(m, {})
                if pw in methods[m]:
                    prev = methods[m][pw]["delta_mean"]
                    cur = rec["delta_mean"]
                    if abs(prev - cur) > 1e-12:
                        sys.stderr.write(
                            "[warn] %s / %s is in several JSONs with different values "
                            "(%.6f vs %.6f). Using the one read first\n" % (m, pw, prev, cur))
                    continue
                methods[m][pw] = rec
    return methods, configs


def check_config_consistency(configs):
    """Check that k_adv / budget / n_cells / pathways agree across the 4 runs."""
    keys = ["k_adv", "budget", "n_cells", "n_target_cells", "n_restarts",
            "n_cand", "top_k_grad", "seed", "pathways"]
    ref_path, ref_cfg = next(iter(configs.items()))
    ok = True
    for p, cfg in configs.items():
        for k in keys:
            if k not in ref_cfg or k not in cfg:
                continue
            if cfg[k] != ref_cfg[k]:
                ok = False
                sys.stderr.write("[warn] config mismatch %s: %s=%r vs %s=%r\n"
                                 % (k, os.path.basename(os.path.dirname(p)), cfg[k],
                                    os.path.basename(os.path.dirname(ref_path)), ref_cfg[k]))
    if ok:
        print("  config agrees across the 4 runs (k_adv=%s, budget=%s, n_cells=%s, seed=%s)"
              % (ref_cfg.get("k_adv"), ref_cfg.get("budget"),
                 ref_cfg.get("n_cells"), ref_cfg.get("seed")))
    return ref_cfg


def cell_vector(rec, agg):
    """Return the cell-level values of one pathway. agg='mean' is the mean, 'best' is best-of-R."""
    if agg == "mean":
        return np.asarray(rec["delta_per_cell"], dtype=float)
    rest = rec.get("delta_restarts")
    if not rest:
        return np.asarray(rec["delta_per_cell"], dtype=float)
    return np.asarray([max(r) for r in rest], dtype=float)


def build_matrix(methods, names, pathways, agg):
    """Build a vector of (n_cells_total,) in the order of names. Also return the pathway boundaries."""
    vecs, bounds, used_pw = [], [], []
    for m in names:
        if m not in methods:
            sys.exit("[error] method %s is not in any JSON. "
                     "Methods found: %s" % (m, ", ".join(methods.keys())))
    for pw in pathways:
        lens = []
        for m in names:
            if pw not in methods[m]:
                lens = []
                break
            lens.append(len(cell_vector(methods[m][pw], agg)))
        if not lens:
            sys.stderr.write("[warn] pathway %s excluded because it is not present in all conditions\n" % pw)
            continue
        if len(set(lens)) != 1:
            sys.stderr.write("[warn] pathway %s excluded because its cell count differs across conditions %r\n" % (pw, lens))
            continue
        used_pw.append(pw)
        bounds.append(lens[0])
    if not used_pw:
        sys.exit("[error] no pathway is common to all conditions")
    mat = np.vstack([
        np.concatenate([cell_vector(methods[m][pw], agg) for pw in used_pw])
        for m in names
    ])
    return mat, used_pw, bounds


# ---------------------------------------------------------------- tests

def paired(x, label):
    """Paired t-test against 0 (x is the vector of differences)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    se = sd / np.sqrt(n) if n > 1 else float("nan")
    t = mean / se if se and se > 0 else float("nan")
    if sstats is not None and n > 1:
        p = float(sstats.ttest_rel(x, np.zeros_like(x)).pvalue)
        try:
            pw = float(sstats.wilcoxon(x).pvalue)
        except Exception:
            pw = float("nan")
    else:
        p, pw = float("nan"), float("nan")
    wins = int((x > 0).sum())
    return dict(label=label, n=n, mean=mean, sd=sd, t=float(t),
                p=p, p_wilcoxon=pw, wins=wins,
                ci90=[mean - 1.645 * se, mean + 1.645 * se] if se == se else None)


def fmt(r):
    return ("  %-34s %+.6f  t=%+7.2f  p=%-10.3g wins %3d/%-3d"
            % (r["label"], r["mean"], r["t"], r["p"], r["wins"], r["n"]))


# ---------------------------------------------------------------- auxiliary output

def pairing_sanity(methods, names, pathways):
    print("\n=== Sanity check of the pairing ===")
    print("  If init_mean agrees across conditions, the seed of the initial sets is shared")
    print("  %-8s %s" % ("pathway", "  ".join("%-14s" % m[:14] for m in names)))
    for pw in pathways:
        row = []
        for m in names:
            rec = methods[m].get(pw)
            iv = rec.get("init_mean") if rec else None
            row.append("%+.6f" % iv if isinstance(iv, (int, float)) else "   n/a    ")
        vals = [methods[m][pw]["init_mean"] for m in names
                if pw in methods[m]
                and isinstance(methods[m][pw].get("init_mean"), (int, float))]
        flag = "" if (len(vals) < 2 or max(vals) - min(vals) < 1e-9) else "  <-- mismatch"
        print("  %-8s %s%s" % (pw, "  ".join("%-14s" % v for v in row), flag))

    print("\n=== Agreement of cost ===")
    print("  %-8s %s" % ("pathway", "  ".join("%-14s" % m[:14] for m in names)))
    for pw in pathways:
        row = []
        for m in names:
            rec = methods[m].get(pw)
            cv = rec.get("cost_mean") if rec else None
            row.append("%14.1f" % cv if isinstance(cv, (int, float)) else "     n/a      ")
        print("  %-8s %s" % (pw, "  ".join(row)))


def anytime_table(methods, names, pathways):
    print("\n=== Anytime (mean over all pathways and cells) ===")
    grids = None
    curves = {}
    for m in names:
        acc = {}
        for pw in pathways:
            rec = methods[m].get(pw)
            if not rec or "curve" not in rec:
                continue
            for cell_curve in rec["curve"]:
                for b, v in cell_curve:
                    acc.setdefault(int(b), []).append(float(v))
        if not acc:
            continue
        curves[m] = {b: float(np.mean(v)) for b, v in acc.items()}
        g = sorted(acc.keys())
        grids = g if grids is None else [b for b in grids if b in acc]
    if not grids:
        print("  skipped because there is no curve")
        return curves
    print("  %-22s %s" % ("method", " ".join("%9d" % b for b in grids)))
    for m in names:
        if m not in curves:
            continue
        print("  %-22s %s" % (m[:22], " ".join("%9.5f" % curves[m][b] for b in grids)))
    return curves


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", required=True,
                    help="path of search_comparison.json (give it 4 times, once per condition)")
    ap.add_argument("--uniform-improve", default="hill_random")
    ap.add_argument("--gradient-improve", default="hill_gradient")
    ap.add_argument("--uniform-uncond", default="hill_random_faithful")
    ap.add_argument("--gradient-uncond", default="gcg_faithful")
    ap.add_argument("--baseline", default="greedy",
                    help="baseline for the ratio (skipped automatically if absent)")
    ap.add_argument("--agg", choices=["mean", "best", "both"], default="both",
                    help="aggregation of cell values. mean = mean over restarts, best = best-of-R")
    ap.add_argument("--out", default=None, help="where to save the result JSON")
    args = ap.parse_args()

    methods, configs = load_runs(args.json)
    print("=== Loading ===")
    for p in args.json:
        print("  %s" % p)
    print("  Methods found: %s" % ", ".join(methods.keys()))
    ref_cfg = check_config_consistency(configs)

    names = [args.uniform_improve, args.gradient_improve,
             args.uniform_uncond, args.gradient_uncond]
    labels = ["a uniform x improving", "b gradient x improving", "c uniform x unconditional", "d gradient x unconditional"]

    pathways = ref_cfg.get("pathways")
    if not pathways:
        pathways = sorted(set().union(*[set(v.keys()) for v in methods.values()]))

    pairing_sanity(methods, names + ([args.baseline] if args.baseline in methods else []),
                   pathways)

    out = {"config": {"k_adv": ref_cfg.get("k_adv"), "budget": ref_cfg.get("budget"),
                      "methods": dict(zip(["a", "b", "c", "d"], names))},
           "agg": {}}

    aggs = ["mean", "best"] if args.agg == "both" else [args.agg]
    for agg in aggs:
        mat, used_pw, bounds = build_matrix(methods, names, pathways, agg)
        a, b, c, d = mat
        n = mat.shape[1]

        print("\n" + "=" * 72)
        print("  2x2 interaction analysis   agg=%s   n=%d cells (%d pathways)"
              % (agg, n, len(used_pw)))
        print("=" * 72)

        print("\n=== Cell means ===")
        print("  %-16s %-14s %-14s" % ("", "improving only", "unconditional"))
        print("  %-16s %+.6f      %+.6f" % ("uniform cand.", a.mean(), c.mean()))
        print("  %-16s %+.6f      %+.6f" % ("gradient cand.", b.mean(), d.mean()))
        if args.baseline in methods:
            bm, _, _ = build_matrix(methods, [args.baseline], used_pw, agg)
            base = bm[0]
            print("\n  Baseline %s: %+.6f" % (args.baseline, base.mean()))
            for nm, v in zip(labels, [a, b, c, d]):
                r = paired(v - base, "%s vs %s" % (nm, args.baseline))
                print(fmt(r) + "   ratio %.3f" % (v.mean() / base.mean()))

        res = {}
        print("\n=== Simple main effects ===")
        for key, vec, lab in [
                ("grad_under_improve", b - a, "effect of gradient (improving acceptance) b-a"),
                ("grad_under_uncond", d - c, "effect of gradient (unconditional acceptance) d-c"),
                ("uncond_under_uniform", c - a, "effect of unconditional acceptance (uniform) c-a"),
                ("uncond_under_gradient", d - b, "effect of unconditional acceptance (gradient) d-b")]:
            res[key] = paired(vec, lab)
            print(fmt(res[key]))

        print("\n=== Main effects ===")
        for key, vec, lab in [
                ("main_gradient", ((b - a) + (d - c)) / 2.0, "main effect of gradient"),
                ("main_uncond", ((c - a) + (d - b)) / 2.0, "main effect of acceptance rule")]:
            res[key] = paired(vec, lab)
            print(fmt(res[key]))

        print("\n=== Interaction ===")
        inter = a - b - c + d
        res["interaction"] = paired(inter, "interaction (d-b)-(c-a)")
        print(fmt(res["interaction"]))
        ci = res["interaction"]["ci90"]
        if ci:
            print("  90%%CI [%+.6f, %+.6f]" % (ci[0], ci[1]))

        print("\n=== Interaction per pathway ===")
        off = 0
        res["interaction_per_pathway"] = {}
        for pw, ln in zip(used_pw, bounds):
            r = paired(inter[off:off + ln], pw)
            res["interaction_per_pathway"][pw] = r
            print(fmt(r))
            off += ln

        out["agg"][agg] = {
            "n_cells": n, "pathways": used_pw,
            "cell_means": dict(zip(["a", "b", "c", "d"],
                                   [float(x.mean()) for x in (a, b, c, d)])),
            "tests": res,
        }

    curves = anytime_table(methods, names, pathways)
    out["anytime"] = {m: {str(k): v for k, v in cv.items()} for m, cv in curves.items()}

    if args.out:
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        os.replace(tmp, args.out)
        print("\n[saved] %s" % args.out)


if __name__ == "__main__":
    main()
