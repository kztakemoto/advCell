#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the main table and the values of figure_data.py from the result JSONs.

The strings printed by this script are meant to be pasted into the manuscript as they are.

Output:
  - LaTeX rows of the main table (6 rows)
  - ATTACK / GREEDY_POOL / MARGIN / SIG of figure_data.py
  - derived quantities quoted in the text (margin, fraction of cells where the
    attack wins, margin ratio against the random pool)

Tests are per-cell paired t-tests (two-sided, df = n-1), pooled across pathways.
Pathway-level sign tests are produced separately by pathway_level_signs.py.

Usage:
  python make_table3.py \
      --gf_attack  pw18_gf_k2/... pw18_gf_k4/... pw18_gf_k8/... \
      --gf_greedy  greedy_pool18_gf_k2/... greedy_pool18_gf_k4/... \
                   greedy_pool18_gf_k8/... \
      --sc_attack  pw18_scgpt_k{2,4,8}/search_comparison.json \
      --sc_greedy  greedy_pool18_scgpt_k{2,4,8}/search_comparison.json
"""
import argparse
import json
import math
import statistics as st

ATTACK = "gcg_faithful"
GRAD_POOL = "greedy_grad_pool"
RAND_POOL = "greedy_random_pool"


def t_sf(t, df):
    """Two-sided p-value from the t distribution, not a normal approximation."""
    x = df / (df + t * t)
    return float(betainc(df / 2.0, 0.5, x))


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-14:
            break
    r = front * (f - 1.0)
    return r if x < (a + 1) / (a + b + 2) else 1.0 - betainc(b, a, 1.0 - x)


def paired_t(diffs):
    n = len(diffs)
    m = st.mean(diffs)
    sd = st.stdev(diffs)
    t = m / (sd / math.sqrt(n))
    return m, t, t_sf(abs(t), n - 1), n


def paired(a_json, b_json, a_m, b_m):
    """List of differences paired by pathway and cell index."""
    ra, rb = a_json["results"], b_json["results"]
    out = []
    for pw in sorted(set(ra) & set(rb)):
        va, vb = ra[pw].get(a_m), rb[pw].get(b_m)
        if not isinstance(va, dict) or not isinstance(vb, dict):
            continue
        da, db = va.get("delta_per_cell"), vb.get("delta_per_cell")
        if not da or not db or len(da) != len(db):
            continue
        out += [x - y for x, y in zip(da, db)]
    return out


def mean_of(j, m):
    r = j["results"]
    v = [st.mean(r[pw][m]["delta_per_cell"]) for pw in r
         if isinstance(r[pw].get(m), dict) and r[pw][m].get("delta_per_cell")]
    return st.mean(v) if v else float("nan")


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"


def sci(p):
    e = math.floor(math.log10(p))
    return f"${p / 10 ** e:.1f}\\times10^{{{e}}}$"


def load(paths):
    out = {}
    for p in paths:
        j = json.load(open(p, encoding="utf-8"))
        out[int(j["config"]["k_adv"])] = j
    return out


def main():
    ap = argparse.ArgumentParser()
    for x in ("gf_attack", "gf_greedy", "sc_attack", "sc_greedy"):
        ap.add_argument("--" + x, nargs="+", default=[])
    args = ap.parse_args()

    models = []
    if args.gf_attack and args.gf_greedy:
        models.append(("Geneformer", load(args.gf_attack),
                       load(args.gf_greedy)))
    if args.sc_attack and args.sc_greedy:
        models.append(("scGPT", load(args.sc_attack), load(args.sc_greedy)))
    if not models:
        ap.error("specify both attack and greedy")

    rows, atk, grd, mar, sig, win, rmar = [], {}, {}, {}, {}, {}, {}
    for name, A, G in models:
        atk[name], grd[name], mar[name] = [], [], []
        sig[name], win[name], rmar[name] = [], [], []
        for k in sorted(A):
            if k not in G:
                continue
            a, g = A[k], G[k]
            ma, mg = mean_of(a, ATTACK), mean_of(g, GRAD_POOL)
            d = paired(a, g, ATTACK, GRAD_POOL)
            m, t, p, n = paired_t(d)
            frac = sum(1 for x in d if x > 0) / len(d)
            dr = paired(a, g, ATTACK, RAND_POOL)
            label = name if name == "scGPT" else name
            pad = "" if name == "Geneformer" else "     "
            rows.append(f"{label}{pad} & {k} & {ma:.6f} & {mg:.6f} & "
                        f"{ma / mg:.3f} & {sci(p)} \\\\")
            atk[name].append(round(ma, 6))
            grd[name].append(round(mg, 6))
            mar[name].append(round(m, 6))
            sig[name].append(stars(p))
            win[name].append(frac)
            # The manuscript's "1.22 to 2.17 times the pool's improvement" is
            # the ratio attack / random_pool, not margin / pool.
            rmar[name].append(ma / (ma - st.mean(dr)) if dr else float("nan"))
            print(f"  [{name} k={k}] n={n}  t={t:+.2f}  p={p:.3g}  "
                  f"attack wins {frac:.1%}", flush=True)

    print("\n" + "=" * 70)
    print("  Main table rows (paste as they are)")
    print("=" * 70)
    for r in rows:
        print(r)

    print("\n" + "=" * 70)
    print("  figure_data.py")
    print("=" * 70)
    for nm, d in (("ATTACK", atk), ("GREEDY_POOL", grd), ("MARGIN", mar)):
        pad = " " * (len(nm) + 4)
        body = (",\n" + pad).join(f'"{k}": {v}' for k, v in d.items())
        print(f"{nm} = {{{body}}}")
    body = ',\n       '.join(f'"{k}": {v}' for k, v in sig.items())
    print(f"SIG = {{{body}}}")

    print("\n" + "=" * 70)
    print("  Derived quantities quoted in the text")
    print("=" * 70)
    for nm in win:
        w = ", ".join(f"{x:.0%}" for x in win[nm])
        print(f"  {nm}: fraction of cells where the attack wins {w}")
    for nm in mar:
        m = " to ".join(f"{x:+.6f}" for x in mar[nm])
        print(f"  {nm}: margin against the gradient pool {m}")
    allr = [x for nm in rmar for x in rmar[nm] if x == x]
    if allr:
        print(f"  margin ratio against the random pool "
              f"{min(allr):.2f}--{max(allr):.2f}")


if __name__ == "__main__":
    main()
