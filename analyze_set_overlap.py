#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Match the gene sets returned by the search against the published cocktails (protocols) of each pathway.

The ground truth is per protocol, not the union. The union is a patchwork nobody performed, and it
cannot distinguish a set picking one factor each from mutually incompatible cocktails
from a set that hits one cocktail. A combinatorial-search paper has to make that distinction.

For each set S take h = max_j |S ∩ C_j|, where C_j are the protocols of that pathway.

The main metric is the hit rate (fraction of sets with h >= 1). The expected number of hits of a random
k-tuple is only 0.03 averaged over pathways even at k=8, so the distribution of mean hit counts sticks to zero
and does not move. Fractions can be read. The secondary metric, the h >= 2 rate, means "two or more factors from the same cocktail",
which excludes picking one each from different cocktails.

The random baseline is computed by the same max-over-protocols procedure, so the multiplicity
that pathways with more protocols are easier to hit also enters the denominator and cancels.

Usage:
    python analyze_set_overlap.py \\
        --results pw18set_gf_k8/search_comparison.json \\
        --greedy  greedy_pool18_gf_k8/search_comparison.json \\
        --protocols protocols_by_pathway.json \\
        --scores "Top-k single-gene" results_kadv/k4/multi_pathway_geneformer_k4_n50.json \\
        --scores "DEG(wilcoxon) top-k" deg_baseline_results/deg_baseline_wilcoxon.json \\
        --n_vocab 2632 --out set_overlap_k8
"""
import argparse
import json
import math
import os
import random
import statistics as st
from collections import defaultdict

N_DRAWS = 20000
SEED = 0
SEARCH_METHODS = {"gcg_faithful", "hill_random", "hill_gradient",
                  "hill_random_faithful", "random_best", "greedy_grad_pool",
                  "greedy_random_pool", "greedy"}


def binom_two_sided(k, n, p=0.5):
    if n == 0:
        return float("nan")

    def pmf(i):
        return math.comb(n, i) * p ** i * (1 - p) ** (n - i)

    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-15))


def spearman(x, y):
    if len(x) < 3:
        return float("nan")

    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            a = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                r[o[t]] = a
            i = j + 1
        return r

    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def best_hit(S, protocols):
    """h = max_j |S ∩ C_j|, the best protocol, and whether it was reproduced completely.

    Complete reproduction = contains all factors of some cocktail C_j (|S ∩ C_j| == |C_j|).
    Single-factor cocktails are excluded, because drawing that one gene would count as
    "reproducing the whole cocktail" and the metric would lose its meaning. Two of the 18 pathways
    (ETV2 for fibroblast -> endothelial, MYOCD for fibroblast -> smooth muscle) are such cases.
    They contribute to h >= 1 but not to complete reproduction.
    """
    best, which, full = 0, None, None
    for pid, C in protocols:
        n = len(S & C)
        if n > best:
            best, which = n, pid
        if len(C) >= 2 and n == len(C) and (full is None or len(C) > full[1]):
            full = (pid, len(C))
    return best, which, full


def random_baseline(protocols, k, n_vocab, rng):
    """Score random k-tuples by the same procedure.

    Give the curated genes consecutive numbers 0..m-1, draw k from the vocabulary, and count
    those below m as hits. Which gene is at which position is exchangeable, so this is an exact null distribution.
    """
    genes = sorted(set().union(*[C for _, C in protocols]))
    idx = {g: i for i, g in enumerate(genes)}
    m = len(genes)
    prot_idx = [(pid, {idx[g] for g in C}) for pid, C in protocols]
    h1 = h2 = hf = 0
    tot = 0
    for _ in range(N_DRAWS):
        drawn = {i for i in rng.sample(range(n_vocab), k) if i < m}
        h = max((len(drawn & c) for _, c in prot_idx), default=0)
        h1 += h >= 1
        h2 += h >= 2
        hf += any(len(c) >= 2 and len(drawn & c) == len(c)
                  for _, c in prot_idx)
        tot += h
    return h1 / N_DRAWS, h2 / N_DRAWS, hf / N_DRAWS, tot / N_DRAWS


def iter_sets(j):
    k = j["config"]["k_adv"]
    for pw, methods in j["results"].items():
        if not isinstance(methods, dict):
            continue
        for m, v in methods.items():
            if not isinstance(v, dict):
                continue
            sr, dr = v.get("sets_restarts"), v.get("delta_restarts")
            if sr and dr:
                for cs, cd in zip(sr, dr):
                    for s, d in zip(cs, cd):
                        if s:
                            yield pw, m, k, float(d), set(s)
                continue
            sp, dp = v.get("sets_per_cell"), v.get("delta_per_cell")
            if sp and dp:
                for s, d in zip(sp, dp):
                    if s:
                        yield pw, m, k, float(d), set(s)


def topk_sets(path, label, k):
    """Top k per pathway from a score JSON (one set per pathway)."""
    j = json.load(open(path, encoding="utf-8"))
    res = j.get("results", j)
    out = []
    for pw, node in res.items():
        if not isinstance(node, dict):
            continue
        sc = node.get("scores") if isinstance(node.get("scores"), dict) else None
        if sc is None:
            for key in sorted(node, reverse=True):
                v = node.get(key)
                if isinstance(v, dict) and isinstance(v.get("mean_scores"),
                                                      dict):
                    sc = v["mean_scores"]
                    break
        if not sc:
            continue
        out.append((pw, label, k, float("nan"),
                    set(sorted(sc, key=lambda g: -sc[g])[:k])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--greedy", nargs="*", default=[])
    ap.add_argument("--protocols", required=True)
    ap.add_argument("--scores", nargs=2, action="append", default=[],
                    metavar=("LABEL", "PATH"))
    ap.add_argument("--n_vocab", type=int, default=2632)
    ap.add_argument("--rho_null", type=float, default=0.10,
                    help="Decision criterion. Fix it before looking at the results. "
                         "To say \"related\", both the absolute value of the median of the within-pathway rho "
                         "must exceed this value and the sign test must give p < 0.05. "
                         "Default 0.10")
    ap.add_argument("--out", default="set_overlap")
    args = ap.parse_args()

    P = json.load(open(args.protocols, encoding="utf-8"))
    protos = {}
    for pid, v in P.items():
        U = set(v["tf_union"])
        ps = [(p["id"], set(p["factors"]) & U) for p in v["protocols"]]
        ps = [(i, c) for i, c in ps if c]
        if ps:
            protos[pid] = ps

    rows = []
    for path in list(args.results) + list(args.greedy):
        if not os.path.exists(path):
            print(f"[skip] {path}")
            continue
        j = json.load(open(path, encoding="utf-8"))
        got = 0
        for pw, m, k, d, S in iter_sets(j):
            if pw not in protos:
                continue
            h, w, full = best_hit(S, protos[pw])
            rows.append({"pathway": pw, "method": m, "k": k, "delta": d,
                         "set": sorted(S), "h": h, "best_protocol": w,
                         "full": full[0] if full else None,
                         "full_size": full[1] if full else 0})
            got += 1
        if got == 0:
            print(f"[warn] {path} contains no sets")

    if not rows:
        print("No sets could be read.")
        return
    ks = sorted({r["k"] for r in rows})

    for label, path in args.scores:
        if not os.path.exists(path):
            print(f"[skip] {path}")
            continue
        for k in ks:
            for pw, m, kk, d, S in topk_sets(path, label, k):
                if pw not in protos:
                    continue
                h, w, full = best_hit(S, protos[pw])
                rows.append({"pathway": pw, "method": m, "k": k,
                             "delta": float("nan"), "set": sorted(S),
                             "h": h, "best_protocol": w,
                             "full": full[0] if full else None,
                             "full_size": full[1] if full else 0})

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(SEED)
    base = {(pw, k): random_baseline(protos[pw], k, args.n_vocab, rng)
            for k in ks for pw in sorted(protos)}
    methods = sorted({r["method"] for r in rows})

    print("\n" + "=" * 88)
    print("  1. Main table: agreement with published cocktails  h = max_j |S ∩ C_j|")
    print("=" * 88)
    for k in ks:
        print(f"\n  k = {k}")
        print(f"  {'method':26s} {'sets':>6s} {'hit rate':>9s} {'random':>9s} "
              f"{'enrich':>7s} {'h>=2':>7s} {'random':>9s} {'complete':>8s} "
              f"{'random':>9s}")
        print("  " + "-" * 102)
        for m in methods:
            sub = [r for r in rows if r["k"] == k and r["method"] == m]
            if not sub:
                continue
            hr = sum(1 for r in sub if r["h"] >= 1) / len(sub)
            h2 = sum(1 for r in sub if r["h"] >= 2) / len(sub)
            cnt = defaultdict(int)
            for r in sub:
                cnt[r["pathway"]] += 1
            tot = sum(cnt.values())
            hf = sum(1 for r in sub if r["full"]) / len(sub)
            e1 = sum(base[(pw, k)][0] * c for pw, c in cnt.items()) / tot
            e2 = sum(base[(pw, k)][1] * c for pw, c in cnt.items()) / tot
            ef = sum(base[(pw, k)][2] * c for pw, c in cnt.items()) / tot
            print(f"  {m:26s} {len(sub):6d} {hr:8.1%} {e1:9.2%} "
                  f"{(hr / e1 if e1 else float('nan')):7.1f} {h2:7.1%} "
                  f"{e2:9.2%} {hf:8.1%} {ef:9.3%}")
    print("\n  hit rate = fraction of sets containing one or more factors of any published cocktail.")
    print("  complete = fraction containing all factors of some cocktail. Single-factor cocktails")
    print("  are excluded (drawing that one gene satisfies it automatically).")
    print("  The random columns are expectations under the same pathway composition and max-over-protocols procedure.")

    print("\n" + "=" * 88)
    print("  1b. Comparison with one set per pathway (18 vs 18)")
    print("=" * 88)
    print("  A search method takes, per pathway, the one set with the largest Delta. This corresponds to actual use")
    print("  and puts it on the same footing as baselines with one set per pathway.\n")
    for k in ks:
        print(f"  k = {k}")
        print(f"  {'method':26s} {'paths':>5s} {'hit':>6s} {'h>=2':>6s} "
              f"{'complete':>8s}")
        print("  " + "-" * 56)
        for m in methods:
            per = {}
            for r in rows:
                if r["k"] != k or r["method"] != m:
                    continue
                cur = per.get(r["pathway"])
                key = r["delta"] if r["delta"] == r["delta"] else 0.0
                if cur is None or key > cur[0]:
                    per[r["pathway"]] = (key, r)
            if not per:
                continue
            rs = [v[1] for v in per.values()]
            print(f"  {m + ' (max Δ)':26s} {len(rs):5d} "
                  f"{sum(1 for r in rs if r['h'] >= 1) / len(rs):5.1%} "
                  f"{sum(1 for r in rs if r['h'] >= 2) / len(rs):5.1%} "
                  f"{sum(1 for r in rs if r['full']) / len(rs):7.1%}")

            # Consensus set: the top k genes by frequency over all sets of that pathway.
            # Reporting only one of 60 sets throws information away.
            # A natural summary that averages out the cell-to-cell variation.
            cons = {}
            for pw in sorted(protos):
                sub = [r for r in rows if r["k"] == k and r["method"] == m
                       and r["pathway"] == pw]
                if len(sub) < 2:
                    continue
                cnt2 = defaultdict(int)
                for r in sub:
                    for g in r["set"]:
                        cnt2[g] += 1
                top = sorted(cnt2, key=lambda g: (-cnt2[g], g))[:k]
                cons[pw] = best_hit(set(top), protos[pw])
            if cons:
                v = list(cons.values())
                print(f"  {m + ' (consensus)':26s} {len(v):5d} "
                      f"{sum(1 for x in v if x[0] >= 1) / len(v):5.1%} "
                      f"{sum(1 for x in v if x[0] >= 2) / len(v):5.1%} "
                      f"{sum(1 for x in v if x[2]) / len(v):7.1%}")
        print()
        print("  max Delta is the one set with the best objective. Consensus is the top k genes that are")
        print("  frequent across all sets of that pathway. Methods with one set per pathway have no consensus row.")

    print("\n" + "=" * 88)
    print("  2. Contrast between methods (sign test with the pathway as the unit)")
    print("=" * 88)
    ref = "gcg_faithful"
    for k in ks:
        if not any(r["k"] == k and r["method"] == ref for r in rows):
            continue
        print(f"\n  k = {k}   reference: {ref}")
        print(f"  {'other':26s} {'paths':>5s} {'ref higher':>9s} {'tie':>5s} "
              f"{'p(sign)':>9s}")
        print("  " + "-" * 62)
        for m in methods:
            if m == ref:
                continue
            a, b = [], []
            for pw in sorted(protos):
                ra = [r["h"] for r in rows if r["k"] == k
                      and r["method"] == ref and r["pathway"] == pw]
                rb = [r["h"] for r in rows if r["k"] == k
                      and r["method"] == m and r["pathway"] == pw]
                if ra and rb:
                    a.append(st.mean(ra))
                    b.append(st.mean(rb))
            if not a:
                continue
            win = sum(1 for x, y in zip(a, b) if x > y)
            tie = sum(1 for x, y in zip(a, b) if x == y)
            n = len(a) - tie
            p = binom_two_sided(max(win, n - win), n) if n else float("nan")
            print(f"  {m:26s} {len(a):5d} {f'{win}/{n}':>9s} {tie:5d} {p:9.4g}")
        print("  A tie is a pathway where the mean h agrees (mostly both 0).")

    print("\n" + "=" * 88)
    print("  3. Relation between the objective and agreement within a method (supplementary)")
    print("=" * 88)
    print(f"  Decision criterion (fixed before looking at the results): \"related\" = "
          f"|median rho| > {args.rho_null} and sign test p < 0.05.")
    print("  If both are not satisfied, \"unrelated\". Nothing is claimed from only one.\n")
    print(f"  {'k':>2s} {'method':26s} {'paths':>8s} {'median rho':>10s} "
          f"{'pos':>7s} {'p(sign)':>9s} {'verdict':>8s}")
    print("  " + "-" * 82)
    any3 = False
    corr = []
    for k in ks:
        for m in methods:
            rs = []
            for pw in sorted(protos):
                sub = [r for r in rows if r["k"] == k and r["method"] == m
                       and r["pathway"] == pw and r["delta"] == r["delta"]]
                if len(sub) < 5 or len({r["h"] for r in sub}) < 2:
                    continue
                rho = spearman([r["delta"] for r in sub],
                               [r["h"] for r in sub])
                if rho == rho:
                    rs.append(rho)
            if len(rs) < 3:
                continue
            any3 = True
            pos = sum(1 for x in rs if x > 0)
            med = st.median(rs)
            p_sign = binom_two_sided(max(pos, len(rs) - pos), len(rs))
            verdict = ("related" if abs(med) > args.rho_null
                       and p_sign < 0.05 else "unrelated")
            corr.append({"k": k, "method": m, "n_pathways": len(rs),
                         "rho_median": med, "n_positive": pos,
                         "p_sign": p_sign, "verdict": verdict,
                         "rho_null": args.rho_null})
            print(f"  {k:2d} {m:26s} {len(rs):8d} {med:10.3f} "
                  f"{f'{pos}/{len(rs)}':>7s} {p_sign:9.4g} {verdict:>8s}")
    if not any3:
        print("  Too few pathways with variation in h; the correlation could not be computed.")
        print("  Read the hit rate of the main table instead.")

    print("\n" + "=" * 88)
    print("  4. Hit rate per pathway")
    print("=" * 88)
    srch = [m for m in methods if m in SEARCH_METHODS]
    for k in ks:
        print(f"\n  k = {k}")
        print(f"  {'path':5s} {'protocol':>8s} {'random':>7s}  "
              + "  ".join(f"{m[:12]:>12s}" for m in srch))
        print("  " + "-" * (24 + 14 * max(1, len(srch))))
        for pw in sorted(protos):
            cells = []
            for m in srch:
                sub = [r for r in rows if r["k"] == k and r["method"] == m
                       and r["pathway"] == pw]
                cells.append(
                    f"{sum(1 for r in sub if r['h'] >= 1) / len(sub):11.1%}"
                    if sub else f"{'-':>12s}")
            print(f"  {pw:5s} {len(protos[pw]):6d} {base[(pw, k)][0]:7.2%}  "
                  + "  ".join(cells))

    print("\n" + "=" * 88)
    print("  5. Examples of sets that matched a published cocktail")
    print("=" * 88)
    seen = defaultdict(list)
    thr = 2 if any(r["h"] >= 2 for r in rows) else 1
    for r in rows:
        if r["h"] >= thr:
            seen[(r["pathway"], r["k"], r["method"])].append(r)
    if not seen:
        print("  No set contained a factor of a published cocktail.")
    for (pw, k, m), rs in sorted(seen.items())[:24]:
        b = max(rs, key=lambda r: (r["h"],
                                   r["delta"] if r["delta"] == r["delta"]
                                   else -1e9))
        C = dict(protos[pw]).get(b["best_protocol"], set())
        print(f"  {pw} k={k} {m:22s} h={b['h']}  "
              f"cocktail {b['best_protocol']} {sorted(C)}")
        print(f"        set {b['set']}")
        print(f"        match {sorted(set(b['set']) & C)}")

    json.dump(corr, open(os.path.join(args.out, "correlation.json"), "w"),
              ensure_ascii=False, indent=1)
    json.dump(rows, open(os.path.join(args.out, "rows.json"), "w"),
              ensure_ascii=False)
    json.dump({f"{pw}_k{k}": v for (pw, k), v in base.items()},
              open(os.path.join(args.out, "random_baseline.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"\n  [saved] {args.out}/rows.json, {args.out}/random_baseline.json,"
          f" {args.out}/correlation.json")


if __name__ == "__main__":
    main()
