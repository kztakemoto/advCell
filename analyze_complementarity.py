#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Recompute the numbers of the supplementary "complementarity" paragraph from the result JSONs.

The paragraph quotes the Spearman correlation, the number of factors that improve,
the median ranks, the factors rising into the top 10% and the AUC of the rank average.
They are generated here, like the main table and DIVERGE.

Definitions of the quantities:
  rank      per pathway, the rank in descending score order (1 is the top; ties get the mean rank)
  lower half  chosen with --split.
            vocab    : pairs whose baseline rank is below half the vocabulary size.
                       Refers to "factors the baseline really misses".
            curated  : the lower half of the curated factors ordered by baseline rank.
                       Always exactly half (43 pairs when there are 86).
                       The "43 factors" in the text came from this definition.
  top 10%   rank within 10% of the vocabulary size
  Spearman  rank correlation of the two methods over the curated factors of all pathways pooled
  rank mean  average the two methods' ranks per pathway, use it as a new score,
            and compute two kinds of AUC.
              conventional     own-pathway GT vs the whole vocabulary
              pathway-specific own-pathway GT vs the GT of other pathways (own ones excluded)
            The negatives are built as in eval_specificity.py. The text states that "the rank
            mean beats both methods in the conventional AUC but does not beat the baseline
            alone in the pathway-specific one", so both are needed.

Usage:
    python analyze_complementarity.py \\
        --method "Geneformer I-GPR(20)" results_kadv/k4/multi_pathway_geneformer_k4_n50.json \\
        --baseline "DEG(target_expr)" deg_baseline_results/deg_baseline_target_expr.json \\
        --baseline "DEG(wilcoxon)"    deg_baseline_results/deg_baseline_wilcoxon.json \\
        --config data/pathway_config_geneformer.json \\
        --step 20 --out complementarity.json
"""
import argparse
import json
import math
import statistics as st


def load_scores(path, step):
    """Take the per-pathway score dictionaries from results[pw].

    Searched in the same order as eval_specificity.py:
      step_<n>.mean_scores -> step_<n>.scores -> scores
    """
    j = json.load(open(path, encoding="utf-8"))
    res = j.get("results", j)
    key = f"step_{step}"
    out = {}
    for pw, node in res.items():
        if not isinstance(node, dict):
            continue
        sub = node.get(key)
        sc = None
        if isinstance(sub, dict):
            sc = sub.get("mean_scores") or sub.get("scores")
        if not sc:
            sc = node.get("scores")
        if sc:
            out[pw] = sc
    return out


def ranks_of(scores, genes):
    """Descending rank. Ties get the mean rank."""
    order = sorted(scores, key=lambda g: -scores[g])
    r, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            r[order[t]] = avg
        i = j + 1
    return {g: r[g] for g in genes if g in r}


def spearman(x, y):
    if len(x) < 3:
        return float("nan")

    def rk(v):
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

    rx, ry = rk(x), rk(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def auc(pos, neg):
    """Fraction of positive scores exceeding negative scores. Ties count 0.5."""
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for p in pos:
        for n in neg:
            s += 1.0 if p > n else (0.5 if p == n else 0.0)
    return s / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", nargs=2, required=True,
                    metavar=("LABEL", "PATH"))
    ap.add_argument("--baseline", nargs=2, action="append", required=True,
                    metavar=("LABEL", "PATH"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--split", choices=["vocab", "curated", "both"],
                    default="both",
                    help="Definition of the \"lower half\". Default: print both")
    ap.add_argument("--norms", default=None,
                    help="JSON of gene -> embedding norm. "
                         "If given, also compute the control divided by the norm")
    ap.add_argument("--out", default="complementarity.json")
    args = ap.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    gt = {p["id"]: set(p["tf_union"]) for p in cfg["pathways"]}

    m_label, m_path = args.method
    M = load_scores(m_path, args.step)
    norms = json.load(open(args.norms, encoding="utf-8")) if args.norms else None

    report = {"method": m_label, "step": args.step, "baselines": {}}
    print(f"Method: {m_label}   curated: total "
          f"{sum(len(v) for v in gt.values())}, unique "
          f"{len(set().union(*gt.values()))}")

    for b_label, b_path in args.baseline:
        B = load_scores(b_path, args.step)
        pws = sorted(set(M) & set(B) & set(gt))
        rows = []
        for pw in pws:
            common = set(M[pw]) & set(B[pw])
            g = sorted(gt[pw] & common)
            if not g:
                continue
            n_vocab = len(common)
            rm = ranks_of({k: M[pw][k] for k in common}, g)
            rb = ranks_of({k: B[pw][k] for k in common}, g)
            for x in g:
                rows.append({"pathway": pw, "gene": x,
                             "rank_method": rm[x], "rank_base": rb[x],
                             "n_vocab": n_vocab})
        rho = spearman([r["rank_method"] for r in rows],
                       [r["rank_base"] for r in rows])
        print(f"\n=== {b_label} ===")
        print(f"  (pathway, factor) pairs in curated: {len(rows)}")
        print(f"  Spearman correlation of ranks: {rho:.3f}")

        splits = (["vocab", "curated"] if args.split == "both"
                  else [args.split])
        per_split = {}
        for sp in splits:
            if sp == "vocab":
                low = [r for r in rows if r["rank_base"] > r["n_vocab"] / 2]
                desc = "baseline rank in the lower half of the vocabulary"
            else:
                srt = sorted(rows, key=lambda r: -r["rank_base"])
                low = srt[:len(rows) // 2]
                desc = "lower half of the curated factors ordered by baseline rank"
            better = [r for r in low if r["rank_method"] < r["rank_base"]]
            lifted = [r for r in low
                      if r["rank_method"] <= r["n_vocab"] * 0.10]
            worse = [r for r in rows if r not in low
                     and r["rank_method"] > r["rank_base"]
                     and r["rank_method"] > r["n_vocab"] / 2]
            print(f"\n  --split {sp}: {desc}")
            print(f"    lower half: {len(low)} pairs")
            print(f"    of which this method ranks higher: {len(better)}")
            print(f"    median rank {st.median([r['rank_base'] for r in low]):.0f}"
                  f" -> {st.median([r['rank_method'] for r in low]):.0f}")
            print(f"    from the lower half into the top 10% of the vocabulary: {len(lifted)}")
            for r in sorted(lifted, key=lambda r: r["rank_method"]):
                print(f"      {r['gene']:9s} {r['pathway']}  "
                      f"{r['rank_base']:.0f} -> {r['rank_method']:.0f}")
            print(f"    reverse (in the upper half, dropping into the lower half of the vocabulary): "
                  f"{len(worse)}")
            for r in worse:
                print(f"      {r['gene']:9s} {r['pathway']}  "
                      f"{r['rank_base']:.0f} -> {r['rank_method']:.0f}")
            per_split[sp] = {
                "n_lower_half": len(low), "n_method_better": len(better),
                "median_rank_base": st.median([r["rank_base"] for r in low]),
                "median_rank_method": st.median([r["rank_method"]
                                                 for r in low]),
                "lifted": [{"gene": r["gene"], "pathway": r["pathway"],
                            "rank_base": r["rank_base"],
                            "rank_method": r["rank_method"]}
                           for r in lifted],
                "moved_down": [{"gene": r["gene"], "pathway": r["pathway"]}
                               for r in worse]}
        low = [r for r in rows if r["rank_base"] > r["n_vocab"] / 2]
        lifted = [r for r in low if r["rank_method"] <= r["n_vocab"] * 0.10]

        # Ensemble by rank mean. For each of the 3 methods,
        # print the conventional AUC and the pathway-specific AUC.
        others = {pw: set().union(*[gt[q] for q in pws if q != pw])
                  for pw in pws}
        ens, me, ba = {}, {}, {}
        for lab, getter in (("rank average", None),
                            ("method alone", "m"),
                            ("baseline alone", "b")):
            conv, spec = [], []
            for pw in pws:
                common = set(M[pw]) & set(B[pw])
                g = gt[pw] & common
                if not g:
                    continue
                rm = ranks_of({k: M[pw][k] for k in common}, common)
                rb = ranks_of({k: B[pw][k] for k in common}, common)
                if getter == "m":
                    sc = {k: -rm[k] for k in common}
                elif getter == "b":
                    sc = {k: -rb[k] for k in common}
                else:
                    sc = {k: -(rm[k] + rb[k]) / 2.0 for k in common}
                neg_all = common - g
                neg_oth = (others[pw] & common) - g
                conv.append(auc([sc[k] for k in g],
                                [sc[k] for k in neg_all]))
                if neg_oth:
                    spec.append(auc([sc[k] for k in g],
                                    [sc[k] for k in neg_oth]))
            d = {"conventional": st.mean(conv),
                 "pathway_specific": st.mean(spec) if spec else float("nan")}
            if lab == "rank average":
                ens = d
            elif lab == "method alone":
                me = d
            else:
                ba = d
        print(f"\n  {'':16s} {'conventional AUC':>16s} {'pathway-specific':>18s}")
        for lab, d in (("this method only", me), ("baseline only", ba),
                       ("rank mean", ens)):
            print(f"  {lab:16s} {d['conventional']:10.3f} "
                  f"{d['pathway_specific']:18.3f}")
        better_conv = (ens["conventional"] > max(me["conventional"],
                                                 ba["conventional"]))
        better_spec = (ens["pathway_specific"] > max(me["pathway_specific"],
                                                     ba["pathway_specific"]))
        print(f"  Does the rank mean beat both methods: "
              f"conventional {'yes' if better_conv else 'no'}, "
              f"pathway-specific {'yes' if better_spec else 'no'}")
        a_vals = [ens["conventional"]]

        ent = {"spearman": rho, "n_pairs": len(rows), "splits": per_split,
               "auc": {"method_alone": me, "baseline_alone": ba,
                       "rank_average": ens},
               "rank_average_beats_both": {"conventional": better_conv,
                                           "pathway_specific": better_spec}}

        if norms:
            nrows = []
            for pw in pws:
                common = {k for k in set(M[pw]) & set(B[pw])
                          if norms.get(k)}
                g = sorted(gt[pw] & common)
                if not g:
                    continue
                sc = {k: M[pw][k] / norms[k] for k in common}
                rn = ranks_of(sc, g)
                rb = ranks_of({k: B[pw][k] for k in common}, g)
                for x in g:
                    nrows.append({"pathway": pw, "gene": x,
                                  "rank_method": rn[x], "rank_base": rb[x],
                                  "n_vocab": len(common)})
            nlow = [r for r in nrows if r["rank_base"] > r["n_vocab"] / 2]
            nlift = [r for r in nlow
                     if r["rank_method"] <= r["n_vocab"] * 0.10]
            print(f"  Control divided by the embedding norm: from the lower half into the top 10% "
                  f"{len(nlift)} (without dividing: {len(lifted)})")
            ent["lifted_norm_divided"] = len(nlift)
        else:
            print("  (pass --norms to also print the control divided by the embedding norm)")

        report["baselines"][b_label] = ent

    json.dump(report, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
