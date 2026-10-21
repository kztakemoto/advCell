#!/usr/bin/env python3
"""
Summarize additivity.json into a lightweight file
====================================
analyze_additivity.py stores the full raw data of set_rows (cell x random sample x k),
which exceeds 100MB for 18 pathways x 50 cells.
Analysis needs only the per-pathway aggregates, so only those are extracted.

Usage:
  python summarize_additivity.py additivity_results_18pw/additivity.json \
      -o additivity_summary.json
"""

import argparse
import json
import statistics as st


def fit(a, p):
    n = len(a)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    ma, mp = st.mean(a), st.mean(p)
    sxy = sum((x - mp) * (y - ma) for x, y in zip(p, a))
    sxx = sum((x - mp) ** 2 for x in p)
    syy = sum((y - ma) ** 2 for y in a)
    slope = sxy / sxx if sxx else float("nan")
    r = sxy / (sxx * syy) ** 0.5 if sxx > 0 and syy > 0 else float("nan")
    return slope, r, st.mean([x - y for x, y in zip(a, p)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-o", "--output", default="additivity_summary.json")
    ap.add_argument("--keep_top", type=int, default=50,
                    help="How many of the top additive-ranking entries to keep")
    args = ap.parse_args()

    with open(args.path) as f:
        d = json.load(f)
    R = d.get("results", d)

    out = {"config": d.get("config", {}), "results": {}}
    for pw, r in R.items():
        e = {k: r[k] for k in
             ("n_cells", "base_cos_mean", "delta_mean", "delta_max",
              "delta_min", "additive_ranking_auc") if k in r}
        e["additive_ranking_ranks"] = r.get("additive_ranking_ranks", {})

        # keep only the top of the additive ranking (all 2632 are not needed)
        sc = r.get("additive_ranking_scores") or {}
        if sc:
            top = sorted(sc.items(), key=lambda x: -x[1])[:args.keep_top]
            e["additive_ranking_top"] = [[n, v] for n, v in top]

        # aggregate set_rows, then discard them
        rows = r.get("set_rows") or []
        summ = []
        for k in sorted({x["k"] for x in rows}):
            for kind in ("additive_top", "random"):
                sub = [x for x in rows if x["k"] == k and x["kind"] == kind]
                if not sub:
                    continue
                a = [x["actual"] for x in sub]
                p = [x["additive"] for x in sub]
                slope, r_p, resid = fit(a, p)
                summ.append({"k": k, "kind": kind, "n": len(sub),
                             "actual": st.mean(a), "actual_sd": st.pstdev(a),
                             "additive": st.mean(p), "resid": resid,
                             "slope": slope, "pearson": r_p})
        e["summary"] = summ or r.get("summary", [])

        # greedy sets are small, so keep the aggregate plus the frequency of chosen TFs
        g = r.get("greedy_rows") or []
        if g:
            gs = []
            for k in sorted({x["k"] for x in g}):
                sub = [x for x in g if x["k"] == k]
                a = [x["actual"] for x in sub]
                freq = {}
                for x in sub:
                    for t in x.get("tfs", []):
                        freq[t] = freq.get(t, 0) + 1
                gs.append({
                    "k": k, "n": len(sub),
                    "actual": st.mean(a), "actual_sd": st.pstdev(a),
                    "additive": st.mean([x["additive"] for x in sub]),
                    "top_tfs": sorted(freq.items(), key=lambda x: -x[1])[:20],
                })
            e["greedy_summary"] = gs
        out["results"][pw] = e

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    import os
    a = os.path.getsize(args.path) / 1e6
    b = os.path.getsize(args.output) / 1e6
    print(f"Summarized {len(out['results'])} pathways")
    print(f"  {args.path}: {a:.1f} MB")
    print(f"  {args.output}: {b:.2f} MB  (1/{a / b if b else 0:.0f} of the original)")


if __name__ == "__main__":
    main()
