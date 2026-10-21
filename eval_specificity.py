#!/usr/bin/env python3
"""
Evaluate any ranking by AUC and pathway specificity (general-purpose)
=====================================================
Whether it comes from DEG, ISP or this method, anything with per-pathway TF scores
can be compared on the same footing.

Metrics:
  auc         own-pathway GT vs all TFs (the usual evaluation; however,
              it is inflated by the signal of "TFs that look generally important")
  spec        own-pathway GT vs other-pathway GT
              (the negatives are also master regulators, so general importance cancels
                and only pathway specificity remains. 0.5 means no discrimination)

The GT sets are taken from the result JSON of this method (the ranks keys of each step).
This guarantees that the definition of GT is identical across methods.

--------------------------------------------------------------------
Ways to give the input (3 kinds, may be mixed)

1) Result JSON of this method (one that stores mean_scores)
     --results results_kadv/k4/multi_pathway_geneformer_k4_n50.json

2) A score file per pathway
     --scores P01 deg/P01.tsv --scores P02 deg/P02.tsv ...
     --scores_glob "deg/{pw}.tsv"        # {pw} is replaced by the pathway ID

   Supported formats (detected automatically from the extension and content):
     - TSV/CSV : a gene-name column and a score column in the header
                 (gene/symbol/tf/name and score/value/logfc/stat/pval, etc.)
     - JSON    : {"GATA4": 3.2, "MYOD1": 1.1, ...}
     - text    : a rank-ordered list with one gene per line (score = -rank)

3) Built-in baselines
     --builtin norm --model_dir ... --dict_dir ... --tf_list ...
        embedding norm only (same ranking for all pathways; specificity should be 0.5)
     --builtin random
        random numbers (control)

--------------------------------------------------------------------
Examples:
  # this method vs DEG
  python eval_specificity.py \
      --gt_from results_kadv/k4/multi_pathway_geneformer_k4_n50.json \
      --results results_kadv/k4/multi_pathway_geneformer_k4_n50.json \
      --label_results "I-GPR(20)" --step 20 \
      --scores_glob "deg_results/{pw}_deg.tsv" --label_scores DEG
"""

import argparse
import csv
import glob
import json
import os
import random
import statistics as st


# ------------------------------------------------------------
def auc_vs_all(scores, positives):
    """Own-pathway GT vs all TFs. Ties count 0.5."""
    pos = [v for k, v in scores.items() if k in positives]
    neg = [v for k, v in scores.items() if k not in positives]
    return _auc(pos, neg)


def auc_vs_other(scores, own_gt, other_gt):
    """Own-pathway GT vs other-pathway GT (those overlapping own-pathway GT are removed from the negatives)."""
    own = set(own_gt)
    pos = [scores[g] for g in own if g in scores]
    neg = [scores[g] for g in other_gt if g in scores and g not in own]
    return _auc(pos, neg)


def _auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    import bisect
    s = sorted(neg)
    tot = 0.0
    for v in pos:
        lo = bisect.bisect_left(s, v)
        hi = bisect.bisect_right(s, v)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


# ------------------------------------------------------------
NAME_COLS = ("gene", "genes", "gene_name", "symbol", "tf", "name", "names",
             "gene_symbol", "feature", "index")
SCORE_COLS = ("score", "value", "stat", "statistic", "logfc", "log2fc",
              "log2foldchange", "logfoldchanges", "scores", "zscore", "rank")


def load_score_file(path):
    """Read a score file into {name: score}. The format is detected automatically."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path) as f:
            d = json.load(f)
        if isinstance(d, dict):
            return {str(k): float(v) for k, v in d.items()}
        if isinstance(d, list):
            return {str(x): -float(i) for i, x in enumerate(d)}
        raise ValueError(f"{path}: unrecognized JSON")

    with open(path, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = "\t" if "\t" in sample.split("\n")[0] else ","
        rows = list(csv.reader(f, delimiter=delim))

    if not rows:
        raise ValueError(f"{path}: empty")

    header = [h.strip().lower() for h in rows[0]]
    n_col = next((i for i, h in enumerate(header) if h in NAME_COLS), None)
    s_col = next((i for i, h in enumerate(header) if h in SCORE_COLS), None)

    # no header and only one column -> treat as a rank-ordered list
    if n_col is None and len(header) == 1:
        names = [r[0].strip() for r in rows if r and r[0].strip()]
        return {n: -float(i) for i, n in enumerate(names)}

    if n_col is None:
        n_col = 0
    if s_col is None:
        # if no score column can be identified, treat row order as rank
        out = {}
        for i, r in enumerate(rows[1:]):
            if r and r[n_col].strip():
                out[r[n_col].strip()] = -float(i)
        return out

    out = {}
    for r in rows[1:]:
        if len(r) <= max(n_col, s_col):
            continue
        name = r[n_col].strip()
        try:
            val = float(r[s_col])
        except ValueError:
            continue
        if name:
            out[name] = val
    if not out:
        raise ValueError(f"{path}: could not read the scores (check the column names)")
    return out


# ------------------------------------------------------------
def load_ranking_json(path, step):
    """Extract {pathway: {tf: score}} from a result JSON.

    The keys are searched in the following order, so any format of this method,
    scGPT, scFoundation or DEG can be read as it is.

      1. results[pw]["step_<step>"]["mean_scores"]   this method
      2. results[pw]["step_<step>"]["scores"]        scGPT / scFoundation
      3. results[pw]["scores"]                       DEG (no hierarchy)
    """
    with open(path) as f:
        d = json.load(f)
    # supports both the {"config":..., "results":{...}} form and the form where the pathway dictionary comes directly
    R = d.get("results") if isinstance(d.get("results"), dict) else d
    key = f"step_{step}"
    out, used = {}, None
    for pw, r in R.items():
        if not isinstance(r, dict):
            continue
        sub = r.get(key) or {}
        for src, tag in ((sub.get("mean_scores"), f"{key}.mean_scores"),
                         (sub.get("scores"), f"{key}.scores"),
                         (r.get("scores"), "scores")):
            if src:
                out[pw] = {k: float(v) for k, v in src.items()}
                used = used or tag
                break
    return out, used


def evaluate(name, scores_by_pw, gt_by_pw, universe):
    """Evaluate one method over all pathways."""
    rows = []
    for pw in sorted(gt_by_pw):
        sc = scores_by_pw.get(pw)
        if not sc:
            continue
        own = gt_by_pw[pw]
        other = sorted(set().union(*[g for q, g in gt_by_pw.items()
                                     if q != pw]) - set(own))
        # the AUC over all TFs is computed on a common TF set
        sc_u = {k: v for k, v in sc.items() if k in universe}
        missing = [g for g in own if g not in sc_u]
        rows.append({
            "pathway": pw,
            "n_gt": len(own),
            "n_scored": len(sc_u),
            "n_missing_gt": len(missing),
            "auc": auc_vs_all(sc_u, own),
            "spec": auc_vs_other(sc_u, own, other),
        })
    return {"name": name, "rows": rows}


def print_result(res):
    rows = res["rows"]
    if not rows:
        print(f"\n[{res['name']}] no pathway can be evaluated")
        return
    print(f"\n[{res['name']}]")
    print(f"  {'Pathway':9s} {'nGT':>4s} {'scoredTF':>8s} {'missGT':>6s} "
          f"{'auc':>8s} {'spec':>8s} {'auc-spec':>9s}")
    print("  " + "-" * 58)
    for r in rows:
        gap = r["auc"] - r["spec"]
        print(f"  {r['pathway']:9s} {r['n_gt']:4d} {r['n_scored']:7d} "
              f"{r['n_missing_gt']:6d} "
              f"{r['auc']:8.3f} {r['spec']:8.3f} {gap:+9.3f}")
    print("  " + "-" * 58)
    a = [r["auc"] for r in rows if r["auc"] == r["auc"]]
    s = [r["spec"] for r in rows if r["spec"] == r["spec"]]
    print(f"  {'mean':9s} {'':4s} {'':7s} {'':6s} "
          f"{st.mean(a):8.3f} {st.mean(s):8.3f} "
          f"{st.mean(a) - st.mean(s):+9.3f}")
    ns = {r["n_scored"] for r in rows}
    if len(ns) > 1 or (len(ns) == 1 and next(iter(ns)) < 1000):
        print(f"  Note: number of scored TFs: {min(ns)}-{max(ns)}. "
              f"If it differs greatly between methods, the auc comparison is unfair.")
    print(f"  Pathways with spec > 0.5: {sum(1 for x in s if x > 0.5)}/{len(s)}")
    miss = sum(r["n_missing_gt"] for r in rows)
    if miss:
        print(f"  Note: {miss} GT TFs in total were absent from the scores "
              f"(excluded from the evaluation)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_from", required=True,
                    help="result JSON from which the GT sets are taken (output of this method)")
    ap.add_argument("--step", type=int, default=20,
                    help="number of I-GPR steps when reading from --results (0=GPR)")
    ap.add_argument("--results", nargs=2, action="append", default=[],
                    metavar=("LABEL", "PATH"),
                    help="--results <label> <path>, repeatable. "
                         "result JSON that has mean_scores")
    ap.add_argument("--scores", nargs=2, action="append", metavar=("PW", "PATH"),
                    default=[], help="--scores P01 path/to/file, repeatable")
    ap.add_argument("--scores_glob", default=None,
                    help="a pattern containing {pw}, such as 'deg/{pw}.tsv'")
    ap.add_argument("--label_scores", default="baseline")
    ap.add_argument("--deg", nargs=2, action="append", default=[],
                    metavar=("LABEL", "PATH"),
                    help="same as --results. Kept for compatibility")
    ap.add_argument("--builtin", choices=["norm", "random"], default=None)
    ap.add_argument("--model_dir", default=None)
    ap.add_argument("--dict_dir", default=None)
    ap.add_argument("--tf_list", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pathways", nargs="+", default=None,
                    help="restrict the pathways to evaluate (e.g. P01 P02 P03 P04). "
                         "If the number of pathways differs between methods, the means are not comparable unless they are aligned.")
    ap.add_argument("--common_only", action="store_true",
                    help="aggregate over only the pathways common to all methods")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    # ---- GT sets ----
    with open(args.gt_from) as f:
        base = json.load(f)
    R = base["results"]
    gt_by_pw = {}
    for pw, r in R.items():
        ranks = (r.get("step_0") or {}).get("ranks") or {}
        if ranks:
            gt_by_pw[pw] = sorted(ranks)
    if args.pathways:
        gt_by_pw = {p: g for p, g in gt_by_pw.items() if p in args.pathways}
    print(f"GT: {len(gt_by_pw)} pathways, total "
          f"{sum(len(v) for v in gt_by_pw.values())} TFs, "
          f"{len(set().union(*gt_by_pw.values()))} unique TFs")

    results = []
    universe = None

    # ---- result JSONs (--results / --deg go through the same loader) ----
    for label, path in list(args.results) + list(args.deg):
        sc, used = load_ranking_json(path, args.step)
        if not sc:
            print(f"\n*** could not read scores from {path}.")
            print("    step_<n>.mean_scores / step_<n>.scores / scores "
                  "one of them is required.")
            continue
        print(f"  [{label}] {len(sc)} pathways, keys read: {used}")
        if universe is None:
            universe = set(next(iter(sc.values())))
        results.append(evaluate(label, sc, gt_by_pw, universe))

    # ---- from files ----
    file_map = dict(args.scores)
    if args.scores_glob:
        for pw in gt_by_pw:
            cand = args.scores_glob.replace("{pw}", pw)
            hits = glob.glob(cand)
            if hits:
                file_map.setdefault(pw, hits[0])
    if file_map:
        sc = {}
        for pw, path in sorted(file_map.items()):
            try:
                sc[pw] = load_score_file(path)
            except Exception as e:
                print(f"  [skip] {pw}: {e}")
        if sc:
            if universe is None:
                universe = set().union(*sc.values())
            results.append(evaluate(args.label_scores, sc, gt_by_pw, universe))

    # ---- built-in ----
    if args.builtin == "norm":
        if not (args.model_dir and args.dict_dir and args.tf_list):
            raise SystemExit("--builtin norm requires "
                             "--model_dir --dict_dir --tf_list")
        import torch
        from transformers import BertModel
        from gpr_geneformer import load_dicts, load_tf_list
        m = BertModel.from_pretrained(args.model_dir)
        E = m.embeddings.word_embeddings.weight.detach()
        td, nd, _, _ = load_dicts(args.dict_dir)
        tfs = load_tf_list(args.tf_list, td, nd)
        norms = {n: float(E[t].norm().item()) for n, t in tfs}
        if universe is None:
            universe = set(norms)
        results.append(evaluate("norm-only",
                                {pw: norms for pw in gt_by_pw},
                                gt_by_pw, universe))
    elif args.builtin == "random":
        if universe is None:
            universe = set().union(*gt_by_pw.values())
        rng = random.Random(args.seed)
        sc = {pw: {t: rng.random() for t in universe} for pw in gt_by_pw}
        results.append(evaluate("random", sc, gt_by_pw, universe))

    if not results:
        print("\nNothing to evaluate. Specify --results or --scores.")
        return

    if args.common_only and len(results) > 1:
        common = set.intersection(*[{r["pathway"] for r in res["rows"]}
                                    for res in results])
        for res in results:
            res["rows"] = [r for r in res["rows"] if r["pathway"] in common]
        print(f"\n[common_only] restricted to the {len(common)} pathways common to all methods: "
              f"{', '.join(sorted(common))}")

    print("\n" + "=" * 62)
    print("  Evaluation results")
    print("  auc  = own-pathway GT vs all TFs   spec = own-pathway GT vs other-pathway GT")
    print("=" * 62)
    for r in results:
        print_result(r)

    if len(results) > 1:
        print("\n" + "=" * 62)
        print(f"  Comparison between methods (mean over {max(len(r['rows']) for r in results)} pathways)")
        print("=" * 62)
        print(f"  {'method':22s} {'auc':>8s} {'spec':>8s} {'auc-spec':>9s} "
              f"{'spec>0.5':>9s}")
        summ = []
        for r in results:
            a = [x["auc"] for x in r["rows"] if x["auc"] == x["auc"]]
            sp = [x["spec"] for x in r["rows"] if x["spec"] == x["spec"]]
            if a and sp:
                summ.append((r["name"], st.mean(a), st.mean(sp),
                             sum(1 for x in sp if x > 0.5), len(sp)))
        for name, a, sp, nz, n in sorted(summ, key=lambda x: -x[2]):
            print(f"  {name:22s} {a:8.3f} {sp:8.3f} {a - sp:+9.3f} "
                  f"{nz:5d}/{n:<3d}")
        print("\n  Sorted by descending spec. The more positive auc-spec is,")
        print("  the more that auc depends on a pathway-independent signal.")
        print("\n  If a method wins on auc but loses on spec, the advantage comes from")
        print("  a signal that is not pathway-specific.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
