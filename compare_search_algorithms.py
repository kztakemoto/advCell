#!/usr/bin/env python3
"""
Matched-budget comparison of search algorithms
==============================
To test the claim that "GCG makes combinatorial optimization practical",
the following must be compared at the same forward budget.

  random_best   : try B random k-tuples and take the best
                  (asks whether the search structure itself is needed)
  hill_random   : random initialization + hill climbing by single-position replacement.
                  Candidates are sampled uniformly from all TFs (no gradient)
                  * This is the direct control for GCG. Budget and acceptance rule are matched,
                    and only the presence of the gradient differs. If there is no difference, the gradient is unnecessary.
  hill_gradient : candidates from the top_k of the gradient projection (GCG as currently implemented)
                  accepted "only on improvement"
  gcg_faithful  : same candidates, but the batch best is accepted unconditionally
                  (faithful to Zou et al. 2023. Allows worsening and escapes local optima)

  greedy        : greedy forward selection. Its budget is large, k x n_TF, so it is shown separately
  exhaustive    : exhaustive search narrowed to the top pool (when --pool is given)
                  reference value for the optimality gap

Usage:
  python compare_search_algorithms.py \\
      --model_dir ./Geneformer/Geneformer-V1-10M \\
      --pathway_config ./data/pathway_config_geneformer.json \\
      --dict_dir ./Geneformer/geneformer/gene_dictionaries_30m \\
      --tf_list ./tf_list_human.txt \\
      --pathways P01 P03 P07 P12 P14 \\
      --n_cells 20 --k_adv 4 --budget 2580 --n_restarts 3 \\
      --greedy --pool 30 \\
      --output_dir ./search_comparison
"""

import argparse
import itertools
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
from advcell_overexpress import (
    overexpress_batch, build_inputs_embeds, set_overexpress_mode,
)
from gpr_geneformer import (
    MAX_SEQ_LEN, load_geneformer_model, load_dicts, load_tf_list,
    prepare_cell_data, compute_target_centroid, forward_cell_embedding,
    set_emb_layer,
)


# ------------------------------------------------------------
@torch.no_grad()
def eval_sets(model, orig, sets, centroid, max_seq_len, bs):
    """Return cos_sim for (N,k) sets of TF tokens."""
    out = []
    for i in range(0, sets.shape[0], bs):
        ids, attn, _ = overexpress_batch(orig, sets[i:i + bs], max_seq_len)
        emb = forward_cell_embedding(model, attn, input_ids=ids)
        out.append(F.cosine_similarity(emb, centroid.unsqueeze(0), dim=1))
    return torch.cat(out)


@torch.no_grad()
def base_cos(model, orig, centroid, max_seq_len):
    ids = orig[:max_seq_len].unsqueeze(0)
    emb = forward_cell_embedding(model, torch.ones_like(ids), input_ids=ids)
    return float(F.cosine_similarity(emb, centroid.unsqueeze(0), dim=1).item())


def grad_topk(model, orig, adv_tokens, centroid, tf_ids, max_seq_len,
              top_k, ban_mask):
    """Return the top_k of the gradient projection at the current set, per position.

    Returns: token id tensor of shape (k, top_k)
    """
    E = model.embeddings.word_embeddings.weight
    ids, attn, _ = overexpress_batch(orig, adv_tokens.view(1, -1), max_seq_len)
    adv_emb = E[adv_tokens].detach().clone().requires_grad_(True)
    combined = build_inputs_embeds(E, ids, adv_tokens.numel(),
                                  adv_emb=adv_emb)
    emb = forward_cell_embedding(model, attn, inputs_embeds=combined)
    loss = -F.cosine_similarity(emb, centroid.unsqueeze(0), dim=1).mean()
    loss.backward()
    if adv_emb.grad is None:
        return None
    scores = -(adv_emb.grad @ E.T)          # (k, V)
    scores[:, ban_mask] = float("-inf")
    return scores.topk(top_k, dim=1).indices  # (k, top_k) = vocab id


# ------------------------------------------------------------
def run_random_best(model, orig, centroid, tf_ids, k, budget,
                    max_seq_len, bs, rng, base):
    n_tf = tf_ids.numel()
    best = -1e9
    best_set = None
    done = 0
    curve = []
    init = None
    while done < budget:
        m = min(bs * 4, budget - done)
        idx = np.stack([rng.choice(n_tf, size=k, replace=False)
                        for _ in range(m)])
        cs = eval_sets(model, orig,
                       tf_ids[torch.tensor(idx, device=tf_ids.device)],
                       centroid, max_seq_len, bs)
        if init is None:
            init = float(cs[0].item()) - base
        bi = int(torch.argmax(cs).item())
        bv = float(cs[bi].item())
        if bv > best:
            best = bv
            best_set = [int(x) for x in idx[bi]]
        done += m
        curve.append((done, best - base))
    return best - base, budget, {"init": init, "curve": curve,
                                 "set": best_set}


def run_hill(model, orig, centroid, tf_ids, k, budget, n_cand,
             max_seq_len, bs, rng, base, mode,
             top_k=256, ban_mask=None, monotone=True):
    """Hill climbing by single-position replacement.

    mode="random"   : candidates uniform over all TFs
    mode="gradient" : candidates from the top_k of the gradient projection
    monotone=True   : accept only on improvement (current implementation)
    monotone=False  : accept the batch best unconditionally (following Zou et al.)
    """
    n_tf = tf_ids.numel()
    cur = torch.tensor(rng.choice(n_tf, size=k, replace=False),
                       dtype=torch.long, device=tf_ids.device)
    cur_tok = tf_ids[cur]
    cur_sim = float(eval_sets(model, orig, cur_tok.view(1, -1), centroid,
                              max_seq_len, bs)[0].item())
    cost = 1
    init_sim = cur_sim
    best_sim = cur_sim
    best_tok = cur_tok.clone()
    curve = [(cost, best_sim - base)]      # (budget spent, best Delta so far)
    steps = max(1, budget // n_cand)

    for _ in range(steps):
        if mode == "gradient":
            tk = grad_topk(model, orig, cur_tok, centroid, tf_ids,
                           max_seq_len, top_k, ban_mask)
            cost += 2                      # backward ~ 2 forwards
            if tk is None:
                break
            pos = rng.integers(0, k, size=n_cand)
            pick = rng.integers(0, tk.shape[1], size=n_cand)
            cand = cur_tok.view(1, -1).repeat(n_cand, 1)
            cand[torch.arange(n_cand), torch.tensor(pos, device=cand.device)] = \
                tk[torch.tensor(pos, device=cand.device),
                   torch.tensor(pick, device=cand.device)]
        else:
            pos = rng.integers(0, k, size=n_cand)
            new = rng.choice(n_tf, size=n_cand)
            cand = cur_tok.view(1, -1).repeat(n_cand, 1)
            cand[torch.arange(n_cand), torch.tensor(pos, device=cand.device)] = \
                tf_ids[torch.tensor(new, device=cand.device)]

        # remove duplicates within a row
        keep = [i for i in range(cand.shape[0])
                if len(set(cand[i].tolist())) == k]
        if not keep:
            continue
        cand = cand[torch.tensor(keep, device=cand.device)]
        sims = eval_sets(model, orig, cand, centroid, max_seq_len, bs)
        cost += cand.shape[0]
        bi = int(torch.argmax(sims).item())
        bs_ = float(sims[bi].item())
        if (not monotone) or bs_ > cur_sim:
            cur_tok = cand[bi].clone()
            cur_sim = bs_
        # With monotone=False, cur_tok can get worse. Take the set at the moment
        # best_sim was updated. It need not coincide with cur_tok at the end of the loop.
        if cur_sim > best_sim:
            best_sim = cur_sim
            best_tok = cur_tok.clone()
        curve.append((cost, best_sim - base))
    # The search runs in token-ID space, so convert back to vocabulary indices before returning.
    # ban restricts candidates to TF tokens, so the lookup always succeeds.
    t2i = {int(t): i for i, t in enumerate(tf_ids.tolist())}
    best_set = [t2i[int(t)] for t in best_tok.tolist()]
    return best_sim - base, cost, {"init": init_sim - base, "curve": curve,
                                   "set": best_set}


def run_greedy(model, orig, centroid, tf_ids, k, max_seq_len, bs, base):
    n_tf = tf_ids.numel()
    chosen, cost = [], 0
    for _ in range(k):
        remain = [i for i in range(n_tf) if i not in chosen]
        cand = torch.tensor([chosen + [i] for i in remain],
                            dtype=torch.long, device=tf_ids.device)
        cs = eval_sets(model, orig, tf_ids[cand], centroid, max_seq_len, bs)
        cost += cand.shape[0]
        chosen.append(remain[int(torch.argmax(cs).item())])
        last = float(cs.max().item())
    return last - base, cost, {"set": list(chosen)}


def run_greedy_pooled(model, orig, centroid, tf_ids, pool, k,
                      max_seq_len, bs, base):
    """Greedy forward selection over a narrowed candidate pool. Cost is k * len(pool)."""
    chosen, cost, last = [], 0, base
    pool = list(pool)
    for _ in range(k):
        remain = [i for i in pool if i not in chosen]
        if not remain:
            break
        cand = torch.tensor([chosen + [i] for i in remain],
                            dtype=torch.long, device=tf_ids.device)
        cs = eval_sets(model, orig, tf_ids[cand], centroid, max_seq_len, bs)
        cost += cand.shape[0]
        bi = int(torch.argmax(cs).item())
        chosen.append(remain[bi])
        last = float(cs[bi].item())
    return last - base, cost, {"set": list(chosen)}


def run_exhaustive(model, orig, centroid, tf_ids, pool_idx, k,
                   max_seq_len, bs, base):
    combos = list(itertools.combinations(pool_idx, k))
    cand = torch.tensor(combos, dtype=torch.long, device=tf_ids.device)
    cs = eval_sets(model, orig, tf_ids[cand], centroid, max_seq_len, bs)
    return float(cs.max().item()) - base, len(combos)


@torch.no_grad()
def order_sensitivity(model, orig, centroid, tf_ids, k, max_seq_len, bs,
                      n_sets, rng, max_perms=64):
    """Measure the spread of cos_sim when the same set is reordered.

    overexpress places the k items at the front "in the given order". Geneformer has absolute
    position embeddings, so the same set in a different order is a different input.
    The search space is not the combinations C(n,k) but the permutations C(n,k)*k!.

    If k! exceeds max_perms, permutations are sampled at random
    (k=8 gives 40,320, so full enumeration is impractical). Because of the sampling,
    the spread obtained is a lower bound of the true spread.
    """
    import math
    n = tf_ids.numel()
    full = math.factorial(k)
    spreads = []
    for _ in range(n_sets):
        base_idx = rng.choice(n, size=k, replace=False).tolist()
        if full <= max_perms:
            perms = list(itertools.permutations(base_idx))
        else:
            seen, perms = set(), []
            while len(perms) < max_perms:
                q = tuple(rng.permutation(base_idx).tolist())
                if q not in seen:
                    seen.add(q); perms.append(q)
        cand = torch.tensor(perms, dtype=torch.long, device=tf_ids.device)
        cs = eval_sets(model, orig, tf_ids[cand], centroid, max_seq_len, bs)
        spreads.append(float((cs.max() - cs.min()).item()))
    return spreads


@torch.no_grad()
def run_exhaustive_full(model, orig, centroid, tf_ids, k,
                        max_seq_len, bs, base, chunk=200_000,
                        ordered=True):
    """Complete exhaustive search over all TFs. Gives the true optimum.

    The objective depends on the insertion order, so permutations are enumerated by default.
      k=2: n*(n-1) = 6,924,792   (3,463,596 for combinations)
    k>=3 is impractical. With ordered=False only combinations are enumerated,
    and the result is merely a lower bound of the true optimum.
    """
    n = tf_ids.numel()
    best = -1e9
    best_set = None
    total = 0
    it = (itertools.permutations(range(n), k) if ordered
          else itertools.combinations(range(n), k))
    while True:
        buf = list(itertools.islice(it, chunk))
        if not buf:
            break
        cand = torch.tensor(buf, dtype=torch.long, device=tf_ids.device)
        cs = eval_sets(model, orig, tf_ids[cand], centroid, max_seq_len, bs)
        i = int(torch.argmax(cs).item())
        if float(cs[i].item()) > best:
            best = float(cs[i].item())
            best_set = [int(x) for x in buf[i]]
        total += len(buf)
        del cand, cs
    return best - base, total, best_set


# ------------------------------------------------------------
CURVE_GRID = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def _interp_curve(curves, budget):
    """Average the (cost, best) curves of several restarts on a grid."""
    out = []
    for g in CURVE_GRID:
        if g > budget:
            break
        vals = []
        for cv in curves:
            v = None
            for c, d in cv:
                if c <= g:
                    v = d
                else:
                    break
            if v is not None:
                vals.append(v)
        if vals:
            out.append([g, st.mean(vals)])
    return out


def paired_test(a, b):
    """Paired t and Wilcoxon signed-rank test for the paired difference a-b.

    Both methods are evaluated on the same cells, so the power is higher than for unpaired tests.
    """
    dif = [x - y for x, y in zip(a, b)]
    n = len(dif)
    if n < 3:
        return {}
    m = st.mean(dif)
    sd = st.stdev(dif) if n > 1 else float("nan")
    se = sd / (n ** 0.5) if sd == sd and sd > 0 else float("nan")
    t = m / se if se == se and se > 0 else float("nan")
    out = {"n": n, "mean_diff": m, "se": se, "t": t,
           "wins": sum(1 for x in dif if x > 0)}
    # two-sided p (normal approximation)
    if t == t:
        import math
        out["p_t"] = math.erfc(abs(t) / (2 ** 0.5))
    try:
        from scipy.stats import wilcoxon
        if any(x != 0 for x in dif):
            w = wilcoxon(dif, alternative="two-sided")
            out["p_wilcoxon"] = float(w.pvalue)
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--pathway_config", required=True)
    ap.add_argument("--dict_dir", required=True)
    ap.add_argument("--tf_list", required=True)
    ap.add_argument("--pathways", nargs="+", default=["all"])
    ap.add_argument("--n_cells", type=int, default=20)
    ap.add_argument("--n_target_cells", type=int, default=200)
    ap.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    ap.add_argument("--k_adv", type=int, default=4)
    ap.add_argument("--budget", type=int, default=2580,
                    help="budget in forward-equivalents. Equivalent to GCG 20 steps x 128 candidates")
    ap.add_argument("--n_cand", type=int, default=128)
    ap.add_argument("--top_k_grad", type=int, default=256)
    ap.add_argument("--n_restarts", type=int, default=3,
                    help="number of repetitions of the stochastic methods (mean and best are reported)")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--methods", nargs="+", default=None,
                    choices=["random_best", "hill_random", "hill_gradient",
                             "hill_random_faithful", "gcg_faithful"],
                    help="restrict the stochastic methods to run (default: all). "
                         "For comparisons with a larger budget, hill_gradient alone suffices.")
    ap.add_argument("--only_greedy", action="store_true",
                    help="skip the stochastic methods and run only the greedy ones. "
                         "Use when adding only the greedy control to existing results.")
    ap.add_argument("--greedy_pool", type=int, default=0,
                    help=">0 also runs greedy with the candidate pool narrowed to this number")
    ap.add_argument("--pool", type=int, default=0,
                    help=">0 runs an exhaustive search over the top pool by gradient and measures the optimality gap")
    ap.add_argument("--exhaustive_full", action="store_true",
                    help="exhaustive search over all TFs to obtain the true optimum. "
                         "Use only at k=2, where C(n,k) is practical.")
    ap.add_argument("--exh_cells", type=int, default=2,
                    help="number of cells for the exhaustive search (from the first)")
    ap.add_argument("--exh_unordered", action="store_true",
                    help="enumerate only combinations, not permutations, in the exhaustive search (gives a lower bound)")
    ap.add_argument("--exh_batch_size", type=int, default=256,
                    help="batch size for the exhaustive search (can be large since no gradient is needed)")
    ap.add_argument("--max_perms", type=int, default=64,
                    help="upper limit on the number of permutations evaluated for order sensitivity. If k! exceeds it, "
                         "sample at random (the spread obtained is a lower bound)")
    ap.add_argument("--order_sets", type=int, default=20,
                    help="number of sets for measuring order sensitivity (0 disables). "
                         "Reorder the same set in k! ways and measure the spread of cos_sim")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--emb_layer", type=int, default=0, choices=[0, -1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_dir", default="./search_comparison")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_overexpress_mode("pop_and_insert")
    set_emb_layer(args.emb_layer)

    model, _ = load_geneformer_model(args.model_dir, device)
    td, nd, _, _ = load_dicts(args.dict_dir)
    tf_token_ids = load_tf_list(args.tf_list, td, nd)
    tf_names = [n for n, _ in tf_token_ids]
    tf_ids = torch.tensor([t for _, t in tf_token_ids],
                          dtype=torch.long, device=device)
    V = model.embeddings.word_embeddings.weight.shape[0]
    ban = torch.ones(V, dtype=torch.bool, device=device)
    ban[tf_ids] = False                     # forbid non-TFs
    print(f"TFs: {len(tf_names)}  budget: {args.budget} forward  k={args.k_adv}")

    pw_config, paths = load_pathway_config(args.pathway_config,
                                          model="geneformer")
    pw_ids = get_pathway_ids(
        pw_config, None if args.pathways == ["all"] else args.pathways)

    # 2x2 of candidate source (uniform/gradient) x acceptance rule (improving only/unconditional) + controls
    METHODS = args.methods or ["random_best", "hill_random", "hill_gradient",
                               "hill_random_faithful", "gcg_faithful"]
    outfile = os.path.join(args.output_dir, "search_comparison.json")
    all_res = {}

    def save():
        tmp = outfile + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"config": vars(args), "results": all_res},
                      f, indent=2, ensure_ascii=False)
        os.replace(tmp, outfile)

    for pw_name in pw_ids:
        pw = pw_config[pw_name]
        print(f"\n{'='*72}\nPathway: {pw_name}\n{'='*72}")
        ds = load_from_disk(paths[pw_name])
        cells = prepare_cell_data(ds, pw["source_type"],
                                  n_cells=args.n_cells, seed=args.seed)
        centroid = compute_target_centroid(
            model, ds, pw["target_type"], n_cells=args.n_target_cells,
            device=device, max_seq_len=args.max_seq_len)

        rec = {m: {"delta": [], "cost": []} for m in METHODS}
        rec["greedy"] = {"delta": [], "cost": []}
        rec["exhaustive"] = {"delta": [], "cost": []}
        rec["exhaustive_full"] = {"delta": [], "cost": []}
        exh_full_sets = []

        for ci, cell in enumerate(cells):
            orig = torch.tensor(cell["input_ids"][:args.max_seq_len],
                                dtype=torch.long, device=device)
            b = base_cos(model, orig, centroid, args.max_seq_len)

            for m in ([] if args.only_greedy else METHODS):
                ds_, cs_, inits, curves, sets_ = [], [], [], [], []
                for r in range(args.n_restarts):
                    rng = np.random.default_rng(args.seed + ci * 100 + r)
                    if m == "random_best":
                        d, c, info = run_random_best(
                            model, orig, centroid, tf_ids, args.k_adv,
                            args.budget, args.max_seq_len, args.batch_size,
                            rng, b)
                    else:
                        d, c, info = run_hill(
                            model, orig, centroid, tf_ids, args.k_adv,
                            args.budget, args.n_cand, args.max_seq_len,
                            args.batch_size, rng, b,
                            mode=("random" if m.startswith("hill_random")
                                  else "gradient"),
                            top_k=args.top_k_grad, ban_mask=ban,
                            monotone=not m.endswith("_faithful"))
                    ds_.append(d); cs_.append(c)
                    inits.append(info["init"]); curves.append(info["curve"])
                    sets_.append([tf_names[i] for i in (info.get("set") or [])])
                rec[m]["delta"].append(st.mean(ds_))
                rec[m]["cost"].append(st.mean(cs_))
                # per-restart values (best-of-restarts can be computed later)
                rec[m].setdefault("delta_restarts", []).append(ds_)
                # same nesting as delta_restarts. (Delta, set) pairs can be made later.
                rec[m].setdefault("sets_restarts", []).append(sets_)
                rec[m].setdefault("init", []).append(st.mean(inits))
                # the anytime curve is interpolated onto the budget grid and averaged
                rec[m].setdefault("curve", []).append(
                    _interp_curve(curves, args.budget))

            if args.greedy:
                d, c, ginfo = run_greedy(model, orig, centroid, tf_ids,
                                         args.k_adv, args.max_seq_len,
                                         args.batch_size, b)
                rec["greedy"]["delta"].append(d)
                rec["greedy"]["cost"].append(c)
                rec["greedy"].setdefault("sets_per_cell", []).append(
                    [tf_names[i] for i in ginfo["set"]])

            if args.greedy_pool > 0:
                nt = tf_ids.numel()
                rp = np.random.default_rng(args.seed + ci).choice(
                    nt, size=min(args.greedy_pool, nt), replace=False)
                d, c, ginfo = run_greedy_pooled(
                    model, orig, centroid, tf_ids, rp.tolist(), args.k_adv,
                    args.max_seq_len, args.batch_size, b)
                rec.setdefault("greedy_random_pool",
                               {"delta": [], "cost": []})
                rec["greedy_random_pool"]["delta"].append(d)
                rec["greedy_random_pool"]["cost"].append(c)
                rec["greedy_random_pool"].setdefault(
                    "sets_per_cell", []).append(
                        [tf_names[i] for i in ginfo["set"]])

                init = torch.tensor(rp[:args.k_adv].tolist(),
                                    dtype=torch.long, device=device)
                tk = grad_topk(model, orig, tf_ids[init], centroid, tf_ids,
                               args.max_seq_len, args.greedy_pool, ban)
                if tk is not None:
                    v2i = {int(t): i for i, t in enumerate(tf_ids.tolist())}
                    gp = sorted({v2i[int(t)] for t in tk[0].tolist()
                                 if int(t) in v2i})
                    d, c, ginfo = run_greedy_pooled(
                        model, orig, centroid, tf_ids, gp, args.k_adv,
                        args.max_seq_len, args.batch_size, b)
                    rec.setdefault("greedy_grad_pool",
                                   {"delta": [], "cost": []})
                    rec["greedy_grad_pool"]["delta"].append(d)
                    rec["greedy_grad_pool"]["cost"].append(c + 2)
                    rec["greedy_grad_pool"].setdefault(
                        "sets_per_cell", []).append(
                            [tf_names[i] for i in ginfo["set"]])

            if (args.order_sets > 0 and ci < args.exh_cells
                    and not args.only_greedy):
                sp = order_sensitivity(
                    model, orig, centroid, tf_ids, args.k_adv,
                    args.max_seq_len, args.batch_size, args.order_sets,
                    np.random.default_rng(args.seed + ci),
                    max_perms=args.max_perms)
                rec.setdefault("order_spread", {"delta": [], "cost": []})
                rec["order_spread"]["delta"].append(st.mean(sp))
                rec["order_spread"]["cost"].append(0)

            if args.exhaustive_full and ci < args.exh_cells:
                import math
                n_ = tf_ids.numel()
                ncomb = (math.perm(n_, args.k_adv) if not args.exh_unordered
                         else math.comb(n_, args.k_adv))
                kind = "permutations" if not args.exh_unordered else "combinations"
                print(f"  cell {ci}: exhaustive search ({kind}) "
                      f"{ncomb:,} candidates ...", flush=True)
                dfull, cfull, bset = run_exhaustive_full(
                    model, orig, centroid, tf_ids, args.k_adv,
                    args.max_seq_len, args.exh_batch_size, b,
                    ordered=not args.exh_unordered)
                rec["exhaustive_full"]["delta"].append(dfull)
                rec["exhaustive_full"]["cost"].append(cfull)
                exh_full_sets.append([tf_names[i] for i in (bset or [])])
                print(f"    optimum Delta={dfull:+.6f}  set={exh_full_sets[-1]}",
                      flush=True)

            if args.pool > 0:
                init = torch.tensor(
                    np.random.default_rng(args.seed).choice(
                        len(tf_names), size=args.k_adv, replace=False),
                    dtype=torch.long, device=device)
                tk = grad_topk(model, orig, tf_ids[init], centroid, tf_ids,
                               args.max_seq_len, args.pool, ban)
                if tk is not None:
                    v2i = {int(t): i for i, t in enumerate(tf_ids.tolist())}
                    pool = sorted({v2i[int(t)] for t in tk[0].tolist()
                                   if int(t) in v2i})[:args.pool]
                    if len(pool) >= args.k_adv:
                        d, c = run_exhaustive(
                            model, orig, centroid, tf_ids, pool, args.k_adv,
                            args.max_seq_len, args.batch_size, b)
                        rec["exhaustive"]["delta"].append(d)
                        rec["exhaustive"]["cost"].append(c)

        print(f"\n  {'method':16s} {'Delta(S)':>11s} {'sd':>10s} {'cost':>10s}")
        print("  " + "-" * 52)
        for m in METHODS + ["greedy", "greedy_random_pool", "greedy_grad_pool",
                            "exhaustive", "exhaustive_full"]:
            # rec lacks some keys depending on the conditions (e.g. --greedy_pool not given)
            if m not in rec or not rec[m]["delta"]:
                continue
            print(f"  {m:16s} {st.mean(rec[m]['delta']):+11.6f} "
                  f"{st.pstdev(rec[m]['delta']):10.6f} "
                  f"{st.mean(rec[m]['cost']):10.0f}")
        all_res[pw_name] = {m: {"delta_mean": st.mean(v["delta"]),
                                "delta_sd": st.pstdev(v["delta"]),
                                "cost_mean": st.mean(v["cost"]),
                                "n_cells": len(v["delta"]),
                                # keep per-cell values for the paired tests
                                "delta_per_cell": [float(x)
                                                   for x in v["delta"]],
                                "delta_restarts": v.get("delta_restarts"),
                                "init_mean": (st.mean(v["init"])
                                              if v.get("init") else None),
                                "curve": v.get("curve"),
                                # search methods: gene symbols of [cell][restart][k]
                                "sets_restarts": v.get("sets_restarts"),
                                # greedy: [cell][k] (deterministic, so no restarts)
                                "sets_per_cell": v.get("sets_per_cell")}
                            for m, v in rec.items() if v["delta"]}
        if exh_full_sets:
            all_res[pw_name]["exhaustive_full_sets"] = exh_full_sets
        save()
        print(f"  [saved] {outfile}")
        del ds

    # ---- Verdict over all pathways ----
    print(f"\n{'='*72}\n  Verdict\n{'='*72}")
    print(f"  {'method':16s} {'Delta(S)':>11s} {'cost':>10s} {'vs base':>13s}")
    # The reference is hill_random. Fall back to greedy if it is absent, e.g. with --only_greedy.
    ref, base_m = None, None
    for cand in ("hill_random", "greedy", "greedy_random_pool"):
        v = [r[cand]["delta_mean"] for r in all_res.values() if cand in r]
        if v:
            ref, base_m = st.mean(v), cand
            break
    print(f"  (reference of the ratio: {base_m})")
    for m in METHODS + ["greedy", "greedy_random_pool", "greedy_grad_pool",
                        "exhaustive", "exhaustive_full"]:
        vals = [r[m]["delta_mean"] for r in all_res.values() if m in r]
        cost = [r[m]["cost_mean"] for r in all_res.values() if m in r]
        if not vals:
            continue
        rat = (st.mean(vals) / ref) if ref else float("nan")
        print(f"  {m:16s} {st.mean(vals):+11.6f} {st.mean(cost):10.0f} "
              f"{rat:13.2f}")

    # ---- Paired tests (cells of all pathways pooled) ----
    def pooled(m):
        out = []
        for pw_ in sorted(all_res):
            v = all_res[pw_].get(m)
            if isinstance(v, dict) and v.get("delta_per_cell"):
                out.append((pw_, v["delta_per_cell"]))
        return out

    print(f"\n{'='*72}")
    print("  Paired tests (differences on the same cells, all pathways pooled)")
    print(f"{'='*72}")
    PAIRS = [
        ("hill_gradient", "hill_random", "contribution of gradient"),
        ("hill_gradient", "random_best", "search vs random search"),
        ("hill_random", "random_best", "contribution of local search"),
        ("hill_gradient", "greedy", "GCG vs greedy"),
        ("gcg_faithful", "hill_gradient", "effect of unconditional acceptance (gradient candidates)"),
        ("hill_random_faithful", "hill_random", "effect of unconditional acceptance (uniform candidates)"),
        ("gcg_faithful", "hill_random_faithful", "contribution of gradient (unconditional acceptance)"),
        ("hill_gradient", "greedy_random_pool",
         "GCG vs greedy (same budget, random pool)"),
        ("hill_gradient", "greedy_grad_pool",
         "GCG vs greedy (same budget, gradient pool)"),
        ("greedy_grad_pool", "greedy_random_pool", "effect of using the gradient for the pool"),
    ]
    for a, b_, label in PAIRS:
        pa, pb = dict(pooled(a)), dict(pooled(b_))
        common = [k for k in pa if k in pb and len(pa[k]) == len(pb[k])]
        if not common:
            continue
        A = [x for k in common for x in pa[k]]
        B = [x for k in common for x in pb[k]]
        r = paired_test(A, B)
        if not r:
            continue
        ca = [v[a]["cost_mean"] for v in all_res.values() if a in v]
        cb = [v[b_]["cost_mean"] for v in all_res.values() if b_ in v]
        cr = (f"  cost {st.mean(cb)/st.mean(ca):.2f}x"
              if ca and cb and st.mean(ca) else "")
        line = (f"  {label:24s} diff {r['mean_diff']:+.6f} "
                f"(SE {r['se']:.6f})  t={r['t']:+.2f}  "
                f"wins {r['wins']}/{r['n']}{cr}")
        if "p_wilcoxon" in r:
            line += f"  p={r['p_wilcoxon']:.2e}"
        elif "p_t" in r:
            line += f"  p={r['p_t']:.2e}"
        print(line)

    print(f"\n  Paired tests per pathway (contribution of gradient)")
    pa, pb = dict(pooled("hill_gradient")), dict(pooled("hill_random"))
    for k in sorted(set(pa) & set(pb)):
        r = paired_test(pa[k], pb[k])
        if r:
            pv = r.get("p_wilcoxon", r.get("p_t", float("nan")))
            print(f"    {k}: diff {r['mean_diff']:+.6f}  t={r['t']:+.2f}  "
                  f"wins {r['wins']}/{r['n']}  p={pv:.3f}")

    # ---- best-of-restarts and anytime curve ----
    print(f"\n{'='*72}")
    print("  Restart strategy (mean vs best)")
    print(f"{'='*72}")
    print(f"  {'method':16s} {'mean':>11s} {'best-of-R':>11s} {'initial':>11s}")
    for m in METHODS:
        mn, bt, it = [], [], []
        for r in all_res.values():
            v = r.get(m) or {}
            if v.get("delta_restarts"):
                for rr in v["delta_restarts"]:
                    mn.append(st.mean(rr)); bt.append(max(rr))
            if v.get("init_mean") is not None:
                it.append(v["init_mean"])
        if mn:
            print(f"  {m:16s} {st.mean(mn):+11.6f} {st.mean(bt):+11.6f} "
                  + (f"{st.mean(it):+11.6f}" if it else f"{'—':>11s}"))
    print("  Note: in practice use best-of-R. Greedy is deterministic, so mean = best.")

    print(f"\n{'='*72}")
    print("  Anytime performance (Delta per budget, mean over all pathways and cells)")
    print(f"{'='*72}")
    grids = sorted({g for r in all_res.values() for m in METHODS
                    for cv in (r.get(m, {}).get("curve") or []) for g, _ in cv})
    if grids:
        print(f"  {'method':16s}" + "".join(f"{g:>10d}" for g in grids))
        for m in METHODS:
            row = []
            for g in grids:
                vals = [d for r in all_res.values()
                        for cv in (r.get(m, {}).get("curve") or [])
                        for gg, d in cv if gg == g]
                row.append(st.mean(vals) if vals else float("nan"))
            print(f"  {m:16s}" + "".join(f"{x:10.5f}" for x in row))
        print("  Note: compare horizontally at the same budget. If GCG is higher over the whole budget range, it dominates.")

    # ---- Order sensitivity ----
    osp = [r["order_spread"]["delta_mean"] for r in all_res.values()
           if "order_spread" in r]
    if osp:
        _hg = [r["hill_gradient"]["delta_mean"]
               for r in all_res.values() if "hill_gradient" in r]
        hgm = st.mean(_hg) if _hg else float("nan")
        print(f"\n{'='*72}")
        print("  Sensitivity to insertion order")
        print(f"{'='*72}")
        print(f"  Spread of cos_sim (mean) when the same set is reordered in {args.k_adv}! ways: "
              f"{st.mean(osp):+.6f}")
        print(f"  Improvement obtained by the search {hgm:+.6f}: "
              f"{st.mean(osp)/hgm*100 if hgm else float('nan'):.0f}%")
        if st.mean(osp) > 0.2 * hgm:
            print("  -> The effect of order is not negligible. The objective is defined over")
            print("     permutations, not sets, and the search space is C(n,k)*k!.")

    # ---- Optimality gap ----
    ef = pooled("exhaustive_full")
    if ef:
        print(f"\n{'='*72}")
        print("  Optimality gap (reach rate with the exhaustive search as 100%)")
        print(f"{'='*72}")
        for m in ["hill_gradient", "gcg_faithful", "hill_random",
                  "random_best", "greedy"]:
            pm = dict(pooled(m))
            rows = []
            for pw_, ex in ef:
                if pw_ in pm:
                    n = min(len(ex), len(pm[pw_]))
                    for i in range(n):
                        if ex[i] > 0:
                            rows.append(pm[pw_][i] / ex[i])
            if rows:
                print(f"  {m:16s} reach rate {st.mean(rows):6.1%}  "
                      f"(n={len(rows)} cells)")

    hg = [r["hill_gradient"]["delta_mean"] for r in all_res.values()
          if "hill_gradient" in r]
    hr = [r["hill_random"]["delta_mean"] for r in all_res.values()
          if "hill_random" in r]
    if hg and hr:
        ratio = st.mean(hg) / st.mean(hr)
        print()
        if ratio < 1.05:
            print("  -> The gradient contributes almost nothing. The essence is local search,")
            print("     and the claim 'by GCG' does not hold.")
        else:
            print(f"  -> The gradient contributes (ratio {ratio:.2f}).")
            print("     The claim for GCG holds.")

    print(f"\nSaved: {outfile}")


if __name__ == "__main__":
    main()
