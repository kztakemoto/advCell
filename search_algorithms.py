#!/usr/bin/env python3
"""
Model-independent search algorithms
==============================
Common implementation so that any foundation model (Geneformer, scGPT, ...) runs
the same algorithm. The model-specific parts are confined to two callbacks.

  eval_fn(sets)  -> (N,) tensor
      sets: (N, k) long tensor (TF indices, subscripts of tf_ids)
      Returns the objective value when the set is overexpressed. No gradient needed.

  grad_fn(current) -> (k, top_k) tensor or None
      current: (k,) long tensor
      Returns, per position, the top_k of the gradient projection at the current set
      (the return value is also subscripts of tf_ids). None means the gradient could not be obtained.

This way the search side does not change by a single line when the model changes,
so "properties of the algorithm" and "properties of the model" can be claimed separately.

Implemented methods:
  random_best   try B random sets and take the best
  hill_random   hill climbing by single-position replacement, uniform candidates (no gradient)
  hill_gradient same, candidates from the top of the gradient (monotone=True)
  gcg_faithful  same, accepting the batch best unconditionally (Zou et al. 2023)
  greedy        greedy forward selection
  exhaustive    exhaustive search (permutations or combinations)
"""

import itertools
import math
import statistics as st

import numpy as np
import torch


CURVE_GRID = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


# ------------------------------------------------------------
def run_random_best(eval_fn, n_tf, k, budget, rng, base, chunk=256):
    """Try B random k-tuples and return the best."""
    best = -1e30
    best_set = None
    done = 0
    curve = []
    init = None
    while done < budget:
        m = min(chunk, budget - done)
        idx = np.stack([rng.choice(n_tf, size=k, replace=False)
                        for _ in range(m)])
        vals = eval_fn(torch.tensor(idx, dtype=torch.long))
        if init is None:
            init = float(vals[0].item()) - base
        bi = int(torch.argmax(vals).item())
        bv = float(vals[bi].item())
        if bv > best:
            best = bv
            best_set = [int(x) for x in idx[bi]]
        done += m
        curve.append((done, best - base))
    return best - base, budget, {"init": init, "curve": curve,
                                 "set": best_set}


def run_hill(eval_fn, grad_fn, n_tf, k, budget, n_cand, rng, base,
             mode="gradient", monotone=True, grad_cost=2):
    """Hill climbing by single-position replacement.

    mode="random"   candidates uniform over all TFs
    mode="gradient" candidates from the top of grad_fn
    monotone=True   accept only on improvement
    monotone=False  accept the batch best unconditionally (following Zou et al.)
    grad_cost       cost of a backward pass in forward-equivalents
    """
    cur = torch.tensor(rng.choice(n_tf, size=k, replace=False),
                       dtype=torch.long)
    cur_val = float(eval_fn(cur.view(1, -1))[0].item())
    cost = 1
    init_val = cur_val
    best_val = cur_val
    best_set = cur.clone()
    curve = [(cost, best_val - base)]
    steps = max(1, budget // n_cand)

    for _ in range(steps):
        if mode == "gradient":
            tk = grad_fn(cur)
            cost += grad_cost
            if tk is None:
                break
            pos = rng.integers(0, k, size=n_cand)
            pick = rng.integers(0, tk.shape[1], size=n_cand)
            cand = cur.view(1, -1).repeat(n_cand, 1)
            cand[torch.arange(n_cand), torch.as_tensor(pos)] = \
                tk[torch.as_tensor(pos), torch.as_tensor(pick)].cpu()
        else:
            pos = rng.integers(0, k, size=n_cand)
            new = rng.choice(n_tf, size=n_cand)
            cand = cur.view(1, -1).repeat(n_cand, 1)
            cand[torch.arange(n_cand), torch.as_tensor(pos)] = \
                torch.as_tensor(new)

        keep = [i for i in range(cand.shape[0])
                if len(set(cand[i].tolist())) == k]
        if not keep:
            continue
        cand = cand[torch.tensor(keep)]
        vals = eval_fn(cand)
        cost += cand.shape[0]
        bi = int(torch.argmax(vals).item())
        bv = float(vals[bi].item())
        if (not monotone) or bv > cur_val:
            cur = cand[bi].clone()
            cur_val = bv
        # With monotone=False, cur can get worse, so take the set at the moment
        # best_val was updated. It need not coincide with cur at the end of the loop.
        if cur_val > best_val:
            best_val = cur_val
            best_set = cur.clone()
        curve.append((cost, best_val - base))
    return best_val - base, cost, {"init": init_val - base, "curve": curve,
                                   "set": [int(x) for x in best_set.tolist()]}


def run_greedy(eval_fn, n_tf, k, base):
    """Greedy forward selection. Cost is k * n_tf."""
    chosen, cost, last = [], 0, base
    for _ in range(k):
        remain = [i for i in range(n_tf) if i not in chosen]
        cand = torch.tensor([chosen + [i] for i in remain], dtype=torch.long)
        vals = eval_fn(cand)
        cost += cand.shape[0]
        bi = int(torch.argmax(vals).item())
        chosen.append(remain[bi])
        last = float(vals[bi].item())
    return last - base, cost, {"set": chosen}


def run_greedy_pooled(eval_fn, pool, k, base):
    """Greedy forward selection over a narrowed candidate pool. Cost is k * len(pool).

    A greedy method that scans all TFs costs k*n_tf, which is large. Narrowing the pool makes it
    cheaper, so it is a direct control against the objection "why not just use greedy at the same budget".
    How the pool is built is left to the caller:
      - random thinning            when only the search structure is made cheaper
      - top of gradient projection when gradient information is also given to the greedy method
    """
    chosen, cost, last = [], 0, base
    pool = list(pool)
    for _ in range(k):
        remain = [i for i in pool if i not in chosen]
        if not remain:
            break
        cand = torch.tensor([chosen + [i] for i in remain], dtype=torch.long)
        vals = eval_fn(cand)
        cost += cand.shape[0]
        bi = int(torch.argmax(vals).item())
        chosen.append(remain[bi])
        last = float(vals[bi].item())
    return last - base, cost, {"set": chosen}


def run_exhaustive_full(eval_fn, n_tf, k, base, chunk=200_000, ordered=True):
    """Complete exhaustive search. Gives the true optimum.

    With ordered=True, permutations are enumerated. This is correct for models in which the insertion
    order affects the objective (e.g. Geneformer, which has absolute position embeddings).
    For models where it does not, ordered=False reduces the count to 1/k!.
    """
    best, best_set, total = -1e30, None, 0
    it = (itertools.permutations(range(n_tf), k) if ordered
          else itertools.combinations(range(n_tf), k))
    while True:
        buf = list(itertools.islice(it, chunk))
        if not buf:
            break
        vals = eval_fn(torch.tensor(buf, dtype=torch.long))
        i = int(torch.argmax(vals).item())
        if float(vals[i].item()) > best:
            best = float(vals[i].item())
            best_set = list(buf[i])
        total += len(buf)
        del vals
    return best - base, total, {"set": best_set}


def order_sensitivity(eval_fn, n_tf, k, n_sets, rng, max_perms=64):
    """Spread of the objective value when the same set is reordered.

    For a position-independent model this should be 0, which in itself
    confirms that "the set can be treated as a set for that model".
    """
    full = math.factorial(k)
    spreads = []
    for _ in range(n_sets):
        base_idx = rng.choice(n_tf, size=k, replace=False).tolist()
        if full <= max_perms:
            perms = list(itertools.permutations(base_idx))
        else:
            seen, perms = set(), []
            while len(perms) < max_perms:
                q = tuple(rng.permutation(base_idx).tolist())
                if q not in seen:
                    seen.add(q)
                    perms.append(q)
        vals = eval_fn(torch.tensor(perms, dtype=torch.long))
        spreads.append(float((vals.max() - vals.min()).item()))
    return spreads


# ------------------------------------------------------------
# 2x2 design of candidate source x acceptance rule.
#   candidate source : random   = uniform over all TFs   / gradient = top of the gradient projection
#   acceptance rule  : monotone = accept only on improvement / faithful = accept the batch best unconditionally
# If only gcg_faithful is included and hill_random_faithful is missing,
# the "effect of unconditional acceptance" and the "effect of the gradient" are confounded and cannot be separated.
METHOD_SPEC = {
    "random_best":          None,                  # control without search structure
    "hill_random":          ("random",   True),
    "hill_gradient":        ("gradient", True),
    "hill_random_faithful": ("random",   False),
    "gcg_faithful":         ("gradient", False),
}
METHODS = ["random_best", "hill_random", "hill_gradient",
           "hill_random_faithful", "gcg_faithful"]


def run_all_methods(eval_fn, grad_fn, n_tf, k, budget, n_cand,
                    base, seed, cell_idx, n_restarts=3, methods=METHODS):
    """Run all methods at the same budget. Returns the aggregate per method."""
    out = {}
    for m in methods:
        ds, cs, inits, curves, sets = [], [], [], [], []
        for r in range(n_restarts):
            rng = np.random.default_rng(seed + cell_idx * 100 + r)
            spec = METHOD_SPEC.get(m)
            if spec is None:
                d, c, info = run_random_best(eval_fn, n_tf, k, budget,
                                             rng, base)
            else:
                mode, monotone = spec
                d, c, info = run_hill(
                    eval_fn, grad_fn, n_tf, k, budget, n_cand, rng, base,
                    mode=mode, monotone=monotone)
            ds.append(d); cs.append(c)
            inits.append(info["init"]); curves.append(info["curve"])
            sets.append(info.get("set"))
        # delta is the mean over restarts, so there is no single corresponding set.
        # set_restarts has the same order as ds, so (delta, set) pairs can be made.
        # set_best_restart is the set corresponding to the best-of-restarts aggregate.
        best_r = max(range(len(ds)), key=lambda i: ds[i])
        out[m] = {"delta": st.mean(ds), "cost": st.mean(cs),
                  "delta_restarts": ds, "init": st.mean(inits),
                  "curve": interp_curve(curves, budget),
                  "set_restarts": sets,
                  "set_best_restart": sets[best_r]}
    return out


def interp_curve(curves, budget):
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


# ------------------------------------------------------------
def paired_test(a, b):
    """Paired t-test of a-b (and Wilcoxon where possible)."""
    dif = [x - y for x, y in zip(a, b)]
    n = len(dif)
    if n < 3:
        return {}
    m = st.mean(dif)
    sd = st.stdev(dif)
    se = sd / (n ** 0.5) if sd > 0 else float("nan")
    t = m / se if se == se and se > 0 else float("nan")
    out = {"n": n, "mean_diff": m, "se": se, "t": t,
           "wins": sum(1 for x in dif if x > 0)}
    if t == t:
        out["p_t"] = _t_sf2(abs(t), n - 1)
    try:
        from scipy.stats import wilcoxon
        if any(x != 0 for x in dif):
            out["p_wilcoxon"] = float(wilcoxon(dif).pvalue)
    except Exception:
        pass
    return out


def _t_sf2(t, df):
    """Two-sided p value of Student's t with df degrees of freedom.

    The normal approximation erfc(|t|/sqrt(2)) understates the p value by
    orders of magnitude in the far tail (at df = 99 and |t| = 8 it is off by
    a factor of ~3000), which matters because these p values are reported.
    """
    try:
        from scipy.stats import t as _t
        return float(2.0 * _t.sf(t, df))
    except Exception:
        pass
    # Without scipy: evaluate the regularized incomplete beta I_x(df/2, 1/2) by a continued fraction
    x = df / (df + t * t)
    return float(_betainc(df / 2.0, 0.5, x))


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b) via the Lentz continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    if x >= (a + 1.0) / (a + b + 2.0):
        return 1.0 - _betainc(b, a, 1.0 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        mm = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (mm * (b - mm) * x) / ((a + 2.0 * mm - 1.0) * (a + 2.0 * mm))
        else:
            num = (-((a + mm) * (a + b + mm) * x)
                   / ((a + 2.0 * mm) * (a + 2.0 * mm + 1.0)))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-15:
            break
    return front * (f - 1.0)


def report_comparison(all_res, methods=METHODS, extra=("greedy",)):
    """Print the comparison between methods (common to all models)."""
    ms = list(methods) + [m for m in extra
                          if any(m in r for r in all_res.values())]
    print(f"\n  {'method':16s} {'Δ':>11s} {'cost':>10s}")
    print("  " + "-" * 40)
    for m in ms:
        v = [r[m]["delta_mean"] for r in all_res.values() if m in r]
        c = [r[m]["cost_mean"] for r in all_res.values() if m in r]
        if v:
            print(f"  {m:16s} {st.mean(v):+11.6f} {st.mean(c):10.0f}")

    PAIRS = [("hill_gradient", "hill_random", "contribution of gradient (improving acceptance)"),
             ("gcg_faithful", "hill_random_faithful", "contribution of gradient (unconditional acceptance)"),
             ("gcg_faithful", "hill_gradient", "effect of unconditional acceptance (gradient candidates)"),
             ("hill_random_faithful", "hill_random", "effect of unconditional acceptance (uniform candidates)"),
             ("hill_gradient", "random_best", "search vs random"),
             ("hill_random", "random_best", "contribution of local search"),
             ("gcg_faithful", "greedy", "GCG vs greedy")]
    print(f"\n  Paired tests (cells of all pathways pooled)")
    for a, b, lab in PAIRS:
        A = [x for r in all_res.values() if a in r and b in r
             for x in r[a]["delta_per_cell"]]
        B = [x for r in all_res.values() if a in r and b in r
             for x in r[b]["delta_per_cell"]]
        if len(A) != len(B) or not A:
            continue
        s = paired_test(A, B)
        if not s:
            continue
        p = s.get("p_wilcoxon", s.get("p_t", float("nan")))
        print(f"    {lab:20s} diff {s['mean_diff']:+.6f} t={s['t']:+6.2f} "
              f"wins {s['wins']:3d}/{s['n']} p={p:.2e}")
