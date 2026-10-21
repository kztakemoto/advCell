#!/usr/bin/env python3
"""
Gamma-weighted aggregation and geometric diagnostics
========================

## Gamma-weighted aggregation

When the loss can be written as a function of cos_sim alone, L = f(c), the within-cell ranking of GPR and
the search trajectory of GCG do not depend on the choice of f. What f affects is only the weight of aggregation across cells

    w = |f'(c)|

(confirmed by measurement in this project: Spearman = 1.0 per (cell, init),
and the GCG trajectories are identical).

    cosine  : L = -c        -> w = 1
    angular : L = arccos(c) -> w = (1 - c^2)^(-1/2)

We therefore generalize to

    w(γ) = (1 - c^2)^(-γ/2)

so that gamma=0 is the cosine loss and gamma=1 the angular loss. Taking cosine as the base,
all gammas can be reconstructed from a single run, so there is no need to rerun for each loss.

## Geometric diagnostics

Metrics for separating out why the choice of embedding layer changes the results.

    separation  : cos_sim between the source and target centroids (smaller is better separated)
    anisotropy  : mean pairwise cos_sim within one cell type
                  (the closer to 1, the more all vectors point the same way,
                    so cos_sim saturates and the directional information of the gradient is easily lost)
"""

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F


COS_CLAMP = 1.0 - 1e-6


def lam(cos_sim: float, gamma: float) -> float:
    """Weight w(gamma) = (1 - c^2)^(-gamma/2). 1 for gamma=0; the angular coefficient for gamma=1."""
    if gamma == 0.0:
        return 1.0
    c = min(max(float(cos_sim), -COS_CLAMP), COS_CLAMP)
    return float((1.0 - c * c) ** (-gamma / 2.0))


class GammaAccumulator:
    """Accumulate, per (cell, init), the cosine-base score and the cos_sim at that point,
    and reconstruct the weighted mean for any gamma.

    Usage:
        acc = GammaAccumulator(tf_names, gammas=[0.0, 0.5, 1.0, 2.0])
        for cell in cells:
            acc.start_cell()
            for init in inits:
                acc.add(scores_base, cos_sim)
            acc.end_cell()
        results = acc.finalize()   # {gamma: {"mean": {...}, "per_cell": [...]}}
    """

    def __init__(self, tf_names: Sequence[str], gammas: Sequence[float]):
        self.tf_names = list(tf_names)
        self.gammas = [float(g) for g in gammas]
        self._cell_rows: List[np.ndarray] = []   # (n_init, n_tf) of the current cell
        self._cell_cos: List[float] = []
        self.cell_rows: List[np.ndarray] = []
        self.cell_cos: List[List[float]] = []

    def start_cell(self):
        self._cell_rows, self._cell_cos = [], []

    def add(self, scores: Dict[str, float], cos_sim: float):
        self._cell_rows.append(
            np.array([scores[n] for n in self.tf_names], dtype=np.float64)
        )
        self._cell_cos.append(float(cos_sim))

    def end_cell(self):
        if not self._cell_rows:
            return
        self.cell_rows.append(np.stack(self._cell_rows))   # (n_init, n_tf)
        self.cell_cos.append(list(self._cell_cos))
        self._cell_rows, self._cell_cos = [], []

    def finalize(self) -> Dict[str, dict]:
        out = {}
        for g in self.gammas:
            per_cell = []
            for rows, coss in zip(self.cell_rows, self.cell_cos):
                w = np.array([lam(c, g) for c in coss], dtype=np.float64)
                # mean over n_init (weighted)
                per_cell.append((rows * w[:, None]).sum(0) / len(w))
            if not per_cell:
                continue
            mean_vec = np.mean(np.stack(per_cell), axis=0)
            out[f"{g:g}"] = {
                "mean": {n: float(v) for n, v in zip(self.tf_names, mean_vec)},
                "per_cell": [
                    {n: float(v) for n, v in zip(self.tf_names, pc)}
                    for pc in per_cell
                ],
            }
        return out

    def cos_sim_stats(self) -> dict:
        flat = [c for cs in self.cell_cos for c in cs]
        if not flat:
            return {}
        a = np.array(flat)
        lams = np.array([lam(c, 1.0) for c in flat])
        return {
            "cos_sim_mean": float(a.mean()),
            "cos_sim_min": float(a.min()),
            "cos_sim_max": float(a.max()),
            "lambda_min": float(lams.min()),
            "lambda_max": float(lams.max()),
            "lambda_ratio": float(lams.max() / lams.min()) if lams.min() > 0
                            else None,
        }


# ------------------------------------------------------------
# Geometric diagnostics
# ------------------------------------------------------------
def _pairwise_mean_cos(X: torch.Tensor) -> float:
    """Mean pairwise cos_sim between row vectors (diagonal excluded)."""
    Xn = F.normalize(X, dim=1)
    S = Xn @ Xn.T
    n = S.shape[0]
    if n < 2:
        return float("nan")
    off = (S.sum() - S.diagonal().sum()) / (n * (n - 1))
    return float(off.item())


@torch.no_grad()
def geometry_diagnostics(embed_fn, source_cells, target_cells) -> dict:
    """Measure the geometry of the embedding.

    Args:
        embed_fn:      function returning a (D,) tensor from a cell dict
        source_cells:  list of cell dict
        target_cells:  list of cell dict

    Returns:
        dict of scalars
    """
    S = torch.stack([embed_fn(c) for c in source_cells])
    T = torch.stack([embed_fn(c) for c in target_cells])

    cs = F.normalize(S.mean(0), dim=0)
    ct = F.normalize(T.mean(0), dim=0)

    sep = float(F.cosine_similarity(cs.unsqueeze(0), ct.unsqueeze(0)).item())
    src_to_tgt = F.cosine_similarity(S, ct.unsqueeze(0), dim=1)

    return {
        "n_source": int(S.shape[0]),
        "n_target": int(T.shape[0]),
        # cos_sim between the centroids. The closer to 1, the less the two classes are separated.
        "centroid_separation_cos": sep,
        # mean pairwise cos_sim within one cell type. The closer to 1, the more anisotropic.
        "anisotropy_source": _pairwise_mean_cos(S),
        "anisotropy_target": _pairwise_mean_cos(T),
        # cos_sim from each source cell to the target centroid (the starting point of optimization)
        "src_to_tgt_cos_mean": float(src_to_tgt.mean().item()),
        "src_to_tgt_cos_std": float(src_to_tgt.std().item()),
        # spread of the embedding norms (effect of taking the centroid as an unnormalized mean)
        "emb_norm_source_mean": float(S.norm(dim=1).mean().item()),
        "emb_norm_source_cv": float(
            (S.norm(dim=1).std() / S.norm(dim=1).mean()).item()
        ),
    }


def format_geometry(diag_by_layer: Dict[str, dict]) -> str:
    """Summarize the geometric diagnostics of each layer in one line."""
    lines = []
    hdr = (f"    {'layer':>16s} {'sep':>8s} {'aniso_src':>10s} "
           f"{'aniso_tgt':>10s} {'src2tgt':>9s}")
    lines.append(hdr)
    for name, d in diag_by_layer.items():
        lines.append(
            f"    {name:>16s} {d['centroid_separation_cos']:8.4f} "
            f"{d['anisotropy_source']:10.4f} {d['anisotropy_target']:10.4f} "
            f"{d['src_to_tgt_cos_mean']:9.4f}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------
# Norm bias and pathway specificity
# ------------------------------------------------------------
def normalize_by_norm(scores: Dict[str, float],
                      norms: Dict[str, float]) -> Dict[str, float]:
    """Divide the scores by ||E[j]|| to convert them to cosine projections.

    Since s_j = -g . E[j] = ||g|| ||E[j]|| cos(theta_j), dividing by
    ||E[j]|| gives ||g|| cos(theta_j), and the TF-specific norm factor disappears.
    ||E[j]|| does not depend on the cell, so dividing the aggregated score
    gives the same ranking as dividing per cell and then aggregating.
    """
    return {n: (v / norms[n] if norms.get(n) else v)
            for n, v in scores.items()}


def auc_vs_other_gt(scores: Dict[str, float],
                    own_gt: Sequence[str],
                    other_gt: Sequence[str]) -> float:
    """Metric of pathway specificity.

    The usual AUC is "GT of the own pathway" versus "all TFs", but then
    a high score can be obtained merely by ranking genes that look like generic master regulators
    highly (the norm-only baseline reaching 0.76 is evidence of that).

    Here the negatives are replaced with "GT TFs of other pathways".
    Both are master regulators of some pathway, so general importance
    cancels out and only pathway specificity remains.
    0.5 means the GT of the own pathway cannot be told from that of other pathways.
    """
    pos = [scores[g] for g in own_gt if g in scores]
    neg = [scores[g] for g in other_gt
           if g in scores and g not in set(own_gt)]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))
