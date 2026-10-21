#!/usr/bin/env python3
"""Regenerate the manuscript figures.

    python figures.py             # all figures
    python figures.py 3a 5       # only figures 3a and 5

Outputs PDF (vector) next to this file. The two panels of fig1 are drawn in
LaTeX (fig1.tex). figS1.pdf (order sensitivity) is a supplementary figure.

Every number comes from figure_data.py. fig2 additionally reads
additivity_summary.json, which is produced by summarize_additivity.py.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figure_data as D

# --------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 200,
})

DARK, MID, LIGHT = "#26215C", "#7F77DD", "#B4B2A9"
SCALE_DELTA = 1e3   # improvements are shown in units of 1e-3
HERE = os.path.dirname(os.path.abspath(__file__))


def out(name):
    return os.path.join(HERE, name)


def panel(ax, letter, x=-0.30, y=1.14):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", clip_on=False)


def save_tight(fig, name):
    """Save with a tight bounding box (panel letters, axis labels and
    out-of-axes legends sit outside the axes and would otherwise be cut)."""
    fig.savefig(out(name), bbox_inches="tight", pad_inches=0.03)


# --------------------------------------------------------------- fig S1
def figS1():
    """Order sensitivity: supplementary figure (file figS1.pdf)."""
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    for model, c, m in [("Geneformer", DARK, "o"), ("scGPT", MID, "s")]:
        ax.plot(D.KS, D.ORDER_SENSITIVITY[model], marker=m, color=c, lw=1.0,
                ms=4, label=model)
    ax.annotate(r"$<10^{-7}$ absolute", xy=(4, 0), xytext=(2.7, 1.8),
                fontsize=7,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="black"))
    ax.set_xscale("log", base=2)
    ax.set_xticks(D.KS); ax.set_xticklabels([str(k) for k in D.KS])
    ax.set_xlabel("set size $k$", fontsize=7.5)
    ax.set_ylabel(r"order sensitivity, % of $\Delta(S)$", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(out("figS1.pdf")); plt.close(fig)


# ---------------------------------------------------------------- fig 2
def fig2():
    """Per-pathway regression slopes and correlations against additivity."""
    path = os.path.join(HERE, D.ADDITIVITY_JSON)
    if not os.path.exists(path):
        print(f"[skip] fig2 needs {D.ADDITIVITY_JSON} "
              f"(produced by summarize_additivity.py)")
        return
    res = json.load(open(path))["results"]
    pws = sorted(res)

    def get(p, k, field):
        return [s for s in res[p]["summary"]
                if s["k"] == k and s["kind"] == "random"][0][field]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    ypos = np.arange(len(pws))[::-1]
    styles = [(2, DARK, "o"), (4, MID, "s"), (8, LIGHT, "^")]
    for ax, field, xlabel, refs in [
            (axes[0], "slope", "regression slope",
             [(1.0, "additive"), (0.9, None)]),
            (axes[1], "pearson", "Pearson $r$", [(1.0, "additive")])]:
        for k, c, m in styles:
            ax.plot([get(p, k, field) for p in pws], ypos, m, color=c, ms=4,
                    ls="none", label=f"$k$={k}")
        for x, lab in refs:
            ax.axvline(x, color="black", lw=0.7, ls="--" if lab else ":")
            if lab:
                ax.text(x - 0.02, len(pws) - 0.2, lab + " ", fontsize=7,
                        va="bottom", ha="right")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlim(-0.45, 1.12)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", lw=0.3, color="black", alpha=0.12)
    # mark the pathways that the search comparison uses
    for ax in axes:
        for y, p in zip(ypos, pws):
            if p in D.SEARCH_PATHWAYS:
                ax.axhspan(y - 0.42, y + 0.42, color=DARK, alpha=0.07, lw=0)
    axes[0].set_yticks(ypos); axes[0].set_yticklabels(pws, fontsize=7)
    for lab, p in zip(axes[0].get_yticklabels(), pws):
        if p in D.SEARCH_PATHWAYS:
            lab.set_fontweight("bold")
    axes[0].set_ylim(-0.8, len(pws) + 0.6)
    axes[0].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0.02, 0.02))
    panel(axes[0], "a", -0.13, 1.02); panel(axes[1], "b", -0.06, 1.02)
    fig.tight_layout(pad=0.4)
    save_tight(fig, "fig2.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig 3a
def fig3a():
    """Matched-budget comparison and its scaling with set size."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for ax, model in zip(axes[:2], D.MODELS):
        x = np.arange(3)
        att = SCALE_DELTA * np.array(D.ATTACK[model])
        grp = SCALE_DELTA * np.array(D.GREEDY_POOL[model])
        ax.bar(x - 0.19, att, 0.36, color=DARK, label="attack")
        ax.bar(x + 0.19, grp, 0.36, color=LIGHT,
               label="greedy, gradient pool")
        for i, s in enumerate(D.SIG[model]):
            ax.text(i, max(att[i], grp[i]) * 1.06,
                    s, ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels([f"$k$={k}" for k in D.KS])
        ax.set_ylabel(r"$\Delta(S)$ ($\times10^{-3}$)"
                      if model == "Geneformer" else "")
        ax.set_title(model, fontsize=8)
        # headroom: the two-line legend sits above the k = 8 bars in the
        # Geneformer panel
        ax.set_ylim(0, att.max() * (1.6 if model == "Geneformer"
                                    else 1.25))
    axes[0].legend(loc="upper left", fontsize=7)
    panel(axes[0], "a", -0.30, 1.16)

    ax = axes[2]
    for model, c, m in [("Geneformer", DARK, "o"), ("scGPT", MID, "s")]:
        ax.plot(D.KS, SCALE_DELTA * np.array(D.MARGIN[model]), marker=m,
                color=c, lw=1.0, ms=4, label=model)
    ax.set_xscale("log", base=2); ax.set_xticks(D.KS)
    ax.set_xticklabels([str(k) for k in D.KS])
    ax.set_yscale("log")
    ax.set_xlabel("set size $k$")
    ax.set_ylabel(r"attack $-$ greedy ($\times10^{-3}$)")
    ax.legend(fontsize=7, loc="upper left")
    panel(ax, "b", -0.32)
    fig.tight_layout(pad=0.4)
    save_tight(fig, "fig3a.pdf"); plt.close(fig)


def fig3b():
    """Budget coverage: the attack relative to greedy, against budget."""
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    budgets = np.array(D.BUDGETS)
    for model, c, m in [("Geneformer", DARK, "o"), ("scGPT", MID, "s")]:
        y = np.array(D.CURVES[model][("gradient", "unconditional")]) \
            / D.GREEDY_FULL[model]
        b = np.append(budgets, D.GREEDY_COST[model])
        y = np.append(y, D.ATTACK_FINAL[model] / D.GREEDY_FULL[model])
        ax.plot(b, y, marker=m, color=c, lw=1.0, ms=3, label=model)
        ax.plot(D.GREEDY_COST[model], 1.0, marker="*", color=c, ms=9, ls="none")
    ax.axhline(1.0, color="black", lw=0.6, ls=":")
    ax.axvspan(64, D.GREEDY_COST["Geneformer"], color="black", alpha=0.05, lw=0)
    ax.set_xscale("log", base=2)
    ax.set_xlim(64, 30000); ax.set_ylim(-0.1, 1.5)
    ax.set_xlabel("forward-pass budget", fontsize=7.5)
    ax.set_ylabel(r"$\Delta(S)$ / greedy $\Delta(S)$", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.text(2000, 0.06, "greedy cannot operate here", fontsize=6.5,
            ha="center", alpha=0.75)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(pad=0.3)
    fig.savefig(out("fig3b.pdf")); plt.close(fig)


# ---------------------------------------------------------------- fig 4a, 4b
def fig4a():
    """Interaction plot, both models on a shared percentage scale."""
    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.2), sharey=True)
    for ax, model in zip(axes, D.MODELS):
        for src, c, m in [("uniform", LIGHT, "s"), ("gradient", DARK, "o")]:
            y = 100 * np.array(D.CELLS[model][src]) / D.SCALE[model]
            ax.plot([0, 1], y, marker=m, color=c, lw=1.1, ms=4,
                    label=f"{src} candidates")
        ax.set_xlim(-0.35, 1.35); ax.set_xticks([0, 1])
        ax.set_xticklabels(["improve-only", "unconditional"], fontsize=7)
        ax.set_title(model, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(r"$\Delta(S)$, % of model mean", fontsize=7.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.3, rect=(0, 0, 1, 0.88))
    save_tight(fig, "fig4a.pdf"); plt.close(fig)


def fig4b():
    """Forest plot of the simple effects and the interaction (panel b).

    Panel (a) is fig4a; this function draws only the forest, so that the two
    are included side by side in the manuscript without duplication.
    """
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    labels = [c[0] for c in D.CONTRASTS["Geneformer"]]
    ypos = np.arange(len(labels))[::-1]
    for model, c, m, off in [("Geneformer", DARK, "o", 0.15),
                             ("scGPT", MID, "s", -0.15)]:
        for y, (_, mean, ci_lo, ci_hi) in zip(ypos, D.CONTRASTS[model]):
            pct = 100 * mean / D.SCALE[model]
            lo = 100 * ci_lo / D.SCALE[model]
            hi = 100 * ci_hi / D.SCALE[model]
            ax.plot([lo, hi], [y + off, y + off], color=c, lw=1.0)
            ax.plot(pct, y + off, marker=m, color=c, ms=4,
                    label=model if y == ypos[0] else None)
    ax.axvline(0, color="black", lw=0.6, ls=":")
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel(r"effect, % of mean $\Delta(S)$", fontsize=7.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlim(-2.6, 7.4)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=2)
    fig.tight_layout(pad=0.3)
    save_tight(fig, "fig4b.pdf"); plt.close(fig)


# ---------------------------------------------------------------- fig 5
def fig5():
    """Acceptance-rule difference against budget, and the gradient ratio."""
    budgets = np.array(D.BUDGETS)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3),
                             gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for ax, model in zip(axes[:2], D.MODELS):
        for src, c, m in [("uniform", LIGHT, "s"), ("gradient", DARK, "o")]:
            d = SCALE_DELTA * (np.array(D.CURVES[model][(src, "unconditional")])
                               - np.array(D.CURVES[model][(src, "improve-only")]))
            ax.plot(budgets, d, color=c, marker=m, lw=1.0, ms=3,
                    label=f"{src} candidates")
            ax.axvline(D.DIVERGE[(model, src)], color=c, lw=0.5,
                            ls=":" if src == "uniform" else "--", alpha=0.7)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("forward-pass budget", fontsize=7)
        ax.set_title(model, fontsize=8)
        ax.tick_params(labelsize=6.5)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        if model == "Geneformer":
            ax.set_ylabel(r"unconditional $-$ improve-only ($\times10^{-3}$)",
                          fontsize=7)
            ax.legend(fontsize=6.5, loc="upper left")
            panel(ax, "a", -0.34)

    ax = axes[2]
    for model, c, m in [("Geneformer", DARK, "o"), ("scGPT", MID, "s")]:
        g = np.array(D.CURVES[model][("gradient", "improve-only")])
        u = np.array(D.CURVES[model][("uniform", "improve-only")])
        ok = (u > 0) & (g > 0)
        ax.plot(budgets[ok], (g / u)[ok], marker=m, color=c, lw=1.0, ms=3,
                label=model)
    ax.axhline(1.0, color="black", lw=0.6, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("forward-pass budget", fontsize=7)
    ax.set_ylabel("gradient / uniform", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.legend(fontsize=6.5)
    panel(ax, "b", -0.34)
    fig.tight_layout(pad=0.4)
    fig.savefig(out("fig5.pdf")); plt.close(fig)


# ---------------------------------------------------------------- fig S2
def figS2():
    """Supplementary figure: attainment of the k = 2 optimum."""
    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    names = [o[0] for o in D.OPTIMALITY][::-1]
    vals = [o[1] for o in D.OPTIMALITY][::-1]
    cols = [MID if ("uniform" in n or "random" in n) else DARK for n in names]
    ax.barh(np.arange(len(names)), vals, 0.6, color=cols)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("% of per-cell optimum attained ($k=2$)", fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.tick_params(labelsize=7)
    for i, v in enumerate(vals):
        ax.text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(out("figS2.pdf")); plt.close(fig)


ALL = {"2": fig2, "3a": fig3a, "3b": fig3b, "4a": fig4a,
       "4b": fig4b, "5": fig5, "S1": figS1, "S2": figS2}

if __name__ == "__main__":
    want = sys.argv[1:] or list(ALL)
    unknown = [w for w in want if w not in ALL]
    if unknown:
        sys.exit(f"unknown figure(s): {unknown}. choose from {list(ALL)}")
    for w in want:
        ALL[w]()
        print(f"wrote fig{w}.pdf")
