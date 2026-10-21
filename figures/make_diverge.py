#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Derive DIVERGE (the first budget at which the curves differ) from the raw anytime values.

CURVES in figure_data.py is rounded to 5 digits; reading it by eye loses
differences of a few e-6 and puts the divergence point two steps off, so DIVERGE
is derived from the raw values.

Usage:
    python make_diverge.py interaction_2x2_gf_k8.json \\
                           interaction_2x2_scgpt_k8.json
"""
import json
import sys

PAIRS = [("uniform", "hill_random_faithful", "hill_random"),
         ("gradient", "gcg_faithful", "hill_gradient")]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = {}
    for path in sys.argv[1:]:
        d = json.load(open(path, encoding="utf-8"))
        a = d["anytime"]
        model = "scGPT" if "scgpt" in path.lower() else "Geneformer"
        budgets = sorted((int(b) for b in a[PAIRS[0][1]]), key=int)
        print(f"=== {model}  ({path}) ===")
        for src, uncond, improve in PAIRS:
            diffs = [a[uncond][str(b)] - a[improve][str(b)] for b in budgets]
            first = next((b for b, v in zip(budgets, diffs) if v != 0.0), None)
            out[(model, src)] = first
            print(f"  {src:9s} first nonzero: {first}")
            print("    " + "  ".join(f"{b}:{v:+.2e}"
                                     for b, v in zip(budgets, diffs)))
        print()
    print("DIVERGE = {" + ", ".join(
        f'("{m}", "{s}"): {v}' for (m, s), v in out.items()) + "}")


if __name__ == "__main__":
    main()
