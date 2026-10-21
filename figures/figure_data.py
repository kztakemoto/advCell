"""All measured values used by the manuscript figures, in one place.

Nothing here is computed: every number is a measured result copied from the
run outputs named in the comments. Keeping the data separate from the plotting
means a figure can be restyled without touching the numbers, and a number can
be corrected without reading matplotlib code.

Provenance
----------
budget-matched comparison   compare_greedy_pool.py summary
2x2 cells and contrasts     interaction_2x2_gf_k8.json,
                            interaction_2x2_scgpt_k8.json
anytime curves              the same two files, "anytime" block
order sensitivity           diag order-sensitivity runs
k=2 optimality              exhaustive k=2 enumeration
per-pathway additivity      additivity_summary.json (read at plot time)
"""

KS = [2, 4, 8]
MODELS = ["Geneformer", "scGPT"]

# ---------------------------------------------------------------- fig3a
# Matched forward-pass budget: attack vs greedy restricted to an equal pool.
# Values generated from the result JSONs by make_table3.py. Do not edit by hand.
ATTACK = {"Geneformer": [0.003629, 0.006417, 0.010483],
          "scGPT": [0.001144, 0.001941, 0.003058]}
GREEDY_POOL = {"Geneformer": [0.003480, 0.005434, 0.007493],
               "scGPT": [0.000974, 0.001631, 0.002540]}
SIG = {"Geneformer": ["**", "***", "***"],
       "scGPT": ["***", "***", "***"]}
MARGIN = {"Geneformer": [0.000149, 0.000983, 0.002991],
          "scGPT": [0.000170, 0.000310, 0.000518]}

# fig3b: greedy at its natural budget k * n_TF (a single point).
GREEDY_FULL = {"Geneformer": 0.009397, "scGPT": 0.003116}
GREEDY_COST = {"Geneformer": 21028, "scGPT": 21228}
ATTACK_FINAL = {"Geneformer": 0.012855, "scGPT": 0.003112}

# ---------------------------------------------------------------- fig4a, fig4b
# 2x2 cell means at the k * n_TF budget, n = 100 source cells per model.
CELLS = {"Geneformer": {"uniform": [0.012392, 0.012283],     # [improve, uncond]
                        "gradient": [0.012376, 0.012855]},
         "scGPT": {"uniform": [0.003104, 0.003079],
                   "gradient": [0.003073, 0.003112]}}
SCALE = {"Geneformer": 0.012477, "scGPT": 0.003092}   # mean of the four cells

# (label, mean, ci90_low, ci90_high) per model, read directly from
# interaction_2x2_gf_k8.json / interaction_2x2_scgpt_k8.json. The intervals are
# computed from the per-cell paired differences, not derived from the reported
# t statistic. Wilcoxon p values are in the manuscript's supplementary table, not plotted.
CONTRASTS = {
    "Geneformer": [
        ("gradient | improve-only",  -0.000016, -0.000149, +0.000117),
        ("unconditional | uniform",  -0.000108, -0.000213, -0.000004),
        ("gradient | unconditional", +0.000572, +0.000430, +0.000714),
        ("unconditional | gradient", +0.000479, +0.000378, +0.000580),
        ("interaction",              +0.000588, +0.000435, +0.000741)],
    "scGPT": [
        ("gradient | improve-only",  -0.000031, -0.000053, -0.000009),
        ("unconditional | uniform",  -0.000025, -0.000033, -0.000018),
        ("gradient | unconditional", +0.000033, +0.000022, +0.000045),
        ("unconditional | gradient", +0.000039, +0.000022, +0.000057),
        ("interaction",              +0.000064, +0.000044, +0.000084)]}

# Main effects, not plotted: each averages two simple effects of opposite sign.
# Geneformer +0.000278 [+0.000164, +0.000392], +0.000185 [+0.000117, +0.000254];
# scGPT      +0.000001 [-0.000013, +0.000016], +0.000007 [-0.000002, +0.000016].

# ---------------------------------------------------------------- fig5
BUDGETS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
CURVES = {
    "Geneformer": {
        ("uniform", "improve-only"):
            [-0.00034, 0.00001, 0.00188, 0.00411, 0.00666, 0.00860, 0.01021,
             0.01139, 0.01217],
        ("uniform", "unconditional"):
            [-0.00034, 0.00001, 0.00188, 0.00411, 0.00666, 0.00857, 0.01008,
             0.01119, 0.01207],
        ("gradient", "improve-only"):
            [-0.00034, -0.00034, 0.00168, 0.00463, 0.00861, 0.01089, 0.01180,
             0.01224, 0.01237],
        ("gradient", "unconditional"):
            [-0.00034, -0.00034, 0.00168, 0.00463, 0.00861, 0.01090, 0.01180,
             0.01231, 0.01274]},
    "scGPT": {
        ("uniform", "improve-only"):
            [0.00082, 0.00088, 0.00122, 0.00165, 0.00217, 0.00255, 0.00283,
             0.00301, 0.00309],
        ("uniform", "unconditional"):
            [0.00082, 0.00088, 0.00122, 0.00165, 0.00217, 0.00254, 0.00281,
             0.00298, 0.00306],
        ("gradient", "improve-only"):
            [0.00082, 0.00082, 0.00112, 0.00163, 0.00235, 0.00286, 0.00302,
             0.00306, 0.00307],
        ("gradient", "unconditional"):
            [0.00082, 0.00082, 0.00112, 0.00163, 0.00235, 0.00286, 0.00303,
             0.00308, 0.00311]}}

# First budget at which the two acceptance rules separate, computed from the
# raw anytime traces rather than from the rounded CURVES above (differences of
# a few times 1e-6 vanish at five decimal places). Below 1024 the two arms are
# identical to floating-point precision in both models.
#   python make_diverge.py interaction_2x2_gf_k8.json interaction_2x2_scgpt_k8.json
DIVERGE = {("Geneformer", "uniform"): 1024, ("Geneformer", "gradient"): 2048,
           ("scGPT", "uniform"): 1024, ("scGPT", "gradient"): 2048}

# ------------------------------------------------- Figures S1 and S2
# Order sensitivity as a percentage of the improvement achieved by search.
ORDER_SENSITIVITY = {"Geneformer": [3.60, 8.25, 9.97],
                     "scGPT": [0.0, 0.0, 0.0]}     # 1.5e-08 to 9.6e-08 absolute

# k = 2, exhaustive enumeration: percent of the per-cell optimum attained.
OPTIMALITY = [("gradient hill climbing", 85.3),
              ("full attack", 84.3),
              ("greedy", 82.5),
              ("random best", 66.8),
              ("uniform hill climbing", 65.6)]

# Pathways used in the search comparison, a subset of the 18 used elsewhere.
SEARCH_PATHWAYS = ["P01", "P03", "P07", "P12", "P14"]

# ---------------------------------------------------------------- fig2
# Read from JSON at plot time; see figures.py:fig2.
ADDITIVITY_JSON = "additivity_summary.json"
