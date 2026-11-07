"""Paper figures without abbreviations: draft_sim_paper.png, ranker_paper.png (Agg)."""
from __future__ import annotations

import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from study_common import FIG, OUT

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
GREY, GREY2 = "#b9b8b2", "#8f8e88"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 10,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def bars(ax, groups, labels, series, ylabel):
    """series: list of (name, values, color, is_oracle)"""
    n = len(series)
    w = 0.8 / n
    for j, (name, vals, col, orc) in enumerate(series):
        x = np.arange(len(groups)) + (j - (n - 1) / 2) * w
        light = orc and ("best of" in name)
        ax.bar(x, vals, w * 0.92, color=("#dcdbd5" if light else GREY) if orc else col, edgecolor=GREY2 if orc else SURF,
               hatch=("..." if light else "///") if orc else None, linewidth=1.0, label=name)
    ax.axhline(100, color=INK2, lw=1, ls=":")
    ax.set_xticks(range(len(groups)), labels)
    ax.set_ylabel(ylabel)


def fig_draft_sim():
    d = json.loads((OUT / "draft_sim.json").read_text(encoding="utf-8"))["sets"]
    sets = ["C1", "C4g", "C7g", "C0", "gemma_E2B_fs", "gemma_E4B_fs", "gemma_12B_fs"]
    labels = ["PoC\nretrieved\nexamples", "PoC\nstructure-\nretrieved", "PoC\ninterleaved-\nCSS examples", "PoC\nno examples",
              "Gemma E2B\n3 examples*", "Gemma E4B\n3 examples*", "Gemma 12B\n3 examples*"]
    pols = [("standard lookup (n-gram up to 3)", "std", BLUE, False), ("longer lookup (n-gram up to 8)", "n8", ORANGE, False),
            ("majority vote of 4 matches", "k4vote", YELLOW, False), ("structure rule (DOM-aware)", "dom", AQUA, False),
            ("best of 4 matches (oracle, 4x verify)", "k4oracle", None, True), ("any position (oracle)", "oracle", None, True)]
    fig, ax = plt.subplots(figsize=(13, 5))
    series = [(n, [100 * d[s][f"fwd_{k}"] / d[s]["fwd_std"] for s in sets], c, o) for n, k, c, o in pols]
    bars(ax, sets, labels, series, "forward passes per request\n(% of standard lookup)")
    for i, s in enumerate(sets):
        ax.text(i, 103, f"{d[s]['fwd_std']:.0f}", ha="center", fontsize=9, color=INK2)
    ax.set_ylim(0, 112)
    ax.set_title("Choosing where to copy the draft from (simulated exact verification, 16-token drafts)", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.text(0.01, 0.01, "Numbers above bars: standard-lookup forward passes per request. *Gemma targets are re-tokenized responses (approximate). "
             "Grey patterned bars = oracle upper bounds (not executable).", fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / "draft_sim_paper.png", dpi=200); plt.close(fig)


def fig_ranker():
    d = json.loads((OUT / "ranker.json").read_text(encoding="utf-8"))["groups"]
    ab = OUT / "ranker_mlp_ablation.json"
    abl = json.loads(ab.read_text(encoding="utf-8"))["groups"] if ab.exists() else None
    gs = ["all", "poc_greedy", "gate2_measured", "cands6", "gemma_approx"]
    labels = ["all\n(n=444)", "PoC greedy\n(n=40)", "measured\nGPU outputs (n=74)", "one-example\noutputs (n=300)", "Gemma\n(n=30)*"]
    mlp_key = "fwd_mlp_full"
    pols = [("standard lookup", "fwd_std", BLUE, False), ("structure rule", "fwd_dom_rule", AQUA, False),
            ("learned ranker (MLP)", mlp_key, ORANGE, False), ("best of top 8 (oracle)", "fwd_top8oracle", None, True),
            ("any position (oracle)", "fwd_oracle", None, True)]
    ncol = 2 if abl else 1
    fig, axes = plt.subplots(1, ncol, figsize=(15 if abl else 10, 5), gridspec_kw={"width_ratios": [2.2, 1]} if abl else None)
    ax = axes[0] if abl else axes
    src = lambda g, k: (abl[g][k] if (abl and k == mlp_key) else d[g][k])
    series = [(n, [100 * src(g, k) / d[g]["fwd_std"] for g in gs], c, o) for n, k, c, o in pols]
    bars(ax, gs, labels, series, "forward passes per request\n(% of standard lookup)")
    ax.set_ylim(0, 112)
    ax.set_title("Learned ranker over the structure rule's top-8 candidates", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    if abl:
        ax2 = axes[1]
        a = abl["all"]
        items = [("remove\nstructure features", a["noStruct_minus_full"], BLUE),
                 ("remove\nhistory features", a["noHist_minus_full"], YELLOW),
                 ("structure rule\ninstead of ranker", a["rule_minus_full"], AQUA)]
        for i, (nm, p_, col) in enumerate(items):
            m, lo, hi = p_["mean"], p_["ci95"][0], p_["ci95"][1]
            ax2.bar(i, m, 0.6, color=col, edgecolor=SURF)
            ax2.errorbar(i, m, yerr=[[m - lo], [hi - m]], color=INK, capsize=5, lw=1.5)
            ax2.text(i, hi + 0.3, f"{m:+.1f} [{lo:.1f}, {hi:.1f}]", ha="center", fontsize=9.5, color=INK)
        ax2.axhline(0, color=INK2, lw=1)
        ax2.set_ylim(-1, max(x[1]["ci95"][1] for x in items) + 1.8)
        ax2.set_xticks(range(3), [x[0] for x in items])
        ax2.set_ylabel("extra forward passes per request\nvs. learned ranker (all, n=444)")
        ax2.set_title("Feature ablation of the learned ranker (MLP)\nmean and 95% bootstrap CI, paired by request", loc="left", fontsize=11)
    fig.text(0.01, 0.01, "App-level 5-fold cross-fitting. Simulated exact verification. *approximate targets. Grey patterned bars = oracle upper bounds (not executable).",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIG / "ranker_paper.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    fig_draft_sim()
    fig_ranker()
    print("ok")
