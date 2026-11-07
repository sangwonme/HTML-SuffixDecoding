"""Figures for POC_study.md (Agg backend). Reads runs/study/*.json, writes runs/study/figures/*.png."""
from __future__ import annotations

import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from study_common import CONDS, FIG, OUT

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
GRAY = "#b9b8b2"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
SHORT = {c: c.replace("_nf4_", " ") for c in CONDS}


def fig_s1():
    d = json.loads((OUT / "s1_predictability.json").read_text(encoding="utf-8"))["conds"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    cls = ["F", "R", "Q", "S", "N"]
    cols = [PAL[0], PAL[2], PAL[3], PAL[1], GRAY]
    names = {"F": "F grammar-forced", "R": "R retrievable", "Q": "Q query copy", "S": "S self-copy", "N": "N novel"}
    for i, c in enumerate(CONDS):
        left = 0
        for k, col in zip(cls, cols):
            v = d[c]["share"][k]
            ax.barh(i, v, left=left, color=col, edgecolor="#fcfcfb", linewidth=1.5, label=names[k] if i == 0 else None)
            left += v
        ax.text(1.01, i, f"π={d[c]['pi']:.2f}", va="center", color=INK, fontsize=8)
    ax.set_yticks(range(len(CONDS)), [SHORT[c] for c in CONDS])
    ax.invert_yaxis(); ax.set_xlim(0, 1.12); ax.set_xlabel("share of output tokens")
    ax.legend(ncol=5, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title("S1  token predictability decomposition", loc="left", fontsize=10, pad=22)
    ax = axes[1]
    groups = ["text", "css_other", "css_numeric_color", "markup", "attr_value", "nonrender_attr", "whitespace", "comment"]
    sel = ["E2B_nf4_rich_zs", "12B_nf4_rich_zs", "E2B_nf4_rich_fs"]
    w = 0.26
    for j, c in enumerate(sel):
        vals = [d[c]["by_context"].get(g, {}).get("share_of_all_N", 0) for g in groups]
        ax.bar(np.arange(len(groups)) + (j - 1) * w, vals, w * 0.9, color=PAL[j], label=SHORT[c])
    ax.set_xticks(range(len(groups)), groups, rotation=30, ha="right")
    ax.set_ylabel("share of all novel (N) tokens")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("where the novel tokens live (lexical context)", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "s1_predictability.png", dpi=150); plt.close(fig)


def fig_s2():
    p = OUT / "s2_pld.json"
    e = OUT / "s2_eta.json"
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        levels = ["L0", "L1", "L2", "L3_d2", "L4t_e0.01", "L4t_e0.05"]
        lab = ["L0\nexact", "L1\nDOM", "L2\nstyle", "L3\nδ=2px", "L4 ε=.01\ntext kept", "L4 ε=.05\ntext kept"]
        ax = axes[0]
        for i, c in enumerate([c for c in CONDS if c in d]):
            a = d[c]["alpha_L"]
            if "L4t_e0.05" not in a:
                continue
            ys = [a["L0_exact"]] + [a[L] for L in levels[1:]]
            ax.plot(range(len(levels)), ys, marker="o", ms=4, lw=2, color=PAL[i % len(PAL)], label=SHORT[c])
            ax.plot([5], [a["L4_e0.05"]], marker="o", ms=5, mfc="none", color=PAL[i % len(PAL)])
        ax.text(5.05, 0.02, "hollow = ε=.05\nwithout text check", fontsize=6, color=INK2, ha="right")
        ax.set_xticks(range(len(levels)), lab, fontsize=7)
        ax.set_ylabel("α_L  (draft-token acceptance)")
        ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7, ncol=2)
        ax.set_title("S2'  PLD acceptance under relaxed equivalence", loc="left", fontsize=10)
    if e.exists():
        d = json.loads(e.read_text(encoding="utf-8"))
        keys = ["L1", "L2", "L3_d2", "L4_e0.01", "L4_e0.05"]
        ax = axes[1]
        cs = [c for c in CONDS if c in d and d[c]]
        w = 0.8 / max(1, len(cs))
        for i, c in enumerate(cs):
            vals = [np.mean([s[k]["eta"] for s in d[c].values()]) for k in keys]
            ax.bar(np.arange(len(keys)) + (i - (len(cs) - 1) / 2) * w, vals, w * 0.9, color=PAL[i % len(PAL)],
                   label=f"{SHORT[c]} (n={len(d[c])})")
        ax.set_xticks(range(len(keys)), ["η_L1\nDOM", "η_L2\nstyle", "η_L3\nδ=2", "η_L4\nε=.01", "η_L4\nε=.05"])
        ax.set_ylabel("redundant token share η"); ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        ax.set_title("level-wise perceptual redundancy", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "s2_equivalence.png", dpi=150); plt.close(fig)


def fig_s3():
    p = OUT / "s3_vc.json"
    if not p.exists():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    ax = axes[0]
    c = "E2B_nf4_rich_fs"
    for s in d[c]["samples"]:
        t = [0] + [cp["t"] for cp in s["checkpoints"]]
        v = [0] + [cp["vc_ssim"] for cp in s["checkpoints"]]
        ax.step(t, v, where="post", color=PAL[0], alpha=0.45, lw=1.2)
    ax.set_xlabel("wall-clock time (s, measured per-token rate)"); ax.set_ylabel("VC(t)  (SSIM-normalized)")
    ax.set_title(f"S3  VC(t), {SHORT[c]}, n={len(d[c]['samples'])}", loc="left", fontsize=10)
    ax = axes[1]
    for i, c in enumerate([c for c in CONDS if c in d]):
        fr = np.linspace(0.05, 1, 20)
        m = np.mean([[cp["vc_ssim"] for cp in s["checkpoints"]] for s in d[c]["samples"]], axis=0)
        ax.plot(fr, m, lw=2, color=PAL[i % len(PAL)], label=f"{SHORT[c]}  Lperc/T={d[c]['agg']['ratio_ssim']:.2f}")
    ax.plot([0, 1], [0, 1], ls=":", color=INK2, lw=1)
    ax.text(0.55, 0.08, "JSON/SQL: VC=0 until complete\n(step at t/T=1)", color=INK2, fontsize=7)
    ax.set_xlabel("fraction of tokens generated  t/T"); ax.set_ylabel("mean VC")
    ax.legend(frameon=False, fontsize=7)
    ax.set_title("mean VC vs progress", loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "s3_vc.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    fig_s1(); fig_s2(); fig_s3()
    print("figures:", sorted(x.name for x in FIG.glob("*.png")))
