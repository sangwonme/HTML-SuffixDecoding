"""Combined S6 figure: VC(t) for orders o, b, s (s6_reorder.json) and i, si (s6b_interleave.json)."""
from __future__ import annotations

import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from study_common import FIG, OUT

PAL = {"o": "#2a78d6", "b": "#eb6834", "s": "#eda100", "i": "#1baf7a", "si": "#6250d6"}
NAME = {"o": "(o) original", "b": "(b) body-first, CSS last", "s": "(s) Smith blocks, slot bound",
        "i": "(i) CSS interleaved per block (verbatim)", "si": "(si) interleaved + Smith, slot bound"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
a = json.loads((OUT / "s6_reorder.json").read_text(encoding="utf-8"))
b = json.loads((OUT / "s6b_interleave_fair.json").read_text(encoding="utf-8"))


def curves(c, sid=None):
    ra = {r["id"]: r for r in a[c]["samples"]}
    rb = {r["id"]: r for r in b[c]["samples"] if r.get("ok")}
    ids = [sid] if sid else [i for i in ra if i in rb]
    out = {}
    for k in ("o", "b", "s"):
        out[k] = np.mean([[cp[k] for cp in ra[i]["checkpoints"]] for i in ids if k in ra[i]["checkpoints"][0]], axis=0)
    for k in ("i", "si"):
        out[k] = np.mean([[cp[k] for cp in rb[i]["checkpoints"]] for i in ids], axis=0)
    return out, len(ids)


fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.6), sharey=True)
c = "E2B_nf4_rich_fs"
rows = sorted([r for r in b[c]["samples"] if r.get("ok")], key=lambda r: r["Lperc_i"])
picks = [rows[0]["id"], rows[len(rows) // 2]["id"], rows[-1]["id"]]
fr = [0] + list(np.linspace(0.05, 1, 20))
for ax, sid in zip(axes[:3], picks):
    cv, _ = curves(c, sid)
    for k in ("o", "b", "s", "i", "si"):
        ax.step(fr, [0] + list(cv[k]), where="post", lw=1.8, color=PAL[k])
    ax.set_title(f"{c.replace('_nf4_', ' ')} {sid}", loc="left", fontsize=9)
    ax.set_xlabel("decoding progress t/T")
axes[0].set_ylabel("VC(t)  (SSIM-normalized)")
cv, n = curves(c)
ax = axes[3]
for k in ("o", "b", "s", "i", "si"):
    ax.step(fr, [0] + list(cv[k]), where="post", lw=2, color=PAL[k], label=NAME[k])
ax.set_title(f"mean, {c.replace('_nf4_', ' ')} (n={n})", loc="left", fontsize=9)
ax.set_xlabel("decoding progress t/T")
ax.legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "s6_reorder.png", dpi=150); plt.close(fig)
print("ok", picks)
