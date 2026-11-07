"""S5b: condition-level view. omega_4 (kappa space) between the retrieved top-3 examples and each condition's
output y, next to latency and accuracy.  Same retrieved examples for every condition (zs: not in the prompt).
Output: runs/study/s5_conditions.json
"""
from __future__ import annotations

import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from study_common import CONDS, OUT, ROOT, guard_setup, load_jsonl, load_runs, retrieved_examples
from study_s5 import build_common, omega, spear
from genui.metrics import extract_html


def main():
    guard_setup()
    pilot, runs = load_runs()
    build_common(load_jsonl(ROOT / "data" / "processed" / "train_pool.jsonl"))
    res, allrows = {}, []
    for c in CONDS:
        sc = {s["id"]: s for s in load_jsonl(ROOT / "runs" / "gemma" / f"{c}_pilot.scores.jsonl") if "id" in s}
        rows = []
        for r in runs[c]:
            exs = [e["gt_html_flow"] for e in retrieved_examples(pilot[r["id"]])]
            o = omega(exs, extract_html(r["response"]))
            rows.append({"cond": c, "id": r["id"], "omega_y_union": o["union"], "omega_y_rare": o["rare"], "total_s": r["total_s"],
                         "out_tok": r["output_tokens"], "elem_f1": sc.get(r["id"], {}).get("elem_f1", np.nan)})
        allrows += rows
        res[c] = {k: float(np.nanmean([x[k] for x in rows])) for k in rows[0] if k not in ("cond", "id")}
        print(c, {k: round(v, 3) for k, v in res[c].items()})
    res["pooled_70"] = {f"omega_y_union~{t}": spear([x["omega_y_union"] for x in allrows], [x[t] for x in allrows])
                        for t in ("total_s", "out_tok", "elem_f1")}
    print(res["pooled_70"])
    res["rows"] = allrows
    (OUT / "s5_conditions.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    fig_main(res)


def fig_main(res):
    """Main S5 figure: omega(e, y) vs accuracy, latency (n=70 pooled) and tokens/forward (runs/poc, n=10)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from study_common import FIG
    PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    rows = res["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, c in enumerate(CONDS):
        sub = [r for r in rows if r["cond"] == c]
        kw = dict(s=30, color=PAL[i], edgecolor="#fcfcfb", linewidth=1.2, label=c.replace("_nf4_", " "))
        axes[0].scatter([r["omega_y_union"] for r in sub], [r["elem_f1"] for r in sub], **kw)
        axes[1].scatter([r["omega_y_union"] for r in sub], [r["total_s"] for r in sub], **kw)
    pc = res["pooled_70"]
    axes[0].set_title(f"ω(e,y) vs element F1  ρ={pc['omega_y_union~elem_f1']['rho']:+.2f} (n=70)", loc="left", fontsize=10)
    axes[0].set_xlabel("ω_4(examples, output)  κ space"); axes[0].set_ylabel("element F1")
    axes[0].legend(frameon=False, fontsize=6, ncol=2)
    axes[1].set_title(f"ω(e,y) vs latency  ρ={pc['omega_y_union~total_s']['rho']:+.2f} (n=70)", loc="left", fontsize=10)
    axes[1].set_xlabel("ω_4(examples, output)  κ space"); axes[1].set_ylabel("total seconds")
    s5 = json.loads((OUT / "s5_omega.json").read_text(encoding="utf-8"))
    for j, cond in enumerate(("pld", "rnsd")):
        sub = [p for p in s5["poc"] if p["cond"] == cond]
        cr = s5["corr"].get(f"poc_{cond}:omega_y_union~tok_per_fwd", {})
        axes[2].scatter([p["omega_y_union"] for p in sub], [p["tok_per_fwd"] for p in sub], s=36, color=PAL[j],
                        edgecolor="#fcfcfb", linewidth=1.2, label=f"E2B {cond}  ρ={cr.get('rho', float('nan')):+.2f}")
    axes[2].set_title("ω(e,y) vs accepted tokens/forward (runs/poc, n=10)", loc="left", fontsize=10)
    axes[2].set_xlabel("ω_4(examples, output)  κ space"); axes[2].set_ylabel("tokens per forward")
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "s5_omega_main.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
