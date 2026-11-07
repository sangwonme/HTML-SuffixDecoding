"""S5: example overlap omega_4 vs accuracy, output length, and accepted tokens per forward.

omega_4(e, x) = share of kappa(x) tokens covered by a 4-gram that also occurs in kappa(e).
  omega_max = max over the 3 few-shot examples, omega_union = coverage by the union of their 4-grams.
x = GT flow HTML (oracle overlap of the retrieved examples with the target screen) and x = model output y.
Correlations: Spearman rho with 95% bootstrap CI (2000 resamples, seed 0).

  .venv/Scripts/python.exe -I scripts/study_s5.py [--checklist]
Output: runs/study/s5_omega.json, figures/s5_omega.png
"""
from __future__ import annotations

import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import spearmanr

from study_common import FIG, OUT, ROOT, encode, guard_setup, load_jsonl, load_runs, retrieved_examples
from study_canon import kappa
from genui.metrics import extract_html

FS = ["E2B_nf4_rich_fs", "E4B_nf4_rich_fs", "12B_nf4_rich_fs"]


def grams(ids, n=4):
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


def coverage(ids, gset, n=4):
    m = np.zeros(len(ids), bool)
    for i in range(len(ids) - n + 1):
        if tuple(ids[i:i + n]) in gset:
            m[i:i + n] = True
    return float(m.mean()) if len(ids) else 0.0


_kc = {}


def kids(html):
    if html not in _kc:
        _kc[html] = encode(kappa(html, 2))[0]
    return _kc[html]


COMMON: set = set()


def build_common(pool_rows, n_docs=1000, thr=0.2):
    """4-grams present in > thr of a fixed sample of train_pool kappa(GT): template boilerplate."""
    from collections import Counter
    df = Counter()
    for r in pool_rows[:n_docs]:
        df.update(grams(kids(r["gt_html_flow"])))
    COMMON.update(g for g, v in df.items() if v > thr * n_docs)


def coverage_rare(ids, gset, n=4):
    """coverage restricted to positions whose 4-gram is not boilerplate; denominator = non-boilerplate positions."""
    m = np.zeros(len(ids), bool)
    elig = np.zeros(len(ids), bool)
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        if g in COMMON:
            continue
        elig[i:i + n] = True
        if g in gset:
            m[i:i + n] = True
    return float(m[elig].mean()) if elig.any() else 0.0


def omega(examples, x):
    xi = kids(x)
    gs = [grams(kids(e)) for e in examples]
    u = set().union(*gs)
    return {"max": max(coverage(xi, g) for g in gs), "union": coverage(xi, u), "rare": coverage_rare(xi, u)}


def spear(a, b, B=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return {"rho": None, "n": int(len(a))}
    rho, p = spearmanr(a, b)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(B):
        idx = rng.integers(0, len(a), len(a))
        if np.std(a[idx]) > 0 and np.std(b[idx]) > 0:
            bs.append(spearmanr(a[idx], b[idx])[0])
    return {"rho": float(rho), "p": float(p), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], "n": int(len(a))}


def main():
    guard_setup()
    pilot, runs = load_runs()
    pool = {r["id"]: r for r in load_jsonl(ROOT / "data" / "processed" / "train_pool.jsonl")}
    mid = {r["id"]: r for r in load_jsonl(ROOT / "data" / "processed" / "mid" / "pilot.jsonl")}
    build_common(load_jsonl(ROOT / "data" / "processed" / "train_pool.jsonl"))
    print("boilerplate 4-grams:", len(COMMON))
    rows = []
    for c in FS:
        scores = {s["id"]: s for s in load_jsonl(ROOT / "runs" / "gemma" / f"{c}_pilot.scores.jsonl") if "id" in s}
        for r in runs[c]:
            g = pilot[r["id"]]
            exs = [e["gt_html_flow"] for e in retrieved_examples(g)]
            y = extract_html(r["response"])
            og = omega(exs, g["gt_html_flow"])
            oy = omega(exs, y)
            rows.append({"cond": c, "id": r["id"], "omega_gt_max": og["max"], "omega_gt_union": og["union"], "omega_gt_rare": og["rare"], "omega_y_rare": oy["rare"],
                         "omega_y_max": oy["max"], "omega_y_union": oy["union"],
                         "elem_f1": scores.get(r["id"], {}).get("elem_f1", np.nan),
                         "type_f1": scores.get(r["id"], {}).get("type_f1", np.nan),
                         "out_tok": r["output_tokens"], "total_s": r["total_s"], "html": y})
    # checklist scores (needs one headless Chromium)
    if "--checklist" in sys.argv:
        from playwright.sync_api import sync_playwright
        import genui.checklist as CL
        cls = [getattr(CL, n) for n in dir(CL) if isinstance(getattr(CL, n), type) and hasattr(getattr(CL, n), "score")][0]
        with sync_playwright() as pw:
            br = pw.chromium.launch(headless=True)
            sc = cls(br)
            for row in rows:
                s = sc.score("```html\n" + row["html"] + "\n```", mid[row["id"]]["checklist"])
                row["check_score"] = s.get("check_score", np.nan)
            br.close()
    # PoC runs (PLD / RNSD): examples from the run's example_ids
    poc = []
    for cond in ("pld", "rnsd", "greedy"):
        p = ROOT / "runs" / "poc" / f"E2B_{cond}.jsonl"
        if not p.exists():
            continue
        for r in load_jsonl(p):
            if "id" not in r or not r.get("example_ids"):
                continue
            exs = [pool[i]["gt_html_flow"] for i in r["example_ids"] if i in pool]
            if not exs:
                continue
            g = pilot.get(r["id"])
            gt = g["gt_html_flow"] if g else None
            out = extract_html(r.get("response") or r.get("text") or "") if (r.get("response") or r.get("text")) else None
            poc.append({"cond": cond, "id": r["id"], "tok_per_fwd": r.get("tokens_per_fwd"), "total_s": r.get("total_s"),
                        "output_tokens": r.get("output_tokens"),
                        "omega_gt_max": omega(exs, gt)["max"] if gt else np.nan,
                        "omega_gt_union": omega(exs, gt)["union"] if gt else np.nan,
                        "omega_gt_rare": omega(exs, gt)["rare"] if gt else np.nan,
                        "omega_y_union": omega(exs, out)["union"] if out else np.nan})
    res = {"rows": [{k: v for k, v in r.items() if k != "html"} for r in rows], "poc": poc, "corr": {}}
    for key in ("omega_gt_max", "omega_gt_union", "omega_gt_rare", "omega_y_union", "omega_y_rare"):
        for tgt in ("elem_f1", "type_f1", "check_score", "out_tok", "total_s"):
            if tgt in rows[0]:
                res["corr"][f"{key}~{tgt}"] = spear([r[key] for r in rows], [r[tgt] for r in rows])
    for cond in ("pld", "rnsd"):
        sub = [p for p in poc if p["cond"] == cond]
        for key in ("omega_gt_union", "omega_gt_rare", "omega_y_union"):
            res["corr"][f"poc_{cond}:{key}~tok_per_fwd"] = spear([p[key] for p in sub], [p["tok_per_fwd"] for p in sub])
    for k, v in res["corr"].items():
        print(f"{k:42s} {v}")
    (OUT / "s5_omega.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PAL = {"E2B_nf4_rich_fs": "#2a78d6", "E4B_nf4_rich_fs": "#eb6834", "12B_nf4_rich_fs": "#1baf7a",
           "pld": "#2a78d6", "rnsd": "#eb6834"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    acc = "check_score" if "check_score" in rows[0] else "elem_f1"
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for c in FS:
        sub = [r for r in rows if r["cond"] == c]
        axes[0].scatter([r["omega_gt_union"] for r in sub], [r[acc] for r in sub], s=36, color=PAL[c],
                        edgecolor="#fcfcfb", linewidth=1.5, label=c.replace("_nf4_", " "))
        axes[1].scatter([r["omega_gt_union"] for r in sub], [r["out_tok"] for r in sub], s=36, color=PAL[c],
                        edgecolor="#fcfcfb", linewidth=1.5, label=c.replace("_nf4_", " "))
    cr = res["corr"].get(f"omega_gt_union~{acc}", {})
    axes[0].set_title(f"ω (examples vs GT) vs {acc}  ρ={cr.get('rho') or float('nan'):.2f}", loc="left", fontsize=10)
    axes[0].set_xlabel("ω_4 union (κ space)"); axes[0].set_ylabel(acc); axes[0].legend(frameon=False, fontsize=7)
    cr = res["corr"].get("omega_gt_union~out_tok", {})
    axes[1].set_title(f"ω vs output length  ρ={cr.get('rho') or float('nan'):.2f}", loc="left", fontsize=10)
    axes[1].set_xlabel("ω_4 union (κ space)"); axes[1].set_ylabel("output tokens")
    for cond in ("pld", "rnsd"):
        sub = [p for p in poc if p["cond"] == cond]
        if sub:
            cr = res["corr"].get(f"poc_{cond}:omega_gt_union~tok_per_fwd", {})
            axes[2].scatter([p["omega_gt_union"] for p in sub], [p["tok_per_fwd"] for p in sub], s=36, color=PAL[cond],
                            edgecolor="#fcfcfb", linewidth=1.5, label=f"E2B {cond} (ρ={cr.get('rho') or float('nan'):.2f})")
    axes[2].set_title("ω vs accepted tokens / forward (runs/poc)", loc="left", fontsize=10)
    axes[2].set_xlabel("ω_4 union (κ space)"); axes[2].set_ylabel("tokens per forward"); axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "s5_omega.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
