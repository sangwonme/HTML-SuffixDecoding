"""Post-hoc evaluation of the RNSD PoC runs (headless Playwright only, figures with MPLBACKEND=Agg).

For each condition c != greedy and each sample, compares y_c with y_greedy:
  identical token ids / first divergence index
  L1 DOM equivalence   : canonical DOM serialization (attributes sorted, whitespace-only text and comments dropped)
  L2 style equivalence : per element (document order) tag + full getComputedStyle; class names ignored
  L3 layout equivalence: per element bounding box within delta = 1 px
  L4 perceptual        : D = 1 - SSIM(viewport screenshots), plus pixel-difference ratios
plus accuracy (checklist score, element F1) and speed (total time, target forwards, tokens/forward, speed-up).
Usage: .venv/Scripts/python.exe -I scripts/poc_eval.py --model E2B [--tag ""]
Writes runs/poc/<model>_eval.jsonl, runs/poc/<model>_summary.md, runs/poc/fig_*.png
"""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "ms-playwright"))
os.environ["MPLBACKEND"] = "Agg"

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402
from skimage.metrics import structural_similarity  # noqa: E402

from genui.checklist import ChecklistScorer  # noqa: E402
from genui.metrics import Scorer, extract_html  # noqa: E402

POC = ROOT / "runs" / "poc"
GAMMA = 16
DELTA_PX = 1.0

DOM_JS = r"""() => {
  const ser = n => {
    if (n.nodeType === 3) { const t = n.textContent.replace(/\s+/g, ' ').trim(); return t ? JSON.stringify(t) : ''; }
    if (n.nodeType !== 1) return '';
    const attrs = Array.from(n.attributes).map(a => a.name + '=' + JSON.stringify(a.value)).sort().join(' ');
    return '<' + n.tagName.toLowerCase() + (attrs ? ' ' + attrs : '') + '>' +
           Array.from(n.childNodes).map(ser).join('') + '</' + n.tagName.toLowerCase() + '>';
  };
  return ser(document.documentElement);
}"""
STYLE_JS = r"""() => Array.from(document.querySelectorAll('body, body *')).map(el => {
  const cs = getComputedStyle(el); const parts = [];
  for (let i = 0; i < cs.length; i++) parts.push(cs[i] + ':' + cs.getPropertyValue(cs[i]));
  return el.tagName + '|' + parts.join(';');
})"""
BOX_JS = r"""() => Array.from(document.querySelectorAll('body, body *')).map(el => {
  const r = el.getBoundingClientRect(); return [el.tagName, r.x, r.y, r.width, r.height];
})"""


def load(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    return rows[0]["_meta"], rows[1:]


def render(page, html):
    page.set_content(html, wait_until="load", timeout=10000)
    shot = np.array(Image.open(io.BytesIO(page.screenshot(full_page=False))).convert("RGB"))
    return {"shot": shot, "dom": page.evaluate(DOM_JS), "style": page.evaluate(STYLE_JS), "box": page.evaluate(BOX_JS)}


def compare(a, b):
    sa = np.array(Image.fromarray(a["shot"]).convert("L"))
    sb = np.array(Image.fromarray(b["shot"]).convert("L"))
    ssim = float(structural_similarity(sa, sb, data_range=255))
    diff = np.abs(a["shot"].astype(int) - b["shot"].astype(int)).max(-1)
    l3 = len(a["box"]) == len(b["box"]) and all(
        x[0] == y[0] and max(abs(x[i] - y[i]) for i in range(1, 5)) <= DELTA_PX for x, y in zip(a["box"], b["box"]))
    return {"L1_dom_eq": a["dom"] == b["dom"], "L2_style_eq": a["style"] == b["style"], "L3_box_eq": bool(l3),
            "ssim": ssim, "D": 1 - ssim, "pix_diff_any": float((diff > 0).mean()), "pix_diff_16": float((diff > 16).mean())}


def pct(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="E2B")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    global POC
    if args.dir:
        POC = Path(args.dir)
    files = {p.stem.split("_", 1)[1].replace(args.tag, "") if args.tag else p.stem.split("_", 1)[1]: p
             for p in sorted(POC.glob(f"{args.model}_*{args.tag}.jsonl"))
             if not p.stem.endswith(("_eval", "_run")) and "_eval" not in p.stem}
    runs = {c: load(p) for c, p in files.items()}
    assert "greedy" in runs, files
    mid = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/mid/pilot.jsonl", encoding="utf-8")}
    gt = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/pilot.jsonl", encoding="utf-8")}
    g_by = {r["id"]: r for r in runs["greedy"][1]}
    gb_by = {r["id"]: r for r in runs["greedy_b"][1]} if "greedy_b" in runs else None
    LCONDS = ("L1", "L2", "L4")

    def base_of(c):
        """Reference greedy run for condition c: same-time greedy_b for L*, greedy_<tag> for *_<tag>, else greedy."""
        if c in LCONDS and gb_by:
            return "greedy_b"
        if "_C" in c and not c.startswith("greedy"):
            t = "greedy_" + c.split("_", 1)[1]
            if t in runs:
                return t
        return "greedy"
    by_cond = {c: {r["id"]: r for r in rows} for c, (_, rows) in runs.items()}
    recs = []
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=True)
        ctx = br.new_context(viewport={"width": 360, "height": 640}, device_scale_factor=1)
        ctx.route("**/*", lambda route: route.abort())
        page = ctx.new_page()
        cs, es = ChecklistScorer(br), Scorer(br)
        rcache = {}
        for c, (meta, rows) in runs.items():
            for r in rows:
                g = by_cond[base_of(c)].get(r["id"]) if c != base_of(c) else g_by.get(r["id"])
                html = extract_html(r["response"]) or ""
                key = (c, r["id"])
                rcache[key] = render(page, html) if html else None
                rec = {"cond": c, "id": r["id"], "output_tokens": r["output_tokens"], "total_s": r["total_s"],
                       "ttft_s": r["ttft_s"], "n_fwd": r["n_fwd"], "tokens_per_fwd": r["tokens_per_fwd"],
                       "hit_max_new": r["hit_max_new"], "fires": r.get("fires") or {"a": r["fire_a"], "b": r["fire_b"]}}
                rec.update({k: v for k, v in cs.score(r["response"], mid[r["id"]]["checklist"]).items()
                            if k in ("check_score", "check_all")})
                e = es.score(r["response"], gt[r["id"]])
                rec.update({"render_ok": e["render_ok"], "elem_f1": e["elem_f1"], "type_f1": e["type_f1"]})
                if g is not None:
                    rec["speedup"] = g["total_s"] / r["total_s"]
                    rec["fwd_ratio"] = g["n_fwd"] / r["n_fwd"]
                    rec["identical_ids"] = r["gen_ids"] == g["gen_ids"]
                    rec["first_div"] = next((i for i, (x, y) in enumerate(zip(r["gen_ids"], g["gen_ids"])) if x != y),
                                            None if len(r["gen_ids"]) == len(g["gen_ids"]) else min(len(r["gen_ids"]), len(g["gen_ids"])))
                recs.append(rec)
        for rec in recs:
            base = base_of(rec["cond"])
            a, b = rcache.get((rec["cond"], rec["id"])), rcache.get((base, rec["id"]))
            rec["baseline"] = base
            if a and b:
                rec.update(compare(a, b))
        br.close()
    with open(POC / f"{args.model}{args.tag}_eval.jsonl", "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec) + "\n")

    # ---- per-position acceptance
    alpha = {}
    for c, (meta, rows) in runs.items():
        if c.startswith("greedy"):
            continue
        reach = np.sum([r["pos_reach"] for r in rows], 0)
        ex = np.sum([r["pos_exact"] for r in rows], 0)
        acc = np.sum([r["pos_acc"] for r in rows], 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            alpha[c] = {"reach": reach.tolist(), "alpha": (ex / reach).tolist(), "alpha_p": (acc / reach).tolist()}

    # ---- table
    order = [c for c in ("greedy", "pld", "rnsd", "greedy_b", "L1", "L2", "L4") if c in runs]
    cols = ["cond", "n", "out_tok", "TTFT p50", "total p50", "total p95(max)", "fwd mean", "tok/fwd", "speedup p50",
            "E[tok/step] formula", "identical", "L1 eq", "L2 eq", "L3 eq", "D p50", "D max", "pixdiff>16 p50",
            "checklist", "elem F1", "render_ok", "hit_max", "fires"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    summary = {}
    for c in order:
        rs = [x for x in recs if x["cond"] == c]
        m = lambda k: statistics.mean(float(x[k]) for x in rs if k in x)  # noqa: E731
        fires = {}
        for x in rs:
            for k, v in (x["fires"] or {}).items():
                fires[k] = fires.get(k, 0) + v
        if c in alpha:
            ap_ = np.array(alpha[c]["alpha_p"], dtype=float)
            reach = np.array(alpha[c]["reach"])
            a_bar = float(np.nansum(ap_ * reach) / reach.sum()) if reach.sum() else float("nan")
            e_form = (1 - a_bar ** (GAMMA + 1)) / (1 - a_bar) if a_bar < 1 else GAMMA + 1
        else:
            a_bar, e_form = None, 1.0
        row = {"cond": c, "n": len(rs), "out_tok": m("output_tokens"), "TTFT p50": pct([x["ttft_s"] for x in rs], 50),
               "total p50": pct([x["total_s"] for x in rs], 50), "total p95(max)": max(x["total_s"] for x in rs),
               "fwd mean": m("n_fwd"), "tok/fwd": m("tokens_per_fwd"),
               "speedup p50": pct([x.get("speedup", 1.0) for x in rs], 50), "E[tok/step] formula": e_form,
               "identical": m("identical_ids") if not c.startswith("greedy") else 1.0,
               "L1 eq": m("L1_dom_eq") if not c.startswith("greedy") else 1.0, "L2 eq": m("L2_style_eq") if not c.startswith("greedy") else 1.0,
               "L3 eq": m("L3_box_eq") if not c.startswith("greedy") else 1.0,
               "D p50": pct([x["D"] for x in rs if "D" in x], 50) if not c.startswith("greedy") else 0.0,
               "D max": max([x["D"] for x in rs if "D" in x] or [0.0]),
               "pixdiff>16 p50": pct([x["pix_diff_16"] for x in rs if "pix_diff_16" in x], 50) if not c.startswith("greedy") else 0.0,
               "checklist": m("check_score"), "elem F1": m("elem_f1"), "render_ok": m("render_ok"),
               "hit_max": m("hit_max_new"), "fires": json.dumps(fires), "alpha_bar_prime": a_bar}
        summary[c] = row
        md.append("| " + " | ".join(f"{row[k]:.3g}" if isinstance(row[k], float) else str(row[k]) for k in cols) + " |")
    (POC / f"{args.model}{args.tag}_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (POC / f"{args.model}{args.tag}_summary.json").write_text(json.dumps({"table": summary, "alpha": alpha}, indent=1))
    print("\n".join(md))

    # ---- figures
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for c in [x for x in order if not x.startswith("greedy")]:
        rs = [x for x in recs if x["cond"] == c and "D" in x]
        ax.scatter([x["D"] for x in rs], [x["speedup"] for x in rs], label=c, s=28, alpha=0.8)
    ax.set_xlabel("D(R(y_c), R(y_greedy)) = 1 - SSIM")
    ax.set_ylabel("wall-clock speed-up vs greedy")
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.axhline(1, color="gray", lw=0.8)
    ax.legend()
    ax.set_title(f"{args.model}: speed-up vs perceptual distance (n={len(runs['greedy'][1])})")
    fig.tight_layout()
    fig.savefig(POC / f"fig_speedup_vs_D_{args.model}{args.tag}.png", dpi=130)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for c in [x for x in order if x in alpha and x != "pld"]:
        pos = np.arange(GAMMA)
        ax.plot(pos, alpha[c]["alpha_p"][:GAMMA], marker="o", ms=3, label=f"alpha' ({c})")
    if "pld" in alpha:
        ax.plot(np.arange(GAMMA), alpha["pld"]["alpha"][:GAMMA], "k--", marker="s", ms=3, label="alpha (pld, exact)")
    ax.set_xlabel("draft position i")
    ax.set_ylabel("acceptance rate (conditional on reaching i)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.set_title(f"{args.model}: per-position acceptance alpha vs alpha'")
    fig.tight_layout()
    fig.savefig(POC / f"fig_alpha_by_pos_{args.model}{args.tag}.png", dpi=130)


if __name__ == "__main__":
    main()
