"""Quality of measured outputs in a run directory (CPU, headless): checklist (query_mid) + element/type F1 vs GT flow.

Usage: .venv/Scripts/python.exe -I scripts/quality_eval.py DIR [ROWS_JSONL] [GT_JSONL]
  defaults: ROWS = data/processed/mid/pilot.jsonl, GT = data/processed/pilot.jsonl
  (test: ROWS = runs/poc/E/test100_rows.jsonl, GT = data/processed/test.jsonl)
Writes DIR/quality.jsonl and DIR/quality.md (means per condition; paired CI vs DOM and vs PLD, rep 0).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "ms-playwright"))
from playwright.sync_api import sync_playwright  # noqa: E402

from genui.checklist import ChecklistScorer  # noqa: E402
from genui.metrics import Scorer  # noqa: E402

d = Path(sys.argv[1])
rows_f = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data/processed/mid/pilot.jsonl"
gt_f = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "data/processed/pilot.jsonl"
mid = {json.loads(l)["id"]: json.loads(l) for l in open(rows_f, encoding="utf-8")}
gt = {json.loads(l)["id"]: json.loads(l) for l in open(gt_f, encoding="utf-8")}
recs = []
with sync_playwright() as pw:
    br = pw.chromium.launch(headless=True)
    cs, es = ChecklistScorer(br), Scorer(br)
    gcache = {}
    for f in sorted(d.glob("E2B_*.jsonl")):
        c = f.stem.split("_", 1)[1]
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if "id" not in r or r.get("rep", 0) != 0 or r["id"] not in mid or r["id"] not in gt:
                continue
            try:
                ck = cs.score(r["response"], mid[r["id"]]["checklist"])["check_score"]
            except Exception:  # noqa: BLE001  page crash -> fresh page, NaN
                ck, cs = float("nan"), ChecklistScorer(br)
            try:
                if r["id"] not in gcache:
                    gcache[r["id"]] = es.analyze(gt[r["id"]]["gt_html_flow"])
                e = es.score(r["response"], gt[r["id"]], gcache[r["id"]])
            except Exception:  # noqa: BLE001
                e, es = {"elem_f1": float("nan"), "type_f1": float("nan"), "render_ok": False}, Scorer(br)
            recs.append({"cond": c, "id": r["id"], "check_score": ck, "elem_f1": e["elem_f1"], "type_f1": e["type_f1"],
                         "render_ok": e.get("render_ok")})
    br.close()
with open(d / "quality.jsonl", "w", encoding="utf-8") as f:
    for x in recs:
        f.write(json.dumps(x) + "\n")
conds = sorted({x["cond"] for x in recs})
by = {c: {x["id"]: x for x in recs if x["cond"] == c} for c in conds}
K = ["check_score", "elem_f1", "type_f1"]
lines = [f"# Quality ({d})", "", "| cond | n | " + " | ".join(K) + " |", "|---|---|" + "---|" * len(K)]
for c in conds:
    xs = list(by[c].values())
    lines.append(f"| {c} | {len(xs)} | " + " | ".join(f"{np.nanmean([x[k] for x in xs]):.3f}" for k in K) + " |")


def ci(v, B=10000):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    bs = v[np.random.default_rng(0).integers(0, len(v), (B, len(v)))].mean(1)
    return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


lines += ["", "| paired (A - B) | " + " | ".join(K) + " |", "|---|" + "---|" * len(K)]
for a, b in [("ranker", "dom"), ("ranker", "pld"), ("dom", "pld")]:
    if a in by and b in by:
        ids = sorted(set(by[a]) & set(by[b]))
        cells = []
        for k in K:
            m, lo, hi = ci([by[a][i][k] - by[b][i][k] for i in ids])
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        lines.append(f"| {a.upper()} - {b.upper()} (n={len(ids)}) | " + " | ".join(cells) + " |")
(d / "quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
