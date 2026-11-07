"""F (WebSight eval 50) output check (CPU, headless; network blocked so the Tailwind CDN is NOT loaded):
render success, #normalized elements of the output vs the reference HTML, element/type F1 and key-text recall vs
the reference (data/processed/external/websight.jsonl). No checklist exists for WebSight.
Usage: .venv/Scripts/python.exe -I scripts/websight_eval.py runs/poc/F_websight
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

from genui.metrics import Scorer  # noqa: E402

d = Path(sys.argv[1])
ref = {}
for l in open(ROOT / "data/processed/external/websight.jsonl", encoding="utf-8"):
    r = json.loads(l)
    ref[r["id"]] = {"gt_html_flow": r["html"], "key_texts": r.get("key_texts") or []}
recs = []
with sync_playwright() as pw:
    br = pw.chromium.launch(headless=True)
    sc = Scorer(br)
    gcache = {}
    for f in sorted(d.glob("E2B_*.jsonl")):
        c = f.stem.split("_", 1)[1]
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if "id" not in r or r["id"] not in ref:
                continue
            try:
                if r["id"] not in gcache:
                    gcache[r["id"]] = sc.analyze(ref[r["id"]]["gt_html_flow"])
                e = sc.score(r["response"], ref[r["id"]], gcache[r["id"]])
                n_ref = len(gcache[r["id"]]["cats"])
            except Exception:  # noqa: BLE001
                e, n_ref, sc = {"render_ok": False, "n_pred_elems": 0, "elem_f1": float("nan"), "type_f1": float("nan"),
                                "text_recall": float("nan")}, float("nan"), Scorer(br)
            recs.append({"cond": c, "id": r["id"], "render_ok": bool(e["render_ok"]), "n_elems": e.get("n_pred_elems", 0),
                         "n_ref_elems": n_ref, "elem_f1": e["elem_f1"], "type_f1": e["type_f1"],
                         "text_recall": e.get("text_recall", float("nan")), "output_tokens": r["output_tokens"]})
    br.close()
with open(d / "websight_quality.jsonl", "w", encoding="utf-8") as f:
    for x in recs:
        f.write(json.dumps(x) + "\n")
K = ["render_ok", "n_elems", "n_ref_elems", "elem_f1", "type_f1", "text_recall", "output_tokens"]
lines = [f"# WebSight eval-50 output check ({d})", "", "| cond | n | " + " | ".join(K) + " |", "|---|---|" + "---|" * len(K)]
for c in sorted({x["cond"] for x in recs}):
    xs = [x for x in recs if x["cond"] == c]
    lines.append(f"| {c} | {len(xs)} | " + " | ".join(f"{np.nanmean([float(x[k]) for x in xs]):.3f}" for k in K) + " |")
(d / "websight_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
