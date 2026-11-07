"""Score non-LLM baselines on a split with the same metrics as the LLM runs.

B0 oracle   : GT flow HTML itself (upper bound / sanity)
B1 template : one fixed generic screen (toolbar + list) for every query (lower bound)
B2 retrieval: TF-IDF(query_rich) nearest train_pool row -> its GT flow HTML
Usage: .venv/Scripts/python.exe -I scripts/eval_baselines.py --split pilot
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "ms-playwright"))

from playwright.sync_api import sync_playwright  # noqa: E402

from genui.metrics import Scorer  # noqa: E402
from genui.retrieval import Retriever  # noqa: E402

PROC = ROOT / "data" / "processed"
B1_HTML = ("<!DOCTYPE html><html><body style='margin:0;width:360px;font-family:sans-serif'>"
           "<header style='background:#eee;padding:8px'><span class='icon'>&#9776;</span> <p>Title</p></header>"
           "<ul><li><img width='40' height='40'><p>Item 1</p></li><li><img width='40' height='40'><p>Item 2</p></li>"
           "<li><img width='40' height='40'><p>Item 3</p></li></ul><button>OK</button></body></html>")


def load(name):
    with open(PROC / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pilot")
    args = ap.parse_args()
    rows = load(args.split)
    ret = Retriever(load("train_pool"))
    out_dir = ROOT / "runs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=True)
        sc = Scorer(br)
        per = {"B0_oracle": [], "B1_template": [], "B2_retrieval": []}
        for r in rows:
            g = sc.analyze(r["gt_html_flow"])
            nn = ret.topk(r["query_rich"], 1)[0]
            for name, html in [("B0_oracle", r["gt_html_flow"]), ("B1_template", B1_HTML),
                               ("B2_retrieval", nn["gt_html_flow"])]:
                s = sc.score("```html\n" + html + "\n```", r, g)
                s["id"] = r["id"]
                per[name].append(s)
        br.close()
    for name, ss in per.items():
        summary[name] = {k: round(statistics.mean(float(s[k]) for s in ss), 4)
                         for k in ("render_ok", "elem_f1", "type_f1", "text_recall")}
        with open(out_dir / f"{name}_{args.split}.jsonl", "w", encoding="utf-8") as f:
            for s in ss:
                f.write(json.dumps(s) + "\n")
    (out_dir / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
