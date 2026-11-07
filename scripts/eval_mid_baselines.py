"""Score baselines with the query_mid checklist (main) and element/type F1 (secondary). CPU only.

B0_flow   : GT flow HTML (checklist was validated on it -> 1.0 by construction)
B0_abs    : GT abs HTML (independent render of the same GT; not used for validation)
B1_template: one fixed generic screen for every query
B2_rich   : TF-IDF(query_rich) nearest train_pool row -> its GT flow (old B2; query leaks texts)
B2_mid    : TF-IDF(query_mid) nearest train_pool row (by its query_mid) -> its GT flow
Gemma     : saved pilot outputs in runs/gemma/*_pilot.jsonl (generated from query_rich/short; read only)
Usage: .venv/Scripts/python.exe -I scripts/eval_mid_baselines.py --splits pilot,dev,test
Writes runs/baselines_mid/{per_<split>.jsonl, summary.json}
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_query_mid import limit_resources, rss_gb, RSS_LIMIT_GB  # noqa: E402  (also sets sys.path/env)

from genui.checklist import ChecklistScorer  # noqa: E402
from genui.metrics import Scorer  # noqa: E402
from genui.retrieval import Retriever  # noqa: E402
from eval_baselines import B1_HTML  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "runs" / "baselines_mid"


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="pilot,dev,test")
    args = ap.parse_args()
    limit_resources()
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    pool = jl(PROC / "train_pool.jsonl")
    pool_mid = {r["id"]: r for r in jl(PROC / "mid" / "train_pool.jsonl")}
    pool_m = [{"query_mid": pool_mid[r["id"]]["query_mid"], "gt_html_flow": r["gt_html_flow"]}
              for r in pool if pool_mid[r["id"]]["usable"]]
    ret_rich, ret_mid = Retriever(pool, "query_rich"), Retriever(pool_m, "query_mid")
    del pool
    gemma = {}
    for f in sorted(glob.glob(str(ROOT / "runs" / "gemma" / "*_pilot.jsonl"))):
        rows = [r for r in jl(f) if "id" in r and r.get("response")]
        gemma[Path(f).stem.replace("_pilot", "")] = {r["id"]: r["response"] for r in rows}
    summary = json.loads((OUT / "summary.json").read_text()) if (OUT / "summary.json").exists() else {}
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=True, args=["--disable-gpu", "--renderer-process-limit=1"])
        cs, ms = ChecklistScorer(br), Scorer(br)
        for split in args.splits.split(","):
            rows = jl(PROC / f"{split}.jsonl")
            mid = {r["id"]: r for r in jl(PROC / "mid" / f"{split}.jsonl")}
            per = []
            for i, r in enumerate(rows):
                m = mid[r["id"]]
                if not m["usable"]:
                    continue
                g = ms.analyze(r["gt_html_flow"])
                outs = {"B0_flow": r["gt_html_flow"], "B0_abs": r["gt_html_abs"], "B1_template": B1_HTML,
                        "B2_rich": ret_rich.topk(r["query_rich"], 1)[0]["gt_html_flow"],
                        "B2_mid": ret_mid.topk(m["query_mid"], 1)[0]["gt_html_flow"]}
                if split == "pilot":
                    for k, d in gemma.items():
                        if r["id"] in d:
                            outs["G_" + k] = d[r["id"]]
                for name, html in outs.items():
                    text = html if name.startswith("G_") else "```html\n" + html + "\n```"
                    c = cs.score(text, m["checklist"])
                    e = ms.score(text, r, g)
                    per.append({"id": r["id"], "method": name, "check_score": c["check_score"],
                                "check_all": c["check_all"], "check_lenient": c["check_lenient"],
                                "item_ok": c["item_ok"], "elem_f1": e["elem_f1"], "type_f1": e["type_f1"],
                                "text_recall": e["text_recall"], "render_ok": e["render_ok"]})
                if i % 100 == 99 and rss_gb() > RSS_LIMIT_GB:
                    sys.exit("RSS guard")
            with open(OUT / f"per_{split}.jsonl", "w", encoding="utf-8") as f:
                for p in per:
                    f.write(json.dumps(p) + "\n")
            summ = {}
            for name in dict.fromkeys(p["method"] for p in per):
                ss = [p for p in per if p["method"] == name]
                summ[name] = {"n": len(ss), **{k: round(statistics.mean(float(s[k]) for s in ss), 4)
                                               for k in ("check_score", "check_all", "check_lenient",
                                                         "elem_f1", "type_f1", "text_recall")}}
            summary[split] = summ
            print(split, json.dumps(summ, indent=1), file=sys.stderr, flush=True)
        br.close()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
