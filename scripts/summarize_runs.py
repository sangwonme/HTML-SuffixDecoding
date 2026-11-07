"""Score Gemma run outputs (headless Playwright) and build the summary table.

Usage: .venv/Scripts/python.exe -I scripts/summarize_runs.py [runs/gemma/*.jsonl ...]
Writes runs/gemma/<run>.scores.jsonl and runs/gemma/summary.json / summary.md
"""
from __future__ import annotations

import glob
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

RUNS = ROOT / "runs" / "gemma"


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


SHARED_SM_PCT = 10


def shared(r):
    """Recomputed: another process at >= SHARED_SM_PCT % SM right before the sample."""
    return any(str(o.get("sm", "")).isdigit() and int(o["sm"]) >= SHARED_SM_PCT
               for o in (r.get("gpu_before") or {}).get("other_procs_sm", []))


def baseline_rows(ids):
    out = []
    for name in ("B1_template", "B2_retrieval"):
        fp = ROOT / "runs" / "baselines" / f"{name}_pilot.jsonl"
        if not fp.exists():
            continue
        ss = [json.loads(l) for l in open(fp, encoding="utf-8")]
        ss = [x for x in ss if x["id"] in ids]
        out.append({"model": name, "cond": "-", "n": len(ss),
                    "render_ok": mean([float(x["render_ok"]) for x in ss]),
                    "elem_f1": mean([x["elem_f1"] for x in ss]), "type_f1": mean([x["type_f1"] for x in ss]),
                    "text_recall": mean([x["text_recall"] for x in ss])})
    return out


def main():
    files = sys.argv[1:] or sorted(glob.glob(str(RUNS / "*.jsonl")))
    files = [f for f in files if not f.endswith(".scores.jsonl") and "_timing" not in f]
    gt = {}
    for split in ("pilot", "dev", "test"):
        with open(ROOT / "data" / "processed" / f"{split}.jsonl", encoding="utf-8") as f:
            for l in f:
                r = json.loads(l)
                gt[r["id"]] = r
    table = []
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=True)
        sc = Scorer(br)
        gcache = {}
        for fp in files:
            lines = [json.loads(l) for l in open(fp, encoding="utf-8")]
            meta = lines[0].get("_meta", {})
            rows = [l for l in lines if "id" in l]
            if not rows:
                continue
            scores = []
            for r in rows:
                g = gt[r["id"]]
                if r["id"] not in gcache:
                    gcache[r["id"]] = sc.analyze(g["gt_html_flow"])
                s = sc.score(r["response"], g, gcache[r["id"]])
                s["id"] = r["id"]
                scores.append(s)
            with open(fp.replace(".jsonl", ".scores.jsonl"), "w", encoding="utf-8") as f:
                for s in scores:
                    f.write(json.dumps(s) + "\n")
            ttft = [r["ttft_s"] for r in rows]
            tot = [r["total_s"] for r in rows]
            table.append({
                "run": meta.get("run_id", Path(fp).stem), "model": rows[0]["model"], "cond": rows[0]["cond"],
                "n": len(rows), "quant": meta.get("quant"),
                "in_tok_mean": mean([r["prompt_tokens"] for r in rows]),
                "out_tok_mean": mean([r["output_tokens"] for r in rows]),
                "out_tok_p95": pct([r["output_tokens"] for r in rows], 95),
                "ttft_p50": pct(ttft, 50), "ttft_p95": pct(ttft, 95),
                "total_p50": pct(tot, 50), "total_p95": pct(tot, 95), "total_mean": mean(tot),
                "decode_tps_p50": pct([r["decode_tok_per_s"] for r in rows], 50),
                "hit_max_rate": mean([float(r["hit_max_new"]) for r in rows]),
                "peak_reserved_mib_max": max(r["peak_reserved_mib"] for r in rows),
                "weights_mib": meta.get("weights_mib"),
                "gpu_shared_rate": mean([float(shared(r)) for r in rows]),
                "render_ok": mean([float(s["render_ok"]) for s in scores]),
                "elem_f1": mean([s["elem_f1"] for s in scores]),
                "type_f1": mean([s["type_f1"] for s in scores]),
                "text_recall": mean([s["text_recall"] for s in scores]),
            })
        br.close()
    ids = set()
    for fp in files:
        ids |= {json.loads(l).get("id") for l in open(fp, encoding="utf-8")}
    order = {"E2B": 0, "E4B": 1, "12B": 2}
    corder = {"short_zs": 0, "rich_zs": 1, "rich_fs": 2}
    table.sort(key=lambda t: (corder.get(t["cond"], 9), order.get(t["model"], 9)))
    table += baseline_rows(ids)
    (RUNS / "summary.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    cols = ["model", "cond", "n", "in_tok_mean", "out_tok_mean", "ttft_p50", "ttft_p95", "total_p50", "total_p95",
            "decode_tps_p50", "peak_reserved_mib_max", "hit_max_rate", "gpu_shared_rate", "render_ok", "elem_f1",
            "type_f1", "text_recall"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for t in table:
        md.append("| " + " | ".join(f"{t[c]:.3g}" if isinstance(t.get(c), float) else str(t.get(c, "")) for c in cols) + " |")
    notes = [
        "",
        "Notes: pilot first 10 rows (same ids for every row). n=10, so p95 is effectively the max.",
        "Gemma 4 QAT checkpoints loaded as bitsandbytes NF4 (approximation of q4_0), greedy, max_new_tokens 2048,",
        "concurrency 1, 3 warm-up runs (64 tokens). rich_fs = query_rich + 3 TF-IDF-retrieved train examples.",
        "gpu_shared = another process at >=10% SM before the sample. Baselines B1/B2 scored on the same 10 ids.",
    ]
    (RUNS / "summary.md").write_text("\n".join(md + notes) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
