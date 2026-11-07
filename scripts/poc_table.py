"""One-page C0..C5 table from runs/poc/E2B_eval.jsonl (run poc_eval.py first).

Rows (E2B, pilot 10, query_mid):
  C0 zero-shot greedy | C1 TF-IDF greedy | C2 TF-IDF PLD | C3 TF-IDF RNSD-L2
  C4g struct greedy   | C4 struct RNSD-L2 | C5 oracle RNSD-L2
Columns: checklist, element F1, type F1 | total p50, forwards, tok/forward, speed-up vs C1 (per-sample ratio, p50)
plus render identity of RNSD-L2 rows vs their own greedy reference (L1 DOM / L2 style / D).
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POC = ROOT / "runs" / "poc"
model = sys.argv[1] if len(sys.argv) > 1 else "E2B"
recs = [json.loads(l) for l in open(POC / f"{model}_eval.jsonl", encoding="utf-8")]
by = {}
for r in recs:
    by.setdefault(r["cond"], {})[r["id"]] = r

ROWS = [("C0", "zero-shot", "greedy", "greedy_C0"), ("C1", "TF-IDF", "greedy", "greedy"),
        ("C2", "TF-IDF", "PLD", "pld"), ("C3-naive", "TF-IDF", "RNSD-L2 v1 (naive rename)", "L2"),
        ("C3", "TF-IDF", "RNSD-L2 v2 (binding-aware)", "L2_P50C1"),
        ("C4g", "struct", "greedy", "greedy_C4"), ("C4", "struct", "RNSD-L2", "L2_C4"),
        ("C5", "oracle", "RNSD-L2", "L2_C5"),
        ("C6", "mix (struct top-1 + TF-IDF top-2)", "RNSD-L2", "L2_P50C6"),
        ("C7g", "TF-IDF, interleaved-CSS exemplars", "greedy", "greedy_C7"),
        ("C7", "TF-IDF, interleaved-CSS exemplars", "RNSD-L2", "L2_C7")]


def prebody(cond):
    """Per id: fraction of output characters before the first '<body' (CSS-first vs interleaved)."""
    f = POC / f"{model}_{cond}.jsonl"
    out = {}
    if f.exists():
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if "id" in r:
                t = r["response"]
                i = t.lower().find("<body")
                out[r["id"]] = (i / len(t)) if i >= 0 and t else float("nan")
    return out


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


c1 = by.get("greedy", {})
lines = ["| row | few-shot | decoding | n | checklist | elem F1 | type F1 | total p50 (s) | forwards | tok/fwd | "
         "speed-up vs C1 (p50) | render = own greedy (L1 DOM / L2 style) | D p50 | pre-<body> char frac |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, fs, dec, cond in ROWS:
    rs = [r for r in by.get(cond, {}).values() if r["id"] in c1]  # same 10 pilot ids as C1
    if not rs:
        lines.append(f"| {name} | {fs} | {dec} | 0 | (not run) |||||||||")
        continue
    sp = [c1[r["id"]]["total_s"] / r["total_s"] for r in rs if r["id"] in c1]
    ident = "-"
    dp = "-"
    if not cond.startswith("greedy") and cond != "greedy_C0":
        l1 = [r.get("L1_dom_eq") for r in rs if "L1_dom_eq" in r]
        l2 = [r.get("L2_style_eq") for r in rs if "L2_style_eq" in r]
        if l1:
            ident = f"{sum(l1)}/{len(l1)} / {sum(l2)}/{len(l2)}"
        ds = [r["D"] for r in rs if "D" in r]
        dp = f"{med(ds):.3g}" if ds else "-"
    lines.append(
        f"| {name} | {fs} | {dec} | {len(rs)} | {mean([r['check_score'] for r in rs]):.3f} | "
        f"{mean([r['elem_f1'] for r in rs]):.3f} | {mean([r.get('type_f1', float('nan')) for r in rs]):.3f} | "
        f"{med([r['total_s'] for r in rs]):.1f} | {mean([r['n_fwd'] for r in rs]):.0f} | "
        f"{mean([r['tokens_per_fwd'] for r in rs]):.2f} | {med(sp):.2f} | {ident} | {dp} | "
        f"{mean([v for k, v in prebody(cond).items() if k in c1 and v == v]):.3f} |")
out = "\n".join(lines) + "\n"
(POC / f"{model}_Ctable.md").write_text(out, encoding="utf-8")
print(out)
