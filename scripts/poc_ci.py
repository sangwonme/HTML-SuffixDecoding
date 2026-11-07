"""Bootstrap CIs (mean, 95%) and paired difference CIs for the pilot-50 accuracy comparison.

Usage: .venv/Scripts/python.exe -I scripts/poc_ci.py E2B L2_P50C1 L2_P50C4 [L2_P50C5 ...]
The first condition is the reference for paired differences. Reads runs/poc/<model>_eval.jsonl.
Writes runs/poc/<model>_ci.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POC = ROOT / "runs" / "poc"
B = 10000
METRICS = ["check_score", "check_all", "elem_f1", "type_f1", "total_s", "tokens_per_fwd"]

model, conds = sys.argv[1], sys.argv[2:]
recs = [json.loads(l) for l in open(POC / f"{model}_eval.jsonl", encoding="utf-8")]
by = {c: {r["id"]: r for r in recs if r["cond"] == c} for c in conds}
ids = sorted(set.intersection(*[set(v) for v in by.values()]))
rng = np.random.default_rng(20261107)
idx = rng.integers(0, len(ids), size=(B, len(ids)))


def ci(x):
    x = np.asarray(x, float)
    bs = x[idx].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


lines = [f"Paired bootstrap (B={B}, n={len(ids)} common ids). Reference = {conds[0]}.", "",
         "| cond | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
for c in conds:
    cells = []
    for m in METRICS:
        mu, lo, hi = ci([float(by[c][i][m]) for i in ids])
        cells.append(f"{mu:.3f} [{lo:.3f}, {hi:.3f}]")
    lines.append(f"| {c} | " + " | ".join(cells) + " |")
lines += ["", "| diff vs " + conds[0] + " | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
for c in conds[1:]:
    cells = []
    for m in METRICS:
        d = [float(by[c][i][m]) - float(by[conds[0]][i][m]) for i in ids]
        mu, lo, hi = ci(d)
        star = " *" if lo > 0 or hi < 0 else ""
        cells.append(f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}")
    lines.append(f"| {c} - {conds[0]} | " + " | ".join(cells) + " |")
sat = {c: float(np.mean([by[c][i]["check_score"] for i in ids])) for c in conds}
lines += ["", "checklist saturated (mean >= 0.98): " + ", ".join(f"{c}={'yes' if v >= 0.98 else 'no'}" for c, v in sat.items()),
          "* = 95% CI excludes 0"]
out = "\n".join(lines) + "\n"
(POC / f"{model}_ci.md").write_text(out, encoding="utf-8")
print(out)
