"""P1 condition B (draft source perturbed, prompt original) -- ESTIMATED wall-clock, no GPU.

Forward counts: study simulator (runs/study/p1_perturb_report.json simB, target = gate2 PLD output; greedy replay is
exact for exact verification: 300/300 and 450/450 in gates 2/2b).
Wall-clock ESTIMATE per request and policy:
  T = TTFT_meas + (F_sim - 1) * t_fwd_meas + (F_sim - 1) * c_draft[policy] + init[policy]
  t_fwd_meas  = per-request mean verify time per forward, gate2b PLD (rep mean)
  c_draft     = measured draft cost per step in gate2b (PLD / DOM / RANKER), init = measured per-request init.
Validation: the same formula on the 'orig' source vs gate2b measured wall-clock of each policy.
Usage: .venv/Scripts/python.exe -I scripts/p1B_estimate.py   -> runs/poc/p1B/p1B_estimate.md
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G2B = ROOT / "runs" / "poc" / "gate2b"
OUT = ROOT / "runs" / "poc" / "p1B"
SIM = json.load(open(ROOT / "runs" / "study" / "p1_perturb_report.json", encoding="utf-8"))["simB"]["per_request"]
POL = {"std": "pld", "rule": "dom", "mlp": "ranker"}


def load(c):
    return [r for r in (json.loads(l) for l in open(G2B / f"E2B_{c}.jsonl", encoding="utf-8")) if "id" in r]


meas = {c: load(c) for c in POL.values()}
per = defaultdict(lambda: defaultdict(list))
for c, rows in meas.items():
    for r in rows:
        per[r["id"]][c].append(r)
cost = {}
for c, rows in meas.items():
    steps = sum(r["draft_steps"] for r in rows)
    cost[c] = (sum(r["draft_s"] - r["draft_init_s"] for r in rows) / steps,
               statistics.mean(r["draft_init_s"] for r in rows))
tf = {i: statistics.mean(r["verify_s"] / max(1, r["n_fwd"] - 1) for r in per[i]["pld"]) for i in per}
ttft = {i: statistics.mean(r["ttft_s"] for r in per[i]["pld"]) for i in per}


def est(i, F, c):
    return ttft[i] + (F - 1) * tf[i] + (F - 1) * cost[c][0] + cost[c][1]


def ci(x, B=10000):
    x = np.asarray(x, float)
    bs = x[np.random.default_rng(0).integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


lines = ["# P1 condition B: ESTIMATED wall-clock (simulated forwards x measured per-forward and draft cost)", "",
         f"draft cost per step (gate2b measured): " + ", ".join(f"{c} {1000 * v[0]:.3f} ms (+init {1000 * v[1]:.1f} ms)" for c, v in cost.items()), ""]
# validation on orig
ids = sorted(SIM["orig"])
lines.append("Validation on original source (estimate vs gate2b measured mean wall-clock):")
for s, c in POL.items():
    e = [est(i, SIM["orig"][i][s], c) for i in ids]
    m = [statistics.mean(r["total_s"] for r in per[i][c]) for i in ids]
    lines.append(f"- {c.upper()}: estimate {statistics.mean(e):.3f} s vs measured {statistics.mean(m):.3f} s "
                 f"(ratio {statistics.mean(e) / statistics.mean(m):.3f}; note: sim target = PLD output for all policies)")
lines += ["", "| source | PLD est. (s) | DOM est. (s) | RANKER est. (s) | DOM - RANKER s [95% CI] | % | PLD - RANKER % [95% CI] | PLD - DOM % [95% CI] |",
          "|---|---|---|---|---|---|---|---|"]
for t in ["orig", "fmt", "css", "dom", "cls"]:
    sim = SIM[t]
    ids = sorted(sim)
    E = {c: [est(i, sim[i][s], c) for i in ids] for s, c in POL.items()}
    d_s = ci([a - b for a, b in zip(E["dom"], E["ranker"])])
    d_p = ci([(a - b) / a for a, b in zip(E["dom"], E["ranker"])])
    p_r = ci([(a - b) / a for a, b in zip(E["pld"], E["ranker"])])
    p_d = ci([(a - b) / a for a, b in zip(E["pld"], E["dom"])])
    lines.append(f"| {t} | {statistics.mean(E['pld']):.2f} | {statistics.mean(E['dom']):.2f} | {statistics.mean(E['ranker']):.2f} | "
                 f"{d_s[0]:+.3f} [{d_s[1]:+.3f}, {d_s[2]:+.3f}] | {100 * d_p[0]:+.2f}% | "
                 f"{100 * p_r[0]:+.2f}% [{100 * p_r[1]:+.2f}, {100 * p_r[2]:+.2f}] | {100 * p_d[0]:+.2f}% [{100 * p_d[1]:+.2f}, {100 * p_d[2]:+.2f}] |")
lines.append("\nAll condition-B wall-clock numbers are ESTIMATES (no GPU run).")
OUT.mkdir(parents=True, exist_ok=True)
out = "\n".join(lines) + "\n"
(OUT / "p1B_estimate.md").write_text(out, encoding="utf-8")
print(out)
