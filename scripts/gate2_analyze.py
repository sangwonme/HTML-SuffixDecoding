"""Gate 2 / 2b analysis: draft policies on real GPU wall-clock (exact verification).

Per request (id): mean over reps, then paired differences with bootstrap 95% CI (B=10000) for every pair among
the conditions present (pld, dom, ranker): wall-clock total_s and target forwards n_fwd (+ relative %).
Also per-condition draft cost (per step, init), verify time, tok/fwd, peak GPU / RSS, rep identity, rep CV.
Verdicts: Gate 2 = PLD - DOM wall-clock CI > 0; Gate 2b = DOM - RANKER wall-clock CI > 0.
Usage: .venv/Scripts/python.exe -I scripts/gate2_analyze.py DIR [DIR ...]   -> DIR/gate2_summary.md
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

B = 10000
CONDS = ["pld", "dom", "ranker"]
PAIRS = [("dom", "ranker"), ("pld", "ranker"), ("pld", "dom")]


def load(d, c):
    f = Path(d) / f"E2B_{c}.jsonl"
    if not f.exists():
        return []
    return [r for r in (json.loads(l) for l in open(f, encoding="utf-8")) if "id" in r]


def ci(x, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


for d in sys.argv[1:]:
    data = {c: load(d, c) for c in CONDS}
    conds = [c for c in CONDS if data[c]]
    per = defaultdict(lambda: defaultdict(list))
    for c in conds:
        for r in data[c]:
            per[r["id"]][c].append(r)
    ids = sorted(i for i, v in per.items() if all(v[c] for c in conds))

    def m(i, c, k):
        return statistics.mean(r[k] for r in per[i][c])

    reps = max(len(per[i][c]) for i in ids for c in conds)
    lines = [f"# Draft-policy wall-clock comparison ({d})", "",
             f"requests n={len(ids)}, conditions {conds}, reps {reps}; exact verification; order shuffled per sample/rep.", "",
             "| metric (mean over requests) | " + " | ".join(c.upper() for c in conds) + " |", "|---|" + "---|" * len(conds)]
    for k, lab in [("total_s", "wall-clock total (s)"), ("n_fwd", "target forwards"), ("tokens_per_fwd", "tokens / forward"),
                   ("verify_s", "verify time (s)"), ("draft_s", "draft time incl. init (s)"), ("output_tokens", "output tokens"),
                   ("ttft_s", "TTFT (s)")]:
        lines.append(f"| {lab} | " + " | ".join(f"{statistics.mean(m(i, c, k) for i in ids):.3f}" for c in conds) + " |")
    lines += ["", "| paired difference (A - B) | wall-clock s [95% CI] | wall-clock % of A [95% CI] | forwards [95% CI] | forwards % of A [95% CI] |",
              "|---|---|---|---|---|"]
    verdict = {}
    for a, b in PAIRS:
        if a not in conds or b not in conds:
            continue
        cells = []
        for k in ("total_s", "n_fwd"):
            diff = [m(i, a, k) - m(i, b, k) for i in ids]
            rel = [(m(i, a, k) - m(i, b, k)) / m(i, a, k) for i in ids]
            mu, lo, hi = ci(diff)
            r_mu, r_lo, r_hi = ci(rel)
            cells.append(f"{mu:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            cells.append(f"{100 * r_mu:+.2f}% [{100 * r_lo:+.2f}, {100 * r_hi:+.2f}]")
            if k == "total_s":
                verdict[(a, b)] = (mu, lo, hi)
        lines.append(f"| {a.upper()} - {b.upper()} | " + " | ".join(cells) + " |")
    lines.append("")
    for c in conds:
        rows = data[c]
        steps = sum(r["draft_steps"] for r in rows)
        dstep = sum(r["draft_s"] - r["draft_init_s"] for r in rows)
        rep_same = sum(all(r["gen_ids"] == per[i][c][0]["gen_ids"] for r in per[i][c]) for i in ids)
        cv = [statistics.pstdev([r["total_s"] for r in per[i][c]]) / statistics.mean([r["total_s"] for r in per[i][c]])
              for i in ids if len(per[i][c]) > 1]
        lines.append(f"- {c.upper()}: draft {1000 * dstep / max(1, steps):.3f} ms/step (steps {steps}), init "
                     f"{1000 * statistics.mean(r['draft_init_s'] for r in rows):.1f} ms/request, draft share "
                     f"{sum(r['draft_s'] for r in rows) / sum(r['total_s'] for r in rows):.2%}; peak GPU "
                     f"{max(r['peak_alloc_mib'] for r in rows)} MiB, peak RSS {max(r['peak_rss_mib'] for r in rows)} MiB; "
                     f"outputs identical across reps {rep_same}/{len(ids)}; wall-clock CV median {(100 * statistics.median(cv)) if cv else float('nan'):.2f}%")
    for a, b in [("pld", "dom"), ("pld", "ranker"), ("dom", "ranker")]:
        if a in conds and b in conds:
            same = sum(per[i][a][0]["gen_ids"] == per[i][b][0]["gen_ids"] for i in ids)
            lines.append(f"- outputs identical {a.upper()} vs {b.upper()} (rep 0): {same}/{len(ids)}")
    lines.append("")
    if ("pld", "dom") in verdict:
        mu, lo, hi = verdict[("pld", "dom")]
        lines.append(f"**Gate 2 {'PASSED' if lo > 0 else 'NOT passed'}** (PLD - DOM wall-clock {mu:+.3f} s [{lo:+.3f}, {hi:+.3f}])")
    if ("dom", "ranker") in verdict:
        mu, lo, hi = verdict[("dom", "ranker")]
        lines.append(f"**Gate 2b {'PASSED' if lo > 0 else 'NOT passed'}** (DOM - RANKER wall-clock {mu:+.3f} s [{lo:+.3f}, {hi:+.3f}])")
    out = "\n".join(lines) + "\n"
    (Path(d) / "gate2_summary.md").write_text(out, encoding="utf-8")
    print(out)
