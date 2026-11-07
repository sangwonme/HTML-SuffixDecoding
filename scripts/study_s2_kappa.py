"""S2' (reduced): eta_L1 = 1-|kappa_L1(y)|/|y|, eta_L2 = 1-|kappa(y)|/|y| (tokens, y = extracted HTML).
Also token share of the four kappa components, and (optional --verify) a render check that kappa_L1(y) is L1-equal
and kappa(y) is L2-equal to y.

  .venv/Scripts/python.exe -I scripts/study_s2_kappa.py [--verify]
Output: runs/study/s2_kappa.json
"""
from __future__ import annotations

import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from study_common import CONDS, OUT, guard_setup, load_runs, ntok
from study_canon import kappa
from genui.metrics import extract_html


def main():
    guard_setup()
    verify = "--verify" in sys.argv
    pilot, runs = load_runs()
    chk = None
    if verify:
        from study_equiv import Checker
        chk = Checker()
    res = {}
    try:
        for c in CONDS:
            rows = []
            for r in runs[c]:
                y = extract_html(r["response"])
                n0, n1, n2 = ntok(y), ntok(kappa(y, 1)), ntok(kappa(y, 2))
                row = {"id": r["id"], "n": n0, "n_k1": n1, "n_k2": n2, "eta_L1": 1 - n1 / n0, "eta_L2": 1 - n2 / n0}
                if chk:
                    ref = chk.measure(y)
                    l1 = chk.levels(ref, kappa(y, 1))
                    l2 = chk.levels(ref, kappa(y, 2))
                    row.update({"k1_L1": l1["L1"], "k1_L2": l1["L2"], "k1_D4": l1["D4"],
                                "k2_L2": l2["L2"], "k2_L3d2": l2["L3_d2"], "k2_D4": l2["D4"]})
                rows.append(row)
            agg = {k: float(np.mean([x[k] for x in rows])) for k in ("eta_L1", "eta_L2")}
            if chk:
                for k in ("k1_L1", "k1_L2", "k2_L2", "k2_L3d2"):
                    agg[k + "_rate"] = float(np.mean([x[k] for x in rows]))
                agg["k2_D4_max"] = float(max(x["k2_D4"] for x in rows))
            res[c] = {"agg": agg, "rows": rows}
            print(c, {k: round(v, 3) for k, v in agg.items()}, flush=True)
    finally:
        if chk:
            chk.r.close()
    (OUT / "s2_kappa.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
