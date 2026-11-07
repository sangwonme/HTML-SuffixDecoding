"""P1: where does the time go?  decode share of wall-clock, and seconds spent per lexical-context group
(token share from S1 x decode time; assumes uniform per-token decode cost).

  .venv/Scripts/python.exe -I scripts/study_p1.py      (needs runs/study/s1_predictability.json)
"""
from __future__ import annotations

import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from study_common import CONDS, OUT, encode, load_runs


def main():
    pilot, runs = load_runs()
    s1 = json.loads((OUT / "s1_predictability.json").read_text(encoding="utf-8"))["conds"]
    res = {}
    for c in CONDS:
        rs = runs[c]
        ttft = np.array([r["ttft_s"] for r in rs]); tot = np.array([r["total_s"] for r in rs])
        dec = tot - ttft
        d = {"ttft_mean": float(ttft.mean()), "total_mean": float(tot.mean()),
             "decode_share_of_total": float(dec.sum() / tot.sum()),
             "out_tok_mean": float(np.mean([r["output_tokens"] for r in rs])),
             "decode_tps_median": float(np.median([r["decode_tok_per_s"] for r in rs]))}
        groups = s1[c]["by_context"]
        d["decode_s_by_context"] = {g: v["tok_share"] * float(dec.mean()) for g, v in groups.items()}
        d["S_null_s"] = s1[c]["S_null_share"] * float(dec.mean())
        d["S_tol_s"] = s1[c]["S_tol_share"] * float(dec.mean())
        d["text_share"] = groups.get("text", {}).get("tok_share", 0.0)
        # where does the first visible content start? token index of '<body' (or first body-level tag) / T
        pre = []
        for r in rs:
            ids, offs = encode(r["response"])
            low = r["response"].lower()
            i = low.find("<body")
            if i < 0:
                i = low.find("</head>")
            if i < 0:
                i = low.rfind("</style>")
            pre.append(sum(1 for a, b in offs if a < i) / len(ids) if i >= 0 else 0.0)
        d["pre_body_token_share"] = float(np.mean(pre))
        res[c] = d
        print(c, f"ttft={d['ttft_mean']:.2f}s total={d['total_mean']:.1f}s decode={d['decode_share_of_total']:.3f}",
              f"preBody={d['pre_body_token_share']:.3f} Snull={d['S_null_s']:.1f}s Stol={d['S_tol_s']:.1f}s text_tok={d['text_share']:.3f}")
    (OUT / "p1_time.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
