"""S3: visual completeness VC(t) of streamed prefixes and perceptual latency L_perc = int_0^T (1-VC(t)) dt.

Prefix after k tokens is rendered as-is (browser error recovery), viewport 360x640.
VC_ssim(p) = clip((SSIM(p,y) - SSIM(blank,y)) / (1 - SSIM(blank,y)), 0, 1)   (primary)
VC_pix(p)  = clip((match(p,y) - match(blank,y)) / (1 - match(blank,y)), 0, 1), match = share of identical pixels
t_k = ttft + k/T * (total - ttft) (measured per-sample timing); VC = 0 before the first checkpoint, step-hold after.

  .venv/Scripts/python.exe -I scripts/study_s3.py
Output: runs/study/s3_vc.json
"""
from __future__ import annotations

import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from study_common import CONDS, OUT, Renderer, check_rss, encode, gray, guard_setup, load_runs, ssim
from genui.metrics import extract_html

NCP = 20


def strip_fence(s):
    i = s.find("<")
    return s[i:] if i >= 0 else ""


def main():
    guard_setup()
    pilot, runs = load_runs()
    rend = Renderer()
    res = {}
    try:
        blank = rend.shot("<html><body></body></html>")
        order = ["E2B_nf4_rich_fs"] + [c for c in CONDS if c != "E2B_nf4_rich_fs"]
        for c in order:
            rows = []
            for r in runs[c]:
                resp = r["response"]
                ids, offs = encode(resp)
                T = len(ids)
                final = rend.shot(extract_html(resp))
                gf = gray(final)
                s0 = ssim(gray(blank), gf)
                m0 = float((np.abs(blank.astype(int) - final.astype(int)).max(-1) == 0).mean())
                ttft, tot = r["ttft_s"], r["total_s"]
                cps = []
                for i in range(1, NCP + 1):
                    k = max(1, round(i / NCP * T))
                    pre = strip_fence(resp[:offs[k - 1][1]])
                    if i == NCP:
                        pre = extract_html(resp)
                    img = rend.shot(pre)
                    s = ssim(gray(img), gf)
                    m = float((np.abs(img.astype(int) - final.astype(int)).max(-1) == 0).mean())
                    vc_s = float(np.clip((s - s0) / max(1e-9, 1 - s0), 0, 1))
                    vc_p = float(np.clip((m - m0) / max(1e-9, 1 - m0), 0, 1))
                    cps.append({"k": k, "frac": k / T, "t": ttft + k / T * (tot - ttft), "vc_ssim": vc_s, "vc_pix": vc_p})
                def lperc(key):
                    L, prev_t, prev_v = 0.0, 0.0, 0.0
                    for cp in cps:
                        L += (1 - prev_v) * (cp["t"] - prev_t)
                        prev_t, prev_v = cp["t"], cp[key]
                    return L
                def t_at(key, th):
                    for cp in cps:
                        if cp[key] >= th:
                            return cp["t"]
                    return tot
                rows.append({"id": r["id"], "T_tok": T, "total_s": tot, "ttft_s": ttft,
                             "Lperc_ssim": lperc("vc_ssim"), "Lperc_pix": lperc("vc_pix"),
                             "ratio_ssim": lperc("vc_ssim") / tot, "ratio_pix": lperc("vc_pix") / tot,
                             "t_vc50_ssim": t_at("vc_ssim", 0.5), "t_vc90_ssim": t_at("vc_ssim", 0.9),
                             "checkpoints": cps})
                check_rss()
            agg = {k: float(np.mean([x[k] for x in rows])) for k in ("ratio_ssim", "ratio_pix", "total_s", "Lperc_ssim", "Lperc_pix")}
            agg["median_ratio_ssim"] = float(np.median([x["ratio_ssim"] for x in rows]))
            agg["t_vc90_over_T"] = float(np.mean([x["t_vc90_ssim"] / x["total_s"] for x in rows]))
            agg["t_vc50_over_T"] = float(np.mean([x["t_vc50_ssim"] / x["total_s"] for x in rows]))
            res[c] = {"agg": agg, "samples": rows}
            print(c, {k: round(v, 3) for k, v in agg.items()}, flush=True)
            (OUT / "s3_vc.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    finally:
        rend.close()


if __name__ == "__main__":
    main()
