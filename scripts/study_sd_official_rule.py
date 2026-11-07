"""Official-rule rerun of the formal SD comparison: global tree also contains the request's own response
(SD_OWN_GLOBAL=1), MAX_SPEC from SD_MAX_SPEC (16 or 64).  Same targets / app-level cross-fit as study_sd_formal.py.
Policies: SD f1, SD f16, SD + learned (frequency + DOM) f1 and f16 (retrained under the same rule).

  SD_OWN_GLOBAL=1 SD_MAX_SPEC=16 .venv/Scripts/python.exe -I scripts/study_sd_official_rule.py
Output: runs/study/sd_official_rule_ms{MAX_SPEC}.json
"""
from __future__ import annotations

import os, sys, pathlib, json, time
os.environ.setdefault("SIM_THREADS", "2")
os.environ.setdefault("SD_OWN_GLOBAL", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

import study_draft_sim as S
import study_ranker as R
import study_sd_formal as F
from study_common import OUT, ROOT

MAIN = ("C1", "C0", "C4g", "C7g", "g2pld", "g2dom")


def main():
    S.guard(int(os.environ["SIM_THREADS"]))
    from transformers import AutoTokenizer
    from sklearn.model_selection import GroupKFold
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    cache = {}

    def dec(i):
        if i not in cache:
            cache[i] = tok.decode([i])
        return cache[i]
    raw, targets = R.load_all(tok)
    c6 = [t for t in targets if t["set"] == "cands6"]
    tg = [t for t in targets if t["set"] in MAIN] + c6[::5]
    extra = [t for k, t in enumerate(c6) if k % 5 != 0]
    cxs = [R.Ctx(t, dec) for t in tg]
    apps = np.array([t["app"] for t in tg])
    resp = []
    for cx in cxs:
        yt = cx.y.tolist(); ann, _ = S.annotate([dec(i) for i in yt]); ann.append(None); resp.append((yt, ann))
    ext = []
    for t in extra:
        yt = list(t["y"]); ann, _ = S.annotate([dec(i) for i in yt]); ann.append(None); ext.append((t["app"], yt, ann))
    t0 = time.time()
    res = [dict(set=t["set"], id=t["id"], app=t["app"]) for t in tg]
    for f, (tr, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(tg)), groups=apps)):
        test_apps = set(apps[te])
        toks, anns = [], []
        for i in tr:
            yt, ann = resp[i]; toks += yt + [-1]; anns += ann[:len(yt)] + [None]
        for app, yt, ann in ext:
            if app not in test_apps:
                toks += yt + [-1]; anns += ann[:len(yt)] + [None]
        glob = F.Seq(toks, anns)
        ch = {}
        for fac in (1.0, 16.0):
            rows = []
            for i in tr:
                rows += F.run(cxs[i], glob, fac, "path", collect=True)[2]
            ch[fac] = F.fit(rows, F.FREQ_COLS + F.STRUCT_COLS)
        for i in te:
            for fac in (1.0, 16.0):
                nf, vt, _ = F.run(cxs[i], glob, fac, "path"); res[i][f"sd_f{fac:g}"] = nf; res[i][f"sd_f{fac:g}_vt"] = vt
                nf, vt, _ = F.run(cxs[i], glob, fac, "learned", chooser=ch[fac]); res[i][f"sdl_f{fac:g}"] = nf; res[i][f"sdl_f{fac:g}_vt"] = vt
        print("fold", f, f"{time.time() - t0:.0f}s", flush=True)
    prev = {(r["set"], r["id"], r["app"]): r for r in json.loads((OUT / "sd_formal.json").read_text(encoding="utf-8"))["rows"]}
    out = {"config": {"own_response_in_global": F.OWN_GLOBAL, "MAX_SPEC": F.MAX_SPEC}, "groups": {}, "rows": res}
    for g, sel in (("main(poc+gate2)", lambda r: r["set"] in MAIN), ("poc_greedy", lambda r: r["set"] in MAIN[:4]),
                   ("gate2", lambda r: r["set"] in ("g2pld", "g2dom")), ("cands6(aux)", lambda r: r["set"] == "cands6")):
        rs = [r for r in res if sel(r)]
        d = {"n": len(rs)}
        for k in ("sd_f1", "sdl_f1", "sd_f16", "sdl_f16"):
            d[k] = float(np.mean([r[k] for r in rs]))
            d[k + "_verify_tok_per_fwd"] = float(np.sum([r[k + "_vt"] for r in rs]) / np.sum([r[k] for r in rs]))
        d["std"] = float(np.mean([prev[(r["set"], r["id"], r["app"])]["std"] for r in rs]))
        d["old_rule_sd_f1"] = float(np.mean([prev[(r["set"], r["id"], r["app"])]["sd_f1"] for r in rs]))
        d["old_rule_sd_f16"] = float(np.mean([prev[(r["set"], r["id"], r["app"])]["sd_f16"] for r in rs]))
        d["sd_f1_minus_sdl_f1"] = R.paired([r["sd_f1"] for r in rs], [r["sdl_f1"] for r in rs])
        d["sd_f16_minus_sdl_f16"] = R.paired([r["sd_f16"] for r in rs], [r["sdl_f16"] for r in rs])
        d["oldrule_minus_official_sd_f1"] = R.paired([prev[(r["set"], r["id"], r["app"])]["sd_f1"] for r in rs], [r["sd_f1"] for r in rs])
        d["oldrule_minus_official_sd_f16"] = R.paired([prev[(r["set"], r["id"], r["app"])]["sd_f16"] for r in rs], [r["sd_f16"] for r in rs])
        out["groups"][g] = d
        print(g, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items() if not isinstance(v, dict)}))
        for k, v in d.items():
            if isinstance(v, dict):
                print("   ", k, round(v["mean"], 2), v["median"], [round(x, 2) for x in v["ci95"]])
    (OUT / f"sd_official_rule_ms{F.MAX_SPEC}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
