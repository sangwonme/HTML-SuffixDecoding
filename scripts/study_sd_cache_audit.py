"""C: SuffixDecoding global-cache audit + G: candidate-set oracles (CPU, replay).

C1 leakage audit of the cross-fit cache used in study_sd_formal.py (same fold construction, re-derived here):
   per test request, does the global cache contain any output of the same request id, of the same app, or an
   output token-identical to the target?
C2 serving-order version: requests processed in a random order (3 seeds); the cache holds only outputs of
   requests processed earlier (excluding outputs of the same request id - other conditions of the same request).
   Variant "excl. same app" also reported.
C3 boilerplate-removed cache: tokens covered by 8-grams that occur in >20% of all outputs (template boilerplate,
   e.g. <head>/<style>) are masked out of the global cache; SD rerun.
G  oracles on the same targets: best candidate inside the formal SD candidate set (f16) per step; any position in
   the local context (O2, from study_draft_sim); any position in local context + global cache (O2+global).

  .venv/Scripts/python.exe -I scripts/study_sd_cache_audit.py
Output: runs/study/sd_cache_audit.json
"""
from __future__ import annotations

import os, sys, pathlib, json, time
os.environ.setdefault("SIM_THREADS", "2")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from collections import Counter

import study_draft_sim as S
import study_ranker as R
import study_sd_formal as F
from study_common import OUT, ROOT

MAIN = ("C1", "C0", "C4g", "C7g", "g2pld", "g2dom")


def cand_oracle(cx, glob, factor):
    """per step choose the SD candidate (all m, local/global) with the longest acceptance."""
    P, y = cx.P, cx.y
    T = len(y)
    local = F.Seq(cx.ctx, cx.ann)
    pos, nf = 1, 1
    while pos < T:
        L = P + pos
        tail = cx.ctx[max(0, L - F.DEPTH):L]
        d, cands = F.speculate(local, L, glob, tail, factor, collect=True)
        a = max([F.accept_path(c["d"], y, pos) for c in cands] + [F.accept_path(d or [], y, pos)])
        nf += 1; pos += a + 1
    return nf


def oracle_global(cx, glob):
    """any start position in local ctx[:L] or in the global cache."""
    P, y = cx.P, cx.y
    T = len(y)
    g = glob.t
    pos, nf = 1, 1
    while pos < T:
        L = P + pos
        gam = min(16, 2048 - pos - 1)
        a1, _ = S.oracle_best(cx.ctx, L, y, pos, gam)
        # global
        cand = np.nonzero(g == y[pos])[0]
        a2 = 0
        if len(cand):
            a2 = 1
            alive = cand
            while a2 < gam and pos + a2 < T:
                nxt = alive[alive + a2 < len(g)]
                nxt = nxt[g[nxt + a2] == y[pos + a2]] if len(nxt) else nxt
                if len(nxt) == 0:
                    break
                alive = nxt; a2 += 1
        nf += 1; pos += max(a1, a2) + 1
    return nf


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
    ids = [t["id"] for t in tg]
    mains = [i for i, t in enumerate(tg) if t["set"] in MAIN]
    resp = []
    for cx in cxs:
        yt = cx.y.tolist(); ann, _ = S.annotate([dec(i) for i in yt]); ann.append(None); resp.append((yt, ann))
    ext = []
    for t in extra:
        yt = list(t["y"]); ann, _ = S.annotate([dec(i) for i in yt]); ann.append(None); ext.append((t["app"], t["id"], yt, ann))
    t0 = time.time()
    out = {"cross_fit_audit": {}, "serving_order": {}, "boilerplate": {}, "oracles": {}}
    prev = {(r["set"], r["id"], r["T"]): r for r in json.loads((OUT / "sd_formal.json").read_text(encoding="utf-8"))["rows"]}

    # ---------------- C1 audit (same fold construction as study_sd_formal.py)
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(tg)), groups=apps))
    leak = Counter()
    for f, (tr, te) in enumerate(folds):
        test_apps = set(apps[te])
        cache_ids = [ids[i] for i in tr] + [e[1] for e in ext if e[0] not in test_apps]
        cache_apps = [apps[i] for i in tr] + [e[0] for e in ext if e[0] not in test_apps]
        cache_outs = {tuple(resp[i][0]) for i in tr} | {tuple(e[2]) for e in ext if e[0] not in test_apps}
        for i in te:
            leak["test_requests"] += 1
            leak["same_id_in_cache"] += ids[i] in cache_ids
            leak["same_app_in_cache"] += apps[i] in cache_apps
            leak["identical_output_in_cache"] += tuple(resp[i][0]) in cache_outs
    out["cross_fit_audit"] = dict(leak)
    print("audit", dict(leak), flush=True)

    # ---------------- boilerplate mask (8-grams in >20% of outputs)
    allouts = [r[0] for r in resp] + [e[2] for e in ext]
    df = Counter()
    for yt in allouts:
        df.update({tuple(yt[j:j + 8]) for j in range(len(yt) - 7)})
    common = {g for g, v in df.items() if v > 0.2 * len(allouts)}

    def mask(yt):
        m = np.zeros(len(yt), bool)
        for j in range(len(yt) - 7):
            if tuple(yt[j:j + 8]) in common:
                m[j:j + 8] = True
        return [(-1 if mm else t) for t, mm in zip(yt, m)], float(m.mean())

    masked = {}
    fracs = []
    for i in range(len(tg)):
        masked[i], fr = mask(resp[i][0]); fracs.append(fr)
    ext_masked = [(e[0], mask(e[2])[0], e[3]) for e in ext]
    out["boilerplate"]["masked_token_share_mean"] = float(np.mean(fracs))
    out["boilerplate"]["n_common_8grams"] = len(common)

    # ---------------- C3 + G: cross-fit cache, main targets
    res = {i: {} for i in mains}
    for f, (tr, te) in enumerate(folds):
        test_apps = set(apps[te])
        toks, anns, mtoks = [], [], []
        for i in tr:
            yt, ann = resp[i]; toks += yt + [-1]; anns += ann[:len(yt)] + [None]; mtoks += masked[i] + [-1]
        for (app, _, yt, ann), (_, myt, _) in zip(ext, ext_masked):
            if app not in test_apps:
                toks += yt + [-1]; anns += ann[:len(yt)] + [None]; mtoks += myt + [-1]
        glob = F.Seq(toks, anns)
        mglob = F.Seq(mtoks, anns)
        for i in te:
            if i not in res:
                continue
            p = prev[(tg[i]["set"], tg[i]["id"], len(tg[i]["y"]))]
            r = res[i]
            r.update({"std": p["std"], "sd_f1": p["sd_f1"], "sd_f16": p["sd_f16"], "sd_local_f1": p["sd_localonly_f1"],
                      "oracle_local": p["oracle"], "sdl_f16": p.get("sd16_learned_full")})
            r["sd_f1_masked"] = F.run(cxs[i], mglob, 1.0, "path")[0]
            r["sd_f16_masked"] = F.run(cxs[i], mglob, 16.0, "path")[0]
            r["oracle_sd_cands_f16"] = cand_oracle(cxs[i], glob, 16.0)
            r["oracle_local_plus_global"] = oracle_global(cxs[i], glob)
        print("crossfit fold", f, f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- C2 serving order (main targets only form the stream; cands6 aux outputs excluded)
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        order = list(rng.permutation(mains))
        done = []
        for i in order:
            for variant in ("excl_same_id", "excl_same_app"):
                toks, anns = [], []
                for j in done:
                    if ids[j] == ids[i] or (variant == "excl_same_app" and apps[j] == apps[i]):
                        continue
                    yt, ann = resp[j]; toks += yt + [-1]; anns += ann[:len(yt)] + [None]
                glob = F.Seq(toks, anns) if toks else None
                res[i][f"serve{seed}_{variant}_f16"] = F.run(cxs[i], glob, 16.0, "path")[0]
                res[i][f"serve{seed}_{variant}_f1"] = F.run(cxs[i], glob, 1.0, "path")[0]
            res[i][f"serve{seed}_rank"] = len(done)
            done.append(i)
        print("serving seed", seed, f"{time.time() - t0:.0f}s", flush=True)

    rows = [dict(set=tg[i]["set"], id=ids[i], app=apps[i], **res[i]) for i in mains]
    for i, rr in zip(mains, rows):
        for v in ("excl_same_id", "excl_same_app"):
            for fac in ("f1", "f16"):
                rr[f"serve_{v}_{fac}"] = float(np.mean([res[i][f"serve{s}_{v}_{fac}"] for s in (0, 1, 2)]))
    summ = {"n": len(rows)}
    keys = ["std", "sd_local_f1", "sd_f1", "sd_f16", "sdl_f16", "sd_f1_masked", "sd_f16_masked",
            "serve_excl_same_id_f1", "serve_excl_same_id_f16", "serve_excl_same_app_f1", "serve_excl_same_app_f16",
            "oracle_sd_cands_f16", "oracle_local", "oracle_local_plus_global"]
    for k in keys:
        summ[k] = float(np.mean([r[k] for r in rows]))
    for a, b in (("sd_f16_masked", "sd_f16"), ("sd_f1_masked", "sd_f1"), ("serve_excl_same_id_f16", "sd_f16"),
                 ("serve_excl_same_app_f16", "sd_f16"), ("std", "sd_f16"), ("std", "sd_f16_masked"),
                 ("std", "serve_excl_same_app_f16"), ("sd_f16", "oracle_sd_cands_f16"), ("oracle_local", "oracle_local_plus_global")):
        summ[f"{a}_minus_{b}"] = R.paired([r[a] for r in rows], [r[b] for r in rows])
    # gain retained after boilerplate masking
    g0 = summ["std"] - summ["sd_f16"]; g1 = summ["std"] - summ["sd_f16_masked"]
    summ["sd_gain_retained_after_mask_f16"] = g1 / g0 if g0 else None
    # serving: early vs late requests (cache size effect)
    for i, rr in zip(mains, rows):
        rr["serve_rank_mean"] = float(np.mean([res[i]["serve%d_rank" % sd] for sd in (0, 1, 2)]))
    first = [r for r in rows if r["serve_rank_mean"] < len(rows) / 3]
    last = [r for r in rows if r["serve_rank_mean"] >= 2 * len(rows) / 3]
    summ["serve_first_third_f16_minus_crossfit"] = float(np.mean([r["serve_excl_same_app_f16"] - r["sd_f16"] for r in first])) if first else None
    summ["serve_last_third_f16_minus_crossfit"] = float(np.mean([r["serve_excl_same_app_f16"] - r["sd_f16"] for r in last])) if last else None
    out["summary"] = summ
    out["rows"] = rows
    (OUT / "sd_cache_audit.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summ.items() if not isinstance(v, dict)}))
    for k, v in summ.items():
        if isinstance(v, dict):
            print("  ", k, round(v["mean"], 2), v["median"], [round(x, 2) for x in v["ci95"]])


if __name__ == "__main__":
    main()
