"""Learnability preview: rank candidate draft positions with a small model (sample-level 5-fold CV).

Candidates at each step: positions c in ctx[:L] whose previous token equals the last context token
(1-gram hits, most recent 400).  Features: backward suffix-match length (<=16), structural annotation agreement
(lexical state, attr/CSS-property key, tag-path suffix, depth, sibling index, last event), own-output flag,
log distance.  Target: accepted length from c.  Model: HistGradientBoostingRegressor.  Policy: argmax prediction
(ties -> most recent).  Evaluated by greedy replay on held-out samples (poc greedy C1/C0/C4g/C7g, n=40).

  .venv/Scripts/python.exe -I scripts/study_draft_learn.py
Output: runs/study/draft_learn.json
"""
from __future__ import annotations

import os, sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

import study_draft_sim as S
from study_common import OUT, ROOT

GAMMA, MAXC = 16, 400


def feats(ctx, L, c, ann, reg_ctx, P):
    bm = 0
    while bm < 16 and c - 1 - bm >= 0 and ctx[c - 1 - bm] == ctx[L - 1 - bm]:
        bm += 1
    a, b = ann[c], ann[L]
    if a is None or b is None:
        st = [0, 0, 0, 0, 0, 0, 0]
    else:
        pa, pb = a[2], b[2]
        k = 0
        while k < min(len(pa), len(pb)) and pa[len(pa) - 1 - k] == pb[len(pb) - 1 - k]:
            k += 1
        st = [1, int(a[0] == b[0]), int(a[1] == b[1] and a[1] != ""), k, int(a[3] == b[3]), int(a[4] == b[4]), int(a[5] == b[5])]
    return [bm] + st + [int(c >= P), np.log1p(L - c)]


def cands(ctx, L):
    h = np.nonzero(ctx[:L - 1] == ctx[L - 1])[0] + 1   # c with ctx[c-1] == last token, c <= L-1
    return h[-MAXC:]


def prep(t, tok):
    prompt = np.array(t["prompt"], np.int64)
    y = np.array(t["y"], np.int64)
    strs = [tok.decode([i]) for i in list(prompt) + list(y)]
    ann, reg = S.annotate(strs)
    ann.append(None)
    return prompt, y, ann, reg


def collect(prompt, y, ann, reg, rng, per_step=60):
    """Training rows along the std trajectory."""
    X, Y = [], []
    _, steps = S.run_policy("std", prompt, y, ann, reg, record=True)
    ctx = np.concatenate([prompt, y])
    P = len(prompt)
    for s in steps:
        L = P + s["pos"]
        g = min(GAMMA, 2048 - s["pos"] - 1)
        cs = cands(ctx, L)
        if len(cs) > per_step:
            cs = np.concatenate([cs[-per_step // 2:], rng.choice(cs[:-per_step // 2], per_step // 2, replace=False)])
        for c in cs:
            X.append(feats(ctx, L, int(c), ann, reg, P))
            Y.append(S.accept_len(ctx, int(c), L, y, s["pos"], g))
    return X, Y


def replay_learned(model, prompt, y, ann, reg):
    ctx = np.concatenate([prompt, y])
    P, T = len(prompt), len(y)
    pos, nf = 1, 1
    while pos < T:
        L = P + pos
        g = min(GAMMA, 2048 - pos - 1)
        cs = cands(ctx, L)
        a = 0
        if len(cs):
            F = np.array([feats(ctx, L, int(c), ann, reg, P) for c in cs])
            pr = model.predict(F)
            best = int(cs[np.flatnonzero(pr == pr.max())[-1]])
            a = S.accept_len(ctx, best, L, y, pos, g)
        nf += 1
        pos += a + 1
    return nf


def main():
    S.guard(int(os.environ.get("SIM_THREADS", "4")))
    from transformers import AutoTokenizer
    from sklearn.ensemble import HistGradientBoostingRegressor
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    sim = json.loads((OUT / "draft_sim.json").read_text(encoding="utf-8"))
    base = {(r["set"], r["id"]): r for r in sim["rows"]}
    targets = [t for t in S.load_targets(tok) if t["set"] in ("C1", "C0", "C4g", "C7g")]
    data = [prep(t, tok) for t in targets]
    rng = np.random.default_rng(0)
    rows = [collect(*d, rng) for d in data]
    folds = np.arange(len(targets)) % 5
    res = []
    t0 = time.time()
    for f in range(5):
        tr = [i for i in range(len(targets)) if folds[i] != f]
        X = np.array([x for i in tr for x in rows[i][0]], float)
        Y = np.array([y for i in tr for y in rows[i][1]], float)
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, random_state=0).fit(X, Y)
        for i in range(len(targets)):
            if folds[i] != f:
                continue
            nf = replay_learned(m, *data[i])
            b = base[(targets[i]["set"], targets[i]["id"])]
            res.append({"set": targets[i]["set"], "id": targets[i]["id"], "fwd_learned": nf, "fwd_std": b["fwd_std"],
                        "fwd_n8": b["fwd_n8"], "fwd_dom": b["fwd_dom"], "fwd_oracle": b["fwd_oracle"]})
        print("fold", f, f"{time.time() - t0:.0f}s", flush=True)
    d = {k: float(np.mean([r[k] for r in res])) for k in ("fwd_learned", "fwd_std", "fwd_n8", "fwd_dom", "fwd_oracle")}
    dif = [r["fwd_dom"] - r["fwd_learned"] for r in res]
    d["dom_minus_learned"] = {"mean": float(np.mean(dif)), "median": float(np.median(dif)), "ci95": S.boot_ci(dif)}
    gap = [(r["fwd_std"] - r["fwd_learned"]) / max(1, r["fwd_std"] - r["fwd_oracle"]) for r in res]
    d["gapclosed_learned"] = {"mean": float(np.mean(gap)), "median": float(np.median(gap)), "ci95": S.boot_ci(gap)}
    red = [1 - r["fwd_learned"] / r["fwd_std"] for r in res]
    d["reduction_learned"] = {"mean": float(np.mean(red)), "ci95": S.boot_ci(red)}
    print(d)
    (OUT / "draft_learn.json").write_text(json.dumps({"summary": d, "rows": res}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
