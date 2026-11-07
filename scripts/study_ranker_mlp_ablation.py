"""MLP ranker feature ablation (app-level GroupKFold(5), n=444, same setting as study_ranker.py).
Variants: full / no structure / no history.  Output: runs/study/ranker_mlp_ablation.json
  .venv/Scripts/python.exe -I scripts/study_ranker_mlp_ablation.py
"""
from __future__ import annotations

import os, sys, pathlib, json, time
os.environ.setdefault("SIM_THREADS", "2")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

import study_draft_sim as S
import study_ranker as R
from study_common import OUT, ROOT


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
    cxs = [R.Ctx(t, dec) for t in targets]
    t0 = time.time()
    rule, rows = [], []
    for cx in cxs:
        nf, rw, _ = R.replay(cx, "rule", collect=True)
        rule.append(nf); rows.append(rw)
    print("rule states", f"{time.time() - t0:.0f}s", flush=True)
    apps = np.array([t["app"] for t in targets])
    variants = {"mlp_full": R.STRUCT + R.LEXI + R.HIST, "mlp_noStruct": R.LEXI + R.HIST, "mlp_noHist": R.STRUCT + R.LEXI}
    fwd = {v: np.zeros(len(cxs), int) for v in variants}
    for f, (tr, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(cxs)), groups=apps)):
        trrows = [r for i in tr for r in rows[i]]
        for v, cols in variants.items():
            w = R.train_mlp_np(trrows, cols)
            for i in te:
                fwd[v][i] = R.replay(cxs[i], lambda F, w=w: int(np.argmax(R.mlp_scores(F, w) - 1e-6 * np.arange(len(F)))))[0]
        print("fold", f, f"{time.time() - t0:.0f}s", flush=True)
    sets = [t["set"] for t in targets]
    groups = {"all": range(len(targets)), "poc_greedy": [i for i, s in enumerate(sets) if s in ("C1", "C0", "C4g", "C7g")],
              "gemma_approx": [i for i, s in enumerate(sets) if s.startswith("gemma")],
              "gate2_measured": [i for i, s in enumerate(sets) if s in ("g2pld", "g2dom")],
              "cands6": [i for i, s in enumerate(sets) if s == "cands6"]}
    out = {"setting": "app-level GroupKFold(5), n=444, DOM-rule states, MLP 64-32 pairwise logistic, 6 epochs", "groups": {}}
    for g, ids in groups.items():
        ids = list(ids)
        d = {"n": len(ids), "fwd_dom_rule": float(np.mean([rule[i] for i in ids]))}
        for v in variants:
            d[f"fwd_{v}"] = float(np.mean([fwd[v][i] for i in ids]))
            d[f"fwd_{v}_median"] = float(np.median([fwd[v][i] for i in ids]))
        d["rule_minus_full"] = R.paired([rule[i] for i in ids], [fwd["mlp_full"][i] for i in ids])
        d["noStruct_minus_full"] = R.paired([fwd["mlp_noStruct"][i] for i in ids], [fwd["mlp_full"][i] for i in ids])
        d["noHist_minus_full"] = R.paired([fwd["mlp_noHist"][i] for i in ids], [fwd["mlp_full"][i] for i in ids])
        d["rule_minus_noStruct"] = R.paired([rule[i] for i in ids], [fwd["mlp_noStruct"][i] for i in ids])
        out["groups"][g] = d
        print(g, json.dumps(d)[:900], flush=True)
    out["per_target"] = [{"set": t["set"], "id": t["id"], "app": t["app"], "dom_rule": rule[i], **{v: int(fwd[v][i]) for v in variants}}
                         for i, t in enumerate(targets)]
    (OUT / "ranker_mlp_ablation.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
