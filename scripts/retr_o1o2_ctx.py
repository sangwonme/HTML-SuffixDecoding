"""Where does O2 (any-position oracle) beat O1 (suffix-match oracle)? Diagnostic, CPU 2 threads, BelowNormal.

Both are ideal upper bounds that use the future target; neither is an executable algorithm.
On the O2 trajectory, a step "needs construction" when the best acceptance over all positions (a2) exceeds the best
over positions SD can see lexically (a1: previous token equals the current last token).
For those steps we record: HTML context of the next token (study_suffix_sim.cat_of), context of the last token,
whether the last token occurred anywhere earlier in ctx (novel vs re-occurring), and the extra accepted tokens.
Also pairs O1/O2 with the learned ranker (runs/study/ranker.json, hgb_full) per sample.
Usage: .venv/Scripts/python.exe -I scripts/retr_o1o2_ctx.py  ->  runs/retrieval/o1o2_ctx.json
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_k] = "2"

import json  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import study_draft_sim as S  # noqa: E402
import study_suffix_sim as SS  # noqa: E402
from retr_rank_construct import annotate_full  # noqa: E402
from retr_seq_oracle import accept_all, boot  # noqa: E402

OUT = ROOT / "runs" / "retrieval" / "o1o2_ctx.json"


def main():
    S.guard(2)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    rk = {(r["set"], r["id"]): r for r in json.load(open(ROOT / "runs" / "study" / "ranker.json"))["per_target"]}
    rc = {(r["set"], r["id"]): r for r in json.load(open(ROOT / "runs" / "retrieval" / "rank_construct.json"))["rows"]}
    targets = [t for t in S.load_targets(tok) if t["set"] != "C2pld"]
    agg = {}
    rows = []
    for t in targets:
        grp = "gemma_fs" if t["set"].startswith("gemma") else "poc_greedy"
        A = agg.setdefault(grp, {"steps": 0, "steps_need": 0, "extra_tok": 0, "next_ctx": Counter(), "next_ctx_tok": Counter(),
                                 "last_ctx": Counter(), "novel_last": 0, "last_tok": Counter(), "region": Counter()})
        prompt, y = np.array(t["prompt"], np.int64), np.array(t["y"], np.int64)
        P, T = len(prompt), len(y)
        ctx = np.concatenate([prompt, y])
        strs = [tok.decode([i]) for i in list(prompt) + list(y)]
        ann, _ = annotate_full(strs)
        ann.append(None)
        pos = 1
        while pos < T:
            L, g = P + pos, min(S.GAMMA, t["max_new"] - pos - 1)
            cand, lens = accept_all(ctx, L, y, pos, g)
            a2 = int(lens.max()) if len(lens) else 0
            a1 = 0
            if len(lens):
                ok = (cand >= 1) & (ctx[np.maximum(cand - 1, 0)] == ctx[L - 1])
                a1 = int(lens[ok].max()) if ok.any() else 0
            A["steps"] += 1
            if a2 > a1:
                A["steps_need"] += 1
                A["extra_tok"] += a2 - a1
                c = SS.cat_of(pos, P, ann, strs)
                A["next_ctx"][c] += 1
                A["next_ctx_tok"][c] += a2 - a1
                A["last_ctx"][SS.cat_of(pos - 1, P, ann, strs)] += 1
                novel = not np.any(ctx[:L - 1] == ctx[L - 1])
                A["novel_last"] += int(novel)
                A["last_tok"][strs[L - 1]] += 1
                best = int(cand[int(np.argmax(lens))])
                A["region"]["own_output" if best >= P else "prompt(exemplars/system)"] += 1
            pos += a2 + 1
        k = (t["set"], t["id"])
        rows.append({"group": grp, "set": t["set"], "id": t["id"], "O1": rc[k]["fwd_O1"], "O2": rc[k]["fwd_O2"],
                     "SD": rc[k]["fwd_SD"], "ranker_hgb": rk[k]["hgb_full"], "dom_rule": rk[k]["rule"], "std": rk[k]["std"]})
    out = {"note": "O1/O2 are oracle upper bounds using the future target; not executable algorithms.", "groups": {}}
    for grp, A in agg.items():
        rs = [r for r in rows if r["group"] == grp]
        d = {"n": len(rs)}
        for k in ("std", "SD", "dom_rule", "ranker_hgb", "O1", "O2"):
            d[f"fwd_{k}"] = float(np.mean([r[k] for r in rs]))
        for a, b in (("O1", "O2"), ("ranker_hgb", "O1"), ("ranker_hgb", "O2"), ("SD", "O1")):
            x = [r[a] - r[b] for r in rs]
            d[f"{a}-{b}"] = {"mean": float(np.mean(x)), "ci95": boot(x), "median": float(np.median(x)),
                             "rel_total": float(sum(x) / sum(r[a] for r in rs))}
        d["steps_on_O2_traj"] = A["steps"]
        d["steps_O2_beats_O1"] = A["steps_need"]
        d["share_steps"] = A["steps_need"] / max(1, A["steps"])
        d["extra_tokens"] = A["extra_tok"]
        d["next_token_ctx(steps)"] = dict(A["next_ctx"].most_common())
        d["next_token_ctx(extra tokens)"] = dict(A["next_ctx_tok"].most_common())
        d["last_token_ctx(steps)"] = dict(A["last_ctx"].most_common())
        d["last_token_novel_share"] = A["novel_last"] / max(1, A["steps_need"])
        d["best_source_region"] = dict(A["region"].most_common())
        d["top_last_tokens"] = [[s, n] for s, n in A["last_tok"].most_common(15)]
        out["groups"][grp] = d
    out["rows"] = rows
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["groups"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
