"""S1: token predictability decomposition F/R/Q/S/N + lexical-context breakdown + S_null/S_tol shares.

  .venv/Scripts/python.exe -I scripts/study_s1.py
Output: runs/study/s1_predictability.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from study_common import (CONDS, CTX_GROUP, OUT, S_NULL, S_TOL, encode, guard_setup, lex, load_runs,
                          retrieved_examples, token_context)

N = 4  # n-gram length for R and S
CLASSES = ["F", "R", "Q", "S", "N"]
CONTENT_CTX = {"text", "attr_val", "head_title", "null_attr"}


def ngram_set(ids, n=N):
    return {tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)}


def cover(ids, pred):
    """Mark tokens covered by any n-window i..i+n-1 with pred(i) True."""
    m = [False] * len(ids)
    for i in range(len(ids) - N + 1):
        if pred(i):
            for k in range(i, i + N):
                m[k] = True
    return m


def classify(text, query, examples):
    ids, offs = encode(text)
    ctx, forced = lex(text)
    tctx = token_context(text, offs, ctx)
    T = len(ids)
    # F: every non-whitespace char of the token is grammar-forced
    F = []
    for a, b in offs:
        nz = [k for k in range(a, b) if not text[k].isspace()]
        F.append(bool(nz) and all(forced[k] for k in nz))
    # R: >=4-gram run that appears in the retrieved example HTMLs
    ex_set = set()
    for ex in examples:
        ex_set |= ngram_set(encode("```html\n" + ex["gt_html_flow"] + "\n```")[0])
    R = cover(ids, lambda i: tuple(ids[i:i + N]) in ex_set)
    R_causal = [t >= N - 1 and tuple(ids[t - N + 1:t + 1]) in ex_set for t in range(T)]
    # S: >=4-gram already present in its own prefix (an earlier occurrence starting before i)
    first = {}
    for i in range(T - N + 1):
        first.setdefault(tuple(ids[i:i + N]), i)
    S = cover(ids, lambda i: first[tuple(ids[i:i + N])] < i)
    S_causal = [t >= N - 1 and first[tuple(ids[t - N + 1:t + 1])] < t - N + 1 for t in range(T)]
    # Q: content tokens inside a window whose text (>=4 chars) is a substring of the query (case-insens.)
    ql = query.lower()
    Q = [False] * T
    for i in range(T):
        for k in range(i + 1, min(T, i + 24) + 1):
            s = text[offs[i][0]:offs[k - 1][1]].strip().lower()
            if not s:
                continue
            if s not in ql:
                break
            if len(s) >= 4:
                for t in range(i, k):
                    if tctx[t] in CONTENT_CTX:
                        Q[t] = True
    cls = []
    for t in range(T):
        cls.append("F" if F[t] else "R" if R[t] else "Q" if Q[t] else "S" if S[t] else "N")
    return {"T": T, "cls": cls, "ctx": tctx, "R_causal": R_causal, "S_causal": S_causal,
            "R_any": R, "S_any": S, "Q_any": Q, "F_any": F}


def main():
    guard_setup()
    pilot, runs = load_runs()
    res = {"definition": {
        "F": "all non-ws chars grammar-forced: closing-tag name+'>' matching tag-stack top; opening quote after '='; closing ``` after </html>",
        "R": "covered by a 4-token window whose ids occur in the 3 retrieved examples (TF-IDF top-3, same as bench rich_fs). For zs conditions the examples were NOT in the prompt (retrievable from corpus).",
        "Q": "content-context token (text/attr value) inside a window whose stripped text (>=4 chars) is a case-insensitive substring of the query",
        "S": "covered by a 4-token window that occurred earlier in the output",
        "priority": "F > R > Q > S > N", "tokenizer": "gemma-4-E2B tokenizer.json, re-tokenized response (incl. ```html fence)"},
        "conds": {}}
    for c in CONDS:
        tot = Counter()
        per_sample = []
        ctx_cls = defaultdict(Counter)
        causal = Counter()
        null_tol = Counter()
        for r in runs[c]:
            row = pilot[r["id"]]
            q = row["query_short"] if "short" in c else row["query_rich"]
            out = classify(r["response"], q, retrieved_examples(row))
            cnt = Counter(out["cls"])
            tot.update(cnt)
            T = out["T"]
            ps = {k: cnt[k] / T for k in CLASSES}
            ps["pi"] = 1 - ps["N"]
            ps["T"] = T
            ps["id"] = r["id"]
            per_sample.append(ps)
            for cl, cx in zip(out["cls"], out["ctx"]):
                ctx_cls[CTX_GROUP[cx]][cl] += 1
                null_tol["S_null" if cx in S_NULL else "S_tol" if cx in S_TOL else "other"] += 1
            causal["R_causal"] += sum(out["R_causal"])
            causal["S_causal"] += sum(out["S_causal"])
            causal["RS_causal"] += sum(a or b for a, b in zip(out["R_causal"], out["S_causal"]))
            causal["F"] += cnt["F"]
            causal["T"] += T
        T = sum(tot.values())
        share = {k: tot[k] / T for k in CLASSES}
        Ntot = tot["N"]
        res["conds"][c] = {
            "n": len(runs[c]), "tokens": T, "share": share, "pi": 1 - share["N"],
            "pi_mean_per_sample": sum(p["pi"] for p in per_sample) / len(per_sample),
            "causal_draftable": {"F_or_R_or_S_causal": (causal["F"] + causal["RS_causal"]) / causal["T"],
                                 "R_causal": causal["R_causal"] / causal["T"], "S_causal": causal["S_causal"] / causal["T"]},
            "S_null_share": null_tol["S_null"] / T, "S_tol_share": null_tol["S_tol"] / T,
            "by_context": {g: {"tok_share": sum(v.values()) / T,
                               "N_share_within": v["N"] / max(1, sum(v.values())),
                               "share_of_all_N": v["N"] / max(1, Ntot),
                               "classes": {k: v[k] / max(1, sum(v.values())) for k in CLASSES}}
                           for g, v in sorted(ctx_cls.items())},
            "per_sample": per_sample}
        d = res["conds"][c]
        print(f"{c:18s} T={T:6d} pi={d['pi']:.3f} " + " ".join(f"{k}={share[k]:.3f}" for k in CLASSES)
              + f" | causal={d['causal_draftable']['F_or_R_or_S_causal']:.3f} Snull={d['S_null_share']:.3f} Stol={d['S_tol_share']:.3f}",
              flush=True)
    (OUT / "s1_predictability.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
