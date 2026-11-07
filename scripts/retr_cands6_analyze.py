"""Latency oracle gap analysis for the 6-candidate pilot run. CPU only, no GPU.

Input : runs/poc/E2B_L2_cands6.jsonl (one row per pilot id x candidate; read only)
        runs/retrieval/pilot_cands6{.json,_feats.jsonl,_analysis.jsonl}
Output: runs/retrieval/cands6_analysis.json (+ printed summary)

1. Within-request gap: min / tfidf1 / max total time per request; mean gap with bootstrap CI.
2. Decomposition: log T = log N_out - log(tok/fwd) + log t_fwd, with t_fwd := T / n_fwd (exact identity;
   t_fwd absorbs prefill/TTFT). Within-request variance share of each term = cov(term, log T) / var(log T)
   after demeaning per request (shares sum to 1).
3. Predictability: within-request Spearman of each deployable feature vs T; pairwise logistic ranker
   (feature differences -> "a faster than b"), leave-one-request-out. Picks: argmin predicted time
   (unconstrained) and the same restricted to candidates whose checklist >= tfidf1's (this constraint uses
   outcome info, as requested). Compared with tfidf1, struct1, oracle1, min-latency oracle.
4. Verdict among: large & predictable / large & unpredictable / small gap / quality-latency trade-off.
Usage: .venv/Scripts/python.exe -I scripts/retr_cands6_analyze.py [--results PATH]
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RET = ROOT / "runs" / "retrieval"
FEATS = ["tfidf_sim", "struct_sim", "n_tokens", "css_frac", "pre_body_frac", "n_elems", "n_classes",
         "n_top_blocks", "category_match"]
SEED = 20261107


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def g(rec, *keys, default=None):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def boot(x, B=2000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(SEED)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "runs" / "poc" / "E2B_L2_cands6.jsonl"))
    ap.add_argument("--scores", default=None, help="optional separate scores jsonl (same keys)")
    args = ap.parse_args()
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass

    cands = json.load(open(RET / "pilot_cands6.json", encoding="utf-8"))
    F = {(f["pilot_id"], f["cand_id"]): f for f in jl(RET / "pilot_cands6_feats.jsonl")}
    A = {(a["pilot_id"], a["cand_id"]): a for a in jl(RET / "pilot_cands6_analysis.jsonl")}
    rows = [r for r in jl(args.results) if "_meta" not in r]
    extra = {}
    if args.scores:
        for s in jl(args.scores):
            extra[(g(s, "id", "pilot_id"), g(s, "cand_id", "exemplar_id"))] = s
    R = {}
    for r in rows:
        pid = g(r, "pilot_id", "id")
        cid = g(r, "cand_id", "exemplar_id")
        if cid is None:
            ex = g(r, "example_ids", default=[])
            cid = ex[0] if len(ex) == 1 else None
        if pid not in cands or cid not in cands[pid]:
            continue
        r = {**r, **extra.get((pid, cid), {})}
        T = float(g(r, "total_s", "time_s", "total_time"))
        N = float(g(r, "output_tokens", "n_out"))
        nf = float(g(r, "n_fwd", "forwards"))
        R[(pid, cid)] = {"T": T, "N": N, "nf": nf, "tpf": N / nf,
                         "check": g(r, "check_score", "checklist", "checklist_score"),
                         "elem_f1": g(r, "elem_f1", "element_f1")}
    reqs = [p for p in cands if all((p, c) in R for c in cands[p])]
    print(f"rows {len(rows)}, matched {len(R)}, complete requests {len(reqs)}/{len(cands)}")
    if len(reqs) < 5:
        raise SystemExit("too few complete requests")
    have_q = all(R[(p, c)]["check"] is not None for p in reqs for c in cands[p])

    def role(p, c):
        return F[(p, c)]["role"].split("->")[0] if "->" not in F[(p, c)]["role"] else "tfidf_next"

    def pick_role(p, name):
        for c in cands[p]:
            if F[(p, c)]["role"] == name:
                return c
        return None

    out = {"n_requests": len(reqs)}
    # ------------------------------------------------------------- 1. gap
    T = {p: np.array([R[(p, c)]["T"] for c in cands[p]]) for p in reqs}
    t1 = {p: R[(p, cands[p][0])]["T"] for p in reqs}  # slot 0 = tfidf1
    out["T_tfidf1_mean"] = float(np.mean([t1[p] for p in reqs]))
    out["T_min_mean"] = float(np.mean([T[p].min() for p in reqs]))
    out["T_max_mean"] = float(np.mean([T[p].max() for p in reqs]))
    out["gap_min_minus_tfidf1"] = boot([T[p].min() - t1[p] for p in reqs])
    out["gap_max_minus_tfidf1"] = boot([T[p].max() - t1[p] for p in reqs])
    out["rel_gap_min"] = boot([(T[p].min() - t1[p]) / t1[p] for p in reqs])
    out["frac_tfidf1_is_min"] = float(np.mean([T[p].argmin() == 0 for p in reqs]))
    by_role = {}
    for p in reqs:
        for c in cands[p]:
            by_role.setdefault(role(p, c), []).append(R[(p, c)]["T"] - t1[p])
    out["dT_vs_tfidf1_by_role"] = {k: boot(v) for k, v in by_role.items()}

    # ------------------------------------------------------------- 2. decomposition
    lt, ln, lp, lf = [], [], [], []
    for p in reqs:
        x = np.array([[np.log(R[(p, c)]["T"]), np.log(R[(p, c)]["N"]), np.log(R[(p, c)]["tpf"]),
                       np.log(R[(p, c)]["T"] / R[(p, c)]["nf"])] for c in cands[p]])
        x -= x.mean(0)
        lt += list(x[:, 0]); ln += list(x[:, 1]); lp += list(x[:, 2]); lf += list(x[:, 3])
    lt, ln, lp, lf = map(np.array, (lt, ln, lp, lf))
    v = float(np.mean(lt * lt))
    out["within_var_logT"] = v
    out["share_Nout"] = float(np.mean(ln * lt) / v)
    out["share_accept(-log tok/fwd)"] = float(-np.mean(lp * lt) / v)
    out["share_t_fwd"] = float(np.mean(lf * lt) / v)

    # ------------------------------------------------------------- 3a. Spearman within request
    sp = {}
    for k in FEATS + ["omega4_vs_gt(analysis)"]:
        rs = []
        for p in reqs:
            if k.endswith("(analysis)"):
                f = [A[(p, c)]["omega4_vs_gt"] for c in cands[p]]
            else:
                f = [F[(p, c)][k] for c in cands[p]]
            if len(set(f)) > 1:
                rho = spearmanr(f, T[p]).statistic
                if np.isfinite(rho):
                    rs.append(rho)
        sp[k] = boot(rs) + [len(rs)] if rs else None
    out["spearman_within_vs_T"] = sp

    # ------------------------------------------------------------- 3b. pairwise logistic, LOO by request
    from sklearn.linear_model import LogisticRegression
    Xf = {p: np.array([[F[(p, c)][k] for k in FEATS] for c in cands[p]], float) for p in reqs}
    allX = np.vstack([Xf[p] for p in reqs])
    mu, sd = allX.mean(0), allX.std(0) + 1e-9

    def pairs(ps):
        X, y = [], []
        for p in ps:
            Z = (Xf[p] - mu) / sd
            for a in range(len(Z)):
                for b in range(len(Z)):
                    if a != b and T[p][a] != T[p][b]:
                        X.append(Z[a] - Z[b]); y.append(int(T[p][a] < T[p][b]))
        return np.array(X), np.array(y)

    picks_u, picks_q, pacc = {}, {}, []
    coefs = []
    for p in reqs:
        Xtr, ytr = pairs([q for q in reqs if q != p])
        m = LogisticRegression(C=1.0, fit_intercept=False, max_iter=2000).fit(Xtr, ytr)
        coefs.append(m.coef_[0])
        s = ((Xf[p] - mu) / sd) @ m.coef_[0]  # higher = predicted faster
        picks_u[p] = int(np.argmax(s))
        Xte, yte = pairs([p])
        pacc.append(float(np.mean((Xte @ m.coef_[0] > 0) == yte)))
        if have_q:
            q0 = R[(p, cands[p][0])]["check"]
            ok = [i for i, c in enumerate(cands[p]) if R[(p, c)]["check"] >= q0]
            picks_q[p] = max(ok, key=lambda i: s[i])
    out["pairwise_acc_loo"] = boot(pacc)
    out["ranker_coef_mean(std units, + = faster)"] = dict(zip(FEATS, np.mean(coefs, 0).round(3).tolist()))

    # ------------------------------------------------------------- 3c. compare selections
    def summarize(sel):
        dT = [R[(p, cands[p][sel[p]])]["T"] - t1[p] for p in reqs]
        d = {"T": float(np.mean([R[(p, cands[p][sel[p]])]["T"] for p in reqs])), "dT_vs_tfidf1": boot(dT)}
        if have_q:
            d["check"] = float(np.mean([R[(p, cands[p][sel[p]])]["check"] for p in reqs]))
            d["dcheck_vs_tfidf1"] = boot([R[(p, cands[p][sel[p]])]["check"] - R[(p, cands[p][0])]["check"]
                                          for p in reqs])
            ef = [R[(p, cands[p][sel[p]])]["elem_f1"] for p in reqs]
            if all(e is not None for e in ef):
                d["elem_f1"] = float(np.mean(ef))
        return d

    def slot_of(p, name):
        c = pick_role(p, name)
        return cands[p].index(c) if c else 0

    sels = {"tfidf1": {p: 0 for p in reqs},
            "struct1": {p: slot_of(p, "struct1") for p in reqs},
            "oracle1(omega)": {p: slot_of(p, "oracle1") for p in reqs},
            "short": {p: slot_of(p, "short") for p in reqs},
            "min_latency_oracle": {p: int(T[p].argmin()) for p in reqs},
            "learned(LOO)": picks_u}
    if have_q:
        sels["learned(LOO)+quality>=tfidf1"] = picks_q
        sels["min_latency_oracle+quality>=tfidf1"] = {
            p: min([i for i, c in enumerate(cands[p]) if R[(p, c)]["check"] >= R[(p, cands[p][0])]["check"]],
                   key=lambda i: T[p][i]) for p in reqs}
    out["selections"] = {k: summarize(v) for k, v in sels.items()}
    note = [k for k in ("struct1", "oracle1(omega)") if any(pick_role(p, k.split("(")[0]) is None for p in reqs)]
    if note:
        out["note"] = f"{note}: some requests had this role replaced (duplicate) -> slot 0 used there"

    # ------------------------------------------------------------- 4. verdict
    gap = out["gap_min_minus_tfidf1"]
    lrn = out["selections"]["learned(LOO)"]["dT_vs_tfidf1"]
    large = gap[2] < 0 and -gap[0] / out["T_tfidf1_mean"] >= 0.10
    captured = lrn[0] / gap[0] if gap[0] else 0.0
    predictable = lrn[2] < 0 and captured >= 0.3
    trade = None
    if have_q:
        mo = out["selections"]["min_latency_oracle"]["dcheck_vs_tfidf1"]
        trade = mo[2] < 0
    out["verdict_inputs"] = {"gap_rel": -gap[0] / out["T_tfidf1_mean"], "gap_ci_excludes_0": gap[2] < 0,
                             "learned_captured_frac": captured, "learned_ci_excludes_0": lrn[2] < 0,
                             "min_latency_quality_drop_significant": trade}
    out["verdict"] = ("quality-latency trade-off" if trade else
                      "small gap" if not large else
                      "large & predictable" if predictable else "large & unpredictable")
    (RET / "cands6_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
