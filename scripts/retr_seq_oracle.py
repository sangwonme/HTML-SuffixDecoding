"""Sequential vs myopic oracle for draft-position choice (greedy replay, CPU, 1 thread, BelowNormal).

Uses scripts/study_draft_sim.py by import only (targets, annotator, DOM scoring, gamma=16, acceptance =
longest common prefix + 1 bonus token, n_fwd = 1 prefill + #verify steps).

Key fact used: in greedy replay the context at output position pos is fixed (prompt + y[:pos]), so the
future depends only on pos.  A position choice matters only through its accepted length a, and the next
state is pos + a + 1.  Hence:
  A(pos)      = set of accepted lengths achievable by any start position in ctx (plus 0 = useless draft)
  myopic      : a = max A(pos)                       (== study "oracle")
  seq(M,H)    : candidates = top-M distinct values of A(pos); for each, take it and then H myopic steps;
                pick the best tokens-per-forward over those 1+H steps (ties -> larger a). Requested approx.
  seq_exact   : DP  f(pos) = 1 + min_{a in A(pos)} f(pos + a + 1),  f(pos >= T) = 0. Exact optimum over all
                position sequences (= M=all, H=inf).
  dom         : study DOM rule (myopic, executable)
  dom_la(M,H) : among DOM's top-M ranked positions, H-step lookahead with DOM continuation, scored with the
                true target -> ORACLE (not a self-rollout).
  dom_dp(M)   : DP restricted to DOM's top-M positions at every step (oracle).
Usage: SIM_THREADS=1 .venv/Scripts/python.exe -I scripts/retr_seq_oracle.py [--sets C1,C0,C4g,C7g,gemma]
Writes runs/retrieval/seq_oracle.json
"""
from __future__ import annotations

import os

os.environ["SIM_THREADS"] = "1"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import study_draft_sim as S  # noqa: E402  (import only; not modified)

OUT = ROOT / "runs" / "retrieval" / "seq_oracle.json"
MS, HS = (4, 8), (1, 3, 6)
DOM_M = 8
SEED = 20261107


def accept_all(ctx, L, y, pos, g):
    """Accepted length (LCP with y[pos:], capped at g, source must stay < L) for every start s < L."""
    cand = np.nonzero(ctx[:L] == y[pos])[0]
    if len(cand) == 0 or g <= 0:
        return cand, np.zeros(0, np.int64)
    lens = np.ones(len(cand), np.int64)
    alive = np.arange(len(cand))
    a = 1
    while a < g and pos + a < len(y) and len(alive):
        s = cand[alive]
        ok = (s + a < L)
        ok[ok] = ctx[s[ok] + a] == y[pos + a]
        alive = alive[ok]
        lens[alive] += 1
        a += 1
    return cand, lens


def boot(x, B=2000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(SEED)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def sample(t, tok):
    prompt = np.array(t["prompt"], np.int64)
    y = np.array(t["y"], np.int64)
    T, P = len(y), len(prompt)
    ctx = np.concatenate([prompt, y]).astype(np.int64)
    max_new = t["max_new"]

    def gcap(pos):
        return min(S.GAMMA, max_new - pos - 1)

    # ---------------- oracle accepted-length sets
    A = [None] * (T + 1)
    for pos in range(1, T):
        _, lens = accept_all(ctx, P + pos, y, pos, gcap(pos))
        A[pos] = np.unique(np.concatenate([lens, [0]]))[::-1]  # descending
    amax = [0] * (T + 1)
    for pos in range(1, T):
        amax[pos] = int(A[pos][0])

    def run(choose):
        pos, nf = 1, 1
        while pos < T:
            pos += choose(pos) + 1
            nf += 1
        return nf

    def myo_after(p, H):
        """Advance H myopic steps from p; return (new_pos, steps_used)."""
        h = 0
        while h < H and p < T:
            p += amax[p] + 1
            h += 1
        return p, h

    res = {"T": T, "fwd_myopic": run(lambda p: amax[p])}
    # exact DP
    f = np.zeros(T + S.GAMMA + 2, np.int64)
    for pos in range(T - 1, 0, -1):
        f[pos] = 1 + min(f[min(T, pos + int(a) + 1)] for a in A[pos])
    res["fwd_seq_exact"] = int(1 + f[1])
    # approximate lookahead
    for M in MS:
        for H in HS:
            def choose(pos, M=M, H=H):
                best, ba = None, None
                for a in A[pos][:M]:
                    p, h = myo_after(pos + int(a) + 1, H)
                    sc = (p - pos) / (1 + h)
                    if best is None or sc > best + 1e-12:
                        best, ba = sc, int(a)
                return ba
            res[f"fwd_seq_M{M}_H{H}"] = run(choose)

    # ---------------- DOM (executable myopic) + oracle lookahead / DP over its top-M
    strs = [tok.decode([i]) for i in list(prompt) + list(y)]
    ann, reg = S.annotate(strs)
    reg = [(-2 if i >= P else r_) for i, r_ in enumerate(reg)]
    ann.append(None)
    idx = S.Index(3)
    idx.extend(ctx, len(ctx))  # hits() filters c <= L-1, so a full index equals the incremental one
    ann_pos = [c for c in range(len(ctx)) if ann[c] is not None]
    dom_cache = {}

    def dom_ranked(pos):
        """DOM candidate start positions in DOM preference order (top DOM_M), with accepted lengths."""
        if pos in dom_cache:
            return dom_cache[pos]
        L = P + pos
        g = gcap(pos)
        cur = ann[L] if L < len(ann) else None
        hits = []
        for n in range(3, 0, -1):
            if L <= n:
                continue
            h = idx.hits(ctx, L, n)
            if h:
                hits = h
                break
        if hits:
            pool = hits[-300:][::-1]
        elif cur is not None:
            pool = [c for c in reversed(ann_pos) if 0 < c < L]
        else:
            pool = []
        sc = [S.sim_score(ann[c], cur) for c in pool]
        order = sorted(range(len(pool)), key=lambda i: -sc[i])  # stable: ties keep most-recent-first
        top = [pool[i] for i in order[:DOM_M]]
        out = [(c, S.accept_len(ctx, c, L, y, pos, g)) for c in top]
        dom_cache[pos] = out
        return out

    def dom_a(pos):
        r = dom_ranked(pos)
        return r[0][1] if r else 0

    res["fwd_dom"] = run(dom_a)
    res["study_dom"] = S.run_policy("dom", prompt, y, ann, reg, max_new=max_new)[0]  # validation

    def dom_after(p, H):
        h = 0
        while h < H and p < T:
            p += dom_a(p) + 1
            h += 1
        return p, h

    for M in (4, 8):
        for H in (0,) + HS:  # H=0: myopic best-of-M by true acceptance (isolates the planning effect)
            def choose(pos, M=M, H=H):
                r = dom_ranked(pos)
                if not r:
                    return 0
                best, ba = None, None
                for _, a in r[:M]:
                    p, h = dom_after(pos + a + 1, H)
                    sc = (p - pos) / (1 + h)
                    if best is None or sc > best + 1e-12:
                        best, ba = sc, a
                return ba
            res[f"fwd_domla_M{M}_H{H}"] = run(choose)
    fd = np.zeros(T + S.GAMMA + 2, np.int64)
    for pos in range(T - 1, 0, -1):
        opts = {a for _, a in dom_ranked(pos)} or {0}
        fd[pos] = 1 + min(fd[min(T, pos + a + 1)] for a in opts)
    res["fwd_domdp_M8"] = int(1 + fd[1])
    return res


def summarize(rows, keys):
    out = {}
    groups = {"poc_greedy(C1,C0,C4g,C7g)": [r for r in rows if r["set"] in ("C1", "C0", "C4g", "C7g")],
              "gemma_fs(approx)": [r for r in rows if r["set"].startswith("gemma")]}
    for s in sorted({r["set"] for r in rows}):
        groups[s] = [r for r in rows if r["set"] == s]
    for g, rs in groups.items():
        if not rs:
            continue
        d = {"n": len(rs)}
        for k in keys:
            d[k] = float(np.mean([r[k] for r in rs]))
        for base, others in (("fwd_myopic", [k for k in keys if k.startswith("fwd_seq")]),
                             ("fwd_dom", [k for k in keys if k.startswith(("fwd_domla", "fwd_domdp"))])):
            for k in others:
                dif = [r[base] - r[k] for r in rs]  # > 0: sequential uses fewer forwards
                red = [1 - r[k] / r[base] for r in rs]
                d[f"{k}__vs_{base[4:]}"] = {
                    "diff_mean": float(np.mean(dif)), "diff_ci95": boot(dif), "diff_median": float(np.median(dif)),
                    "red_mean": float(np.mean(red)), "red_mean_ci95": boot(red),
                    "red_total": float(1 - sum(r[k] for r in rs) / sum(r[base] for r in rs)),
                    "n_better": int(sum(x > 0 for x in dif)), "n_worse": int(sum(x < 0 for x in dif))}
        e = d.get("fwd_seq_exact__vs_myopic")
        if e:
            d["verdict_seq_exact_meaningful"] = bool(e["diff_ci95"][0] > 0 and e["red_mean"] >= 0.03)
        out[g] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="C1,C0,C4g,C7g,gemma")
    args = ap.parse_args()
    S.guard(1)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    want = args.sets.split(",")
    targets = [t for t in S.load_targets(tok)
               if t["set"] in want or ("gemma" in want and t["set"].startswith("gemma"))]
    targets.sort(key=lambda t: (t["set"].startswith("gemma"), len(t["y"])))  # poc first, short first
    print("targets", len(targets), flush=True)
    rows, t0 = [], time.time()
    keys = None
    for i, t in enumerate(targets):
        r = sample(t, tok)
        # validation against the study simulator (myopic oracle and DOM must match run_policy)
        prompt, y = np.array(t["prompt"], np.int64), np.array(t["y"], np.int64)
        r["study_oracle"] = S.run_policy("oracle", prompt, y, None, None, max_new=t["max_new"])[0]
        r.update({"set": t["set"], "id": t["id"]})
        rows.append(r)
        keys = [k for k in r if k.startswith("fwd_")]
        print(f"[{i + 1}/{len(targets)}] {t['set']} {t['id']} T={r['T']} myo={r['fwd_myopic']} "
              f"(study {r['study_oracle']}) exact={r['fwd_seq_exact']} M8H3={r['fwd_seq_M8_H3']} "
              f"dom={r['fwd_dom']} (study {r['study_dom']}) domla={r['fwd_domla_M8_H3']} domdp={r['fwd_domdp_M8']} "
              f"{time.time() - t0:.0f}s rss={S.check_rss():.2f}", flush=True)
        if (i + 1) % 10 == 0 or i + 1 == len(targets):
            res = {"gamma": S.GAMMA, "M": MS, "H": HS, "dom_M": DOM_M, "n_done": len(rows),
                   "myopic_matches_study_oracle": int(sum(x["fwd_myopic"] == x["study_oracle"] for x in rows)),
                   "dom_matches_study_dom": int(sum(x["fwd_dom"] == x["study_dom"] for x in rows)),
                   "summary": summarize(rows, keys), "rows": rows}
            OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res["summary"], indent=1)[:6000])
    print("myopic == study oracle:", res["myopic_matches_study_oracle"], "/", len(rows))


if __name__ == "__main__":
    main()
