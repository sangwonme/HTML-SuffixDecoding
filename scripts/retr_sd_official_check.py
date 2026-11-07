"""How much do the differences between official ArcticInference SuffixDecoding and our reproduction
(scripts/study_sd_formal.py) matter?  CPU, 2 threads, BelowNormal.  Imports study_* modules only (not modified).

The official C++ extension could not be built/executed here (permission policy: no building or running external
code), so the official rules were read as text (ArcticInference main @ aca5d9a, csrc/suffix_decoding/suffix_tree.cc,
arctic_inference/suffix_decoding/cache.py, simulator.py, vllm/config.py) and ported below in our own code.
Variants (path draft, factor 1 unless noted), each toggling one official rule on top of the reproduction:
  repro        : study_sd_formal rules (should reproduce runs/study/sd_formal.json sd_f1)
  tie1         : sibling order for equal counts: count>=2 -> most recently incremented first (= repro),
                 count==1 -> creation order (earliest occurrence first)  [official _increment_count / insert after tail]
  m63          : match_len < len(context) with context = last 64 tokens  => m <= 63 (repro allows 64)
  ownglob      : global tree also holds the current request's own response (add_active_response extends global)
  official16   : tie1 + m63 + ownglob, MAX_SPEC 16 (vLLM plugin default num_speculative_tokens = 16)
  official64   : same with MAX_SPEC 64 (standalone simulator default: max_spec_tokens 0 -> max_tree_depth 64)
  tree_repro / tree_official : token tree, repro expansion vs official token-level best-first heap (factor 1)
Data: poc greedy + gate2 measured (n=114, the study's main set); global tree = study's app-level cross-fit build.
Usage: .venv/Scripts/python.exe -I scripts/retr_sd_official_check.py -> runs/retrieval/sd_official_check.json
"""
from __future__ import annotations

import os

os.environ["SIM_THREADS"] = "2"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"

import heapq  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import study_draft_sim as S  # noqa: E402
import study_ranker as R  # noqa: E402
import study_sd_formal as F  # noqa: E402
from retr_seq_oracle import boot  # noqa: E402

OUT = ROOT / "runs" / "retrieval" / "sd_official_check.json"
DEPTH, MIN_P = 64, 0.1


def occ_ends(t, n, tail, mmax):
    E = np.nonzero(t[:n] == tail[-1])[0] + 1
    m = 1
    while len(E) and m <= min(mmax, len(tail)):
        yield m, E
        if m == len(tail):
            break
        s = E - m - 1
        ok = s >= 0
        E, s = E[ok], s[ok]
        E = E[t[s] == tail[-m - 1]]
        m += 1


def path_draft(t, n, E, m, max_tokens, tie1):
    base = len(E)
    d, probs, alive = [], [], E
    lim = min(max_tokens, DEPTH - m)
    while len(d) < lim:
        nxt = alive[alive < n]
        nxt = nxt[t[nxt] >= 0]
        if len(nxt) == 0:
            break
        toks = t[nxt]
        vals, cnt = np.unique(toks, return_counts=True)
        best = cnt.max()
        cand = vals[cnt == best]
        if len(cand) > 1:
            if tie1 and best == 1:   # official: count-1 children in creation order -> earliest occurrence first
                first = {int(v): int(nxt[toks == v].min()) for v in cand}
                tok = min(first, key=first.get)
            else:                    # most recently incremented (= most recent occurrence)
                last = {int(v): int(nxt[toks == v].max()) for v in cand}
                tok = max(last, key=last.get)
        else:
            tok = int(cand[0])
        p = best / base
        if p < MIN_P:
            break
        d.append(tok); probs.append(p)
        alive = nxt[toks == tok] + 1
    return d, sum(probs)


def tree_official(t, n, E, m, max_tokens):
    """official _speculate_tree: one token per heap pop, children pushed at node ends (here: every step)."""
    base = len(E)
    lim = min(max_tokens, DEPTH - m)
    nodes, cnt_id = [], 0
    heap = []

    def push_children(alive, parent, prob_parent_ok=True):
        nonlocal cnt_id
        nxt = alive[alive < n]
        nxt = nxt[t[nxt] >= 0]
        if len(nxt) == 0:
            return
        toks = t[nxt]
        vals, cnt = np.unique(toks, return_counts=True)
        order = sorted(zip(vals.tolist(), cnt.tolist(), [int(nxt[toks == v].max()) for v in vals]),
                       key=lambda x: (-x[1], -x[2]))
        for v, c, _ in order:
            p = c / base
            if p < MIN_P:
                break
            cnt_id += 1
            heapq.heappush(heap, (-p, cnt_id, v, nxt[toks == v] + 1, parent))

    push_children(E, -1)
    while len(nodes) < lim and heap:
        negp, _, v, alive, parent = heapq.heappop(heap)
        nodes.append((v, parent, -negp))
        push_children(alive, len(nodes) - 1)
    return nodes


def speculate(cx, L, glob_t, own_n, tail, max_spec, tie1, mmax, tree=None):
    """local first, global must be strictly better (official: draft1 if score1 >= score2)."""
    best, best_sc = None, -1.0
    srcs = [("local", cx.ctx, L)]
    if glob_t is not None:
        srcs.append(("global", glob_t, own_n))
    for src, t, n in srcs:
        sbest, ssc = None, -1.0
        for m, E in occ_ends(t, n, tail, mmax):
            mt = max(0, min(max_spec, int(m * 1.0 + 1e-6)))
            if tree == "official":
                nodes = tree_official(t, n, E, m, mt); sc = sum(p for _, _, p in nodes); dr = nodes
            elif tree == "repro":
                nodes = F.tree_draft(F.Seq(t, None), n, E, m, mt); sc = sum(p for _, _, p in nodes); dr = nodes
            else:
                dr, sc = path_draft(t, n, E, m, mt, tie1)
            if sc >= ssc:
                sbest, ssc = dr, sc
        if ssc > best_sc:
            best, best_sc = sbest, ssc
    return best


def run(cx, glob, max_spec=16, tie1=False, mmax=64, ownglob=False, tree=None):
    P, y = cx.P, cx.y
    T = len(y)
    gt = None
    if glob is not None:
        gt = np.concatenate([glob.t, [-1], y]).astype(np.int64) if ownglob else glob.t
    pos, nf, vt = 1, 1, 0
    while pos < T:
        L = P + pos
        tail = cx.ctx[max(0, L - DEPTH):L]
        own_n = (len(glob.t) + 1 + pos) if (glob is not None and ownglob) else (len(gt) if gt is not None else None)
        d = speculate(cx, L, gt, own_n, tail, max_spec, tie1, mmax, tree) or []
        if tree:
            a = F.accept_tree(d, y, pos)
        else:
            a = F.accept_path(d, y, pos)
        vt += len(d)
        nf += 1
        pos += a + 1
    return nf, vt


VARIANTS = {
    "repro": dict(),
    "tie1": dict(tie1=True),
    "m63": dict(mmax=63),
    "ownglob": dict(ownglob=True),
    "official16": dict(tie1=True, mmax=63, ownglob=True),
    "official64": dict(tie1=True, mmax=63, ownglob=True, max_spec=64),
    "tree_repro": dict(tree="repro"),
    "tree_official": dict(tree="official"),
}


def main():
    S.guard(2)
    from transformers import AutoTokenizer
    from sklearn.model_selection import GroupKFold
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    cache = {}

    def dec(i):
        if i not in cache:
            cache[i] = tok.decode([i])
        return cache[i]
    raw, targets = R.load_all(tok)
    main_sets = ("C1", "C0", "C4g", "C7g", "g2pld", "g2dom")
    c6 = [t for t in targets if t["set"] == "cands6"]
    tg = [t for t in targets if t["set"] in main_sets] + c6[::5]   # same target list/order as study_sd_formal
    extra_resp = [t for k, t in enumerate(c6) if k % 5 != 0]
    cxs = [R.Ctx(t, dec) for t in tg]
    apps = np.array([t["app"] for t in tg])
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(tg)), groups=apps))
    prev = {(r["set"], r["id"], r["T"]): r for r in json.load(open(ROOT / "runs" / "study" / "sd_formal.json"))["rows"]}
    eval_idx = [i for i, t in enumerate(tg) if t["set"] in main_sets]   # main data n=114
    res = {}
    t0 = time.time()
    for f, (tr, te) in enumerate(folds):
        toks = []
        test_apps = set(apps[te])
        for i in tr:
            toks += cxs[i].y.tolist() + [-1]
        for t in extra_resp:
            if t["app"] not in test_apps:
                toks += list(t["y"]) + [-1]
        glob = F.Seq(toks, None)
        for i in te:
            if i not in eval_idx:
                continue
            t = tg[i]
            r = {"set": t["set"], "id": t["id"], "T": len(t["y"]),
                 "study_sd_f1": prev[(t["set"], t["id"], len(t["y"]))]["sd_f1"],
                 "study_sdtree_f1": prev[(t["set"], t["id"], len(t["y"]))]["sdtree_f1"]}
            for name, kw in VARIANTS.items():
                r[name], r[name + "_vt"] = run(cxs[i], glob, **kw)
            res[i] = r
            print(f"fold {f} {t['set']} {t['id']} repro={r['repro']} (study {r['study_sd_f1']}) "
                  f"off16={r['official16']} off64={r['official64']} tree r/o={r['tree_repro']}/{r['tree_official']} "
                  f"{time.time() - t0:.0f}s rss={S.check_rss():.2f}", flush=True)
    rows = [res[i] for i in sorted(res)]
    summ = {"n": len(rows), "repro_matches_study": int(sum(r["repro"] == r["study_sd_f1"] for r in rows)),
            "tree_repro_matches_study": int(sum(r["tree_repro"] == r["study_sdtree_f1"] for r in rows))}
    for name in VARIANTS:
        summ[f"fwd_{name}"] = float(np.mean([r[name] for r in rows]))
        summ[f"verify_tok_per_fwd_{name}"] = float(sum(r[name + "_vt"] for r in rows) / sum(r[name] for r in rows))
        if name not in ("repro", "tree_repro"):
            base = "tree_repro" if name.startswith("tree") else "repro"
            x = [r[base] - r[name] for r in rows]   # > 0: official rule uses fewer forwards
            summ[f"{base}-{name}"] = {"mean": float(np.mean(x)), "ci95": boot(x), "median": float(np.median(x)),
                                      "rel": float(sum(x) / sum(r[base] for r in rows)),
                                      "n_diff": int(sum(v != 0 for v in x))}
    OUT.write_text(json.dumps({"summary": summ, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
