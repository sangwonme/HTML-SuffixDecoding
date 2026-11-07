"""Formal SuffixDecoding re-implementation (from ArcticInference csrc/suffix_decoding/suffix_tree.cc +
arctic_inference/suffix_decoding/cache.py, read as text, not executed) vs DOM rule / learned ranker / SD + learned
DOM features.  Ideal greedy-target acceptance replay (CPU).

Official algorithm reproduced (occurrence-count formulation, equivalent to the suffix tree):
  - trees: LOCAL = prompt + response so far (one sequence; the current suffix itself is an occurrence),
           GLOBAL = previous responses (here: model outputs of OTHER apps, app-level cross-fit), max depth 64.
  - speculate(context[-64:]): for match_len m = 1,2,... while the length-m suffix exists in the tree:
        max_tokens = min(MAX_SPEC, floor(m*factor + offset)); depth limit 64-m
        path draft: follow the most frequent child; prob_j = count(P+d[:j]) / count(P);
                    stop when prob < min_token_prob; score = sum(prob)
        keep the best score (>=, so longer matches win ties).  Local vs global: higher score (local on ties).
  - tree draft (use_tree_spec): best-first expansion by prob with a token budget of max_tokens; accepted =
    deepest tree path that matches the target (exact tree verification).
  - defaults: factor 1.0, offset 0, min_token_prob 0.1, depth 64; MAX_SPEC = 16 here (gamma of our runs;
    official default = depth 64).  factor also swept (1, 2, 4, 16) as a tuned variant.
SD + learned: candidate set = the official per-match-length drafts (local and global, deduplicated); a
HistGradientBoosting ranker (app-level 5-fold cross-fit) scores them with frequency features (score, m, count,
probabilities, source) + DOM structure features (agreement of the current annotation with the annotations at the
occurrences of the matched suffix); argmax predicted accepted length.

  .venv/Scripts/python.exe -I scripts/study_sd_formal.py
Output: runs/study/sd_formal.json, runs/study/figures/sd_formal.png
"""
from __future__ import annotations

import os, sys, pathlib, json, time, math
os.environ.setdefault("SIM_THREADS", "2")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

import study_draft_sim as S
import study_ranker as R
from study_common import FIG, OUT, ROOT, check_rss, load_jsonl

DEPTH, MAX_SPEC, MIN_P = 64, int(os.environ.get("SD_MAX_SPEC", "16")), 0.1
OWN_GLOBAL = os.environ.get("SD_OWN_GLOBAL", "0") == "1"   # official: global tree also holds this request's own response
FACTORS = [1.0, 2.0, 4.0, 16.0]


class Seq:
    """token array + annotations for occurrence lookup (local: grows; global: fixed concatenation)."""

    def __init__(self, toks, ann):
        self.t = np.asarray(toks, np.int64)
        self.ann = ann


def occ_ends(seq, n, ctx_tail):
    """ends e (exclusive) in seq[:n] of the suffix ctx_tail, iterating match lengths.
    Yields (m, E) for m = 1.. while non-empty."""
    t = seq.t[:n]
    m = 1
    E = np.nonzero(t == ctx_tail[-1])[0] + 1
    while len(E) and m <= min(DEPTH, len(ctx_tail)):
        yield m, E
        if m == len(ctx_tail):
            break
        s = E - m - 1
        ok = s >= 0
        E = E[ok]
        s = s[ok]
        E = E[t[s] == ctx_tail[-m - 1]]
        m += 1


def path_draft(seq, n, E, m, max_tokens):
    """official _speculate_path on occurrences E (ends) of a length-m suffix."""
    t = seq.t
    base = len(E)
    d, probs = [], []
    alive = E
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
        if len(cand) > 1:   # tie -> most recent occurrence
            last = {int(v): int(nxt[toks == v].max()) for v in cand}
            tok = max(last, key=last.get)
        else:
            tok = int(cand[0])
        p = best / base
        if p < MIN_P:
            break
        d.append(tok); probs.append(p)
        alive = nxt[toks == tok] + 1
    return d, probs


def tree_draft(seq, n, E, m, max_tokens):
    """official _speculate_tree: best-first by prob; returns list of (token, parent_idx)."""
    import heapq
    t = seq.t
    base = len(E)
    lim = min(max_tokens, DEPTH - m)
    nodes = []
    heap = [(-1.0, 0, E, -1)]
    cnt_id = 1
    while len(nodes) < lim and heap:
        negp, _, alive, parent = heapq.heappop(heap)
        nxt = alive[alive < n]
        nxt = nxt[t[nxt] >= 0]
        if len(nxt) == 0:
            continue
        toks = t[nxt]
        vals, cnt = np.unique(toks, return_counts=True)
        for v, c in sorted(zip(vals.tolist(), cnt.tolist()), key=lambda x: -x[1]):
            p = c / base
            if p < MIN_P:
                break
            if len(nodes) >= lim:
                break
            nodes.append((v, parent, p))
            heapq.heappush(heap, (-p, cnt_id, nxt[toks == v] + 1, len(nodes) - 1)); cnt_id += 1
    return nodes


def speculate(local, nloc, glob, ctx_tail, factor, tree=False, collect=False):
    """returns best draft (list) or tree nodes, plus candidate list if collect."""
    best, best_score, cands = None, -1.0, []
    gseq, gn = (glob if isinstance(glob, tuple) else (glob, None))
    for src, seq, n in (("local", local, nloc), ("global", gseq, gn)):
        if seq is None:
            continue
        nn = len(seq.t) if n is None else n
        sbest, sscore = None, -1.0
        for m, E in occ_ends(seq, nn, ctx_tail):
            mt = max(0, min(MAX_SPEC, int(m * factor + 1e-6)))
            if tree:
                nodes = tree_draft(seq, nn, E, m, mt)
                sc = sum(p for _, _, p in nodes)
                dr = nodes
            else:
                d, probs = path_draft(seq, nn, E, m, mt)
                sc = sum(probs)
                dr = d
                if collect and d:
                    cands.append({"d": d, "probs": probs, "m": m, "count": len(E), "src": src, "E": E[E < nn], "seq": seq, "score": sc})
            if sc >= sscore:
                sbest, sscore = dr, sc
        if sscore > best_score:   # local first; global must be strictly better
            best, best_score = sbest, sscore
    return best, cands


def accept_tree(nodes, y, pos):
    if not nodes:
        return 0
    depth_ok = {}
    best = 0
    for i, (tok, par, _) in enumerate(nodes):
        dpar = 0 if par < 0 else depth_ok.get(par, None)
        if dpar is None:
            continue
        if pos + dpar < len(y) and y[pos + dpar] == tok:
            depth_ok[i] = dpar + 1
            best = max(best, dpar + 1)
    return best


def accept_path(d, y, pos):
    a = 0
    while a < len(d) and pos + a < len(y) and d[a] == y[pos + a]:
        a += 1
    return a


def cand_feats(c, cur):
    probs = c["probs"]
    E = c["E"][-60:]
    sims = [S.sim_score(c["seq"].ann[e] if e < len(c["seq"].ann) else None, cur) for e in E]
    sims = [s for s in sims if s >= 0] or [0.0]
    return [c["score"], c["m"], math.log1p(c["count"]), len(c["d"]), probs[0], probs[-1], int(c["src"] == "local"),
            max(sims), float(np.mean(sims)), float(np.mean([s >= 9 for s in sims]))]


FREQ_COLS, STRUCT_COLS = list(range(0, 7)), list(range(7, 10))


def run(cx, glob, factor=1.0, mode="path", chooser=None, collect=False):
    """mode: path | tree | learned.  Returns n_fwd, verify_tokens, rows."""
    P, y = cx.P, cx.y
    T = len(y)
    local = Seq(cx.ctx, cx.ann)
    gbase = glob
    if OWN_GLOBAL:
        gt = np.concatenate([glob.t if glob is not None else np.zeros(0, np.int64), np.array([-1], np.int64), cx.y])
        ga = (list(glob.ann) if glob is not None else []) + [None] + list(cx.ann[P:P + T])
        gcomb, g0 = Seq(gt, ga), (len(glob.t) if glob is not None else 0) + 1
    pos, nf, vt, rows = 1, 1, 0, []
    while pos < T:
        L = P + pos
        tail = cx.ctx[max(0, L - DEPTH):L]
        if OWN_GLOBAL:
            glob = (gcomb, g0 + pos)
        if mode == "tree":
            nodes, _ = speculate(local, L, glob, tail, factor, tree=True)
            a = accept_tree(nodes or [], y, pos); vt += len(nodes or [])
        else:
            d, cands = speculate(local, L, glob, tail, factor, collect=(mode == "learned" or collect))
            d = d or []
            if mode == "learned" and len(cands) > 1:
                cur = cx.ann[L]
                F = np.array([cand_feats(c, cur) for c in cands], float)
                j = chooser(F)
                d = cands[j]["d"]
            if collect and len(cands) > 1:
                cur = cx.ann[L]
                F = np.array([cand_feats(c, cur) for c in cands], float)
                A = np.array([accept_path(c["d"], y, pos) for c in cands], float)
                rows.append((F, A))
            a = accept_path(d, y, pos); vt += len(d)
        nf += 1
        pos += a + 1
    return nf, vt, rows


def fit(rows, cols):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.concatenate([F[:, cols] for F, A in rows]); Y = np.concatenate([A for F, A in rows])
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, random_state=0).fit(X, Y)
    return lambda F: int(np.argmax(m.predict(F[:, cols]) + 1e-6 * -np.arange(len(F))))


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
    main_sets = ("C1", "C0", "C4g", "C7g", "g2pld", "g2dom")
    c6 = [t for t in targets if t["set"] == "cands6"]
    tg = [t for t in targets if t["set"] in main_sets] + c6[::5]   # cands6: every 5th (aux, time)
    extra_resp = [t for k, t in enumerate(c6) if k % 5 != 0]       # other cands6 outputs: global-tree data only
    t0 = time.time()
    cxs = [R.Ctx(t, dec) for t in tg]
    apps = np.array([t["app"] for t in tg])
    print("targets", len(tg), f"{time.time() - t0:.0f}s", flush=True)
    # response sequences + annotations for global trees
    extra = []
    for t in extra_resp:
        yt = list(t["y"]); ann, _ = S.annotate([dec(i) for i in yt]); ann.append(None)
        extra.append((t["app"], yt, ann))
    resp = []
    for cx in cxs:
        yt = cx.y.tolist()
        ann, _ = S.annotate([dec(i) for i in yt]); ann.append(None)
        resp.append((yt, ann))
    # baselines from ranker replay (DOM rule, cross-fit MLP) + std
    fmap = json.loads((OUT / "ranker_fold_map.json").read_text(encoding="utf-8"))
    W = {f: R.load_weights(OUT / f"ranker_mlp_fold{f}.npz") for f in range(5)}
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(np.zeros(len(tg)), groups=apps))
    res = [dict(set=t["set"], id=t["id"], app=t["app"], T=len(t["y"])) for t in tg]
    LF = float(os.environ.get("ONLY_LEARNED", "0"))
    if LF:
        prev = {(r["set"], r["id"], r["T"]): r for r in json.loads((OUT / "sd_formal_f1.json").read_text(encoding="utf-8"))["rows"]}
        res = [prev[(r["set"], r["id"], r["T"])] for r in res]
    for i, cx in enumerate(cxs if not LF else []):
        res[i]["std"] = S.run_policy("std", cx.prompt, cx.y, cx.ann, cx.reg)[0]
        res[i]["dom_rule"] = R.replay(cx, "rule")[0]
        w = W[fmap[tg[i]["id"]]]
        res[i]["ranker_mlp"] = R.replay(cx, lambda F, w=w: int(np.argmax(R.mlp_scores(F, w) - 1e-6 * np.arange(len(F)))))[0]
        res[i]["oracle"] = S.run_policy("oracle", cx.prompt, cx.y, cx.ann, cx.reg)[0]
    print("baselines", f"{time.time() - t0:.0f}s", flush=True)
    for f, (tr, te) in enumerate(folds):
        toks, anns = [], []
        test_apps = set(apps[te])
        for i in tr:
            yt, ann = resp[i]
            toks += yt + [-1]; anns += ann[:len(yt)] + [None]
        for app, yt, ann in extra:
            if app not in test_apps:
                toks += yt + [-1]; anns += ann[:len(yt)] + [None]
        glob = Seq(toks, anns)
        # official SD (local+global), factors; tree for factor 1 and best
        rows_tr = []
        fac_l = LF if LF else 1.0
        for i in tr:
            if True:
                rows_tr += run(cxs[i], glob, fac_l, "path", collect=True)[2]
        ch_full = fit(rows_tr, FREQ_COLS + STRUCT_COLS)
        ch_freq = fit(rows_tr, FREQ_COLS)
        if LF:
            for i in te:
                nf, vt, _ = run(cxs[i], glob, LF, "learned", chooser=ch_full); res[i][f"sd{LF:g}_learned_full"] = nf; res[i][f"sd{LF:g}_learned_full_vt"] = vt
                nf, vt, _ = run(cxs[i], glob, LF, "learned", chooser=ch_freq); res[i][f"sd{LF:g}_learned_freq"] = nf
            print("fold", f, f"{time.time() - t0:.0f}s", flush=True)
            continue
        for i in te:
            for fac in FACTORS:
                nf, vt, _ = run(cxs[i], glob, fac, "path")
                res[i][f"sd_f{fac:g}"] = nf; res[i][f"sd_f{fac:g}_vt"] = vt
            nf, vt, _ = run(cxs[i], glob, 1.0, "tree"); res[i]["sdtree_f1"] = nf; res[i]["sdtree_f1_vt"] = vt
            nf, vt, _ = run(cxs[i], None, 1.0, "path"); res[i]["sd_localonly_f1"] = nf
            nf, vt, _ = run(cxs[i], glob, 1.0, "learned", chooser=ch_full); res[i]["sd_learned_full"] = nf; res[i]["sd_learned_full_vt"] = vt
            nf, vt, _ = run(cxs[i], glob, 1.0, "learned", chooser=ch_freq); res[i]["sd_learned_freq"] = nf
        print("fold", f, f"{time.time() - t0:.0f}s rss={check_rss():.2f}", flush=True)
        (OUT / "sd_formal.json").write_text(json.dumps({"partial_fold": f, "rows": res}, default=float), encoding="utf-8")
    summarize(res)


def summarize(res):
    pol = ["std", "dom_rule", "ranker_mlp", "sd_f1", "sd_f2", "sd_f4", "sd_f16", "sd_localonly_f1", "sdtree_f1",
           "sd_learned_freq", "sd_learned_full", "oracle"] + [k for k in ("sd16_learned_freq", "sd16_learned_full") if k in res[0]]
    groups = {"main(poc+gate2)": [r for r in res if r["set"] != "cands6"],
              "poc_greedy": [r for r in res if r["set"] in ("C1", "C0", "C4g", "C7g")],
              "gate2": [r for r in res if r["set"] in ("g2pld", "g2dom")],
              "cands6(aux)": [r for r in res if r["set"] == "cands6"]}
    out = {"config": {"DEPTH": DEPTH, "MAX_SPEC": MAX_SPEC, "MIN_P": MIN_P, "FACTORS": FACTORS,
                      "source": "ArcticInference main (suffix_tree.cc speculate/_speculate_path/_speculate_tree, cache.py speculate)"},
           "groups": {}, "rows": res}
    for g, rs in groups.items():
        if not rs:
            continue
        d = {"n": len(rs)}
        for p in pol:
            v = [r[p] for r in rs]
            d[p] = {"mean": float(np.mean(v)), "median": float(np.median(v)), "sum_red_vs_std": float(1 - np.sum(v) / np.sum([r["std"] for r in rs]))}
        best_sd = min(("sd_f1", "sd_f2", "sd_f4", "sd_f16"), key=lambda p: np.sum([r[p] for r in rs]))
        d["best_sd_factor"] = best_sd
        for a, b in (("sd_f1", "sd_learned_full"), (best_sd, "sd_learned_full"), ("sd_learned_freq", "sd_learned_full"),
                     ("sd_f1", "ranker_mlp"), (best_sd, "ranker_mlp"), ("dom_rule", "ranker_mlp"), ("sd_f1", "sd_learned_freq"),
                     ("sd_f16", "sd16_learned_full"), ("sd_f16", "sd16_learned_freq"), ("sd16_learned_freq", "sd16_learned_full")):
            if b not in rs[0] or a not in rs[0]:
                continue
            dif = [r[a] - r[b] for r in rs]
            d[f"paired_{a}_minus_{b}"] = {"mean": float(np.mean(dif)), "median": float(np.median(dif)), "ci95": S.boot_ci(dif)}
        d["verify_tokens_per_fwd"] = {p: float(np.sum([r[p + "_vt"] for r in rs]) / np.sum([r[p] for r in rs]))
                                      for p in ("sd_f1", "sdtree_f1", "sd_learned_full")}
        d["verdict_cond4_learned_over_formal_SD"] = bool(d["paired_sd_f1_minus_sd_learned_full"]["ci95"][0] > 0)
        if "paired_sd_f16_minus_sd16_learned_full" in d:
            d["verdict_cond4_tuned_f16"] = bool(d["paired_sd_f16_minus_sd16_learned_full"]["ci95"][0] > 0)
        out["groups"][g] = d
        print(g, json.dumps({k: (v if not isinstance(v, dict) or "mean" not in v else round(v["mean"], 1)) for k, v in d.items()
                             if not k.startswith("paired")}, default=str)[:900])
        for k, v in d.items():
            if k.startswith("paired"):
                print("   ", k, round(v["mean"], 2), v["median"], [round(x, 2) for x in v["ci95"]])
    (OUT / "sd_formal.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    keys = ["std", "dom_rule", "ranker_mlp", "sd_f1", "sdtree_f1", "sd_learned_freq", "sd_learned_full", "oracle"]
    names = ["PLD", "DOM rule", "ranker", "SD (official)", "SD tree", "SD+learned(freq)", "SD+learned(+DOM)", "oracle"]
    cols = ["#2a78d6", "#1baf7a", "#eb6834", "#6250d6", "#9085e9", "#eda100", "#e87ba4", "#b9b8b2"]
    gs = [g for g in ("main(poc+gate2)", "poc_greedy", "gate2", "cands6(aux)") if g in out["groups"]]
    fig, axes = plt.subplots(1, len(gs), figsize=(3.6 * len(gs), 3.6))
    for ax, g in zip(axes, gs):
        d = out["groups"][g]
        vals = [d[k]["mean"] for k in keys]
        ax.bar(range(len(keys)), vals, color=cols, edgecolor="#fcfcfb")
        ax.set_xticks(range(len(keys)), names, rotation=60, fontsize=7)
        p = d["paired_sd_f1_minus_sd_learned_full"]
        ax.set_title(f"{g} n={d['n']}\nSD − SD+learned {p['mean']:.1f} [{p['ci95'][0]:.1f},{p['ci95'][1]:.1f}]", loc="left", fontsize=8)
    axes[0].set_ylabel("forwards per request (mean)")
    fig.tight_layout(); fig.savefig(FIG / "sd_formal.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
