"""Gate 1: SuffixDecoding (approximate re-implementation) vs SuffixDecoding + DOM structure, greedy replay.

Ideal greedy-target acceptance analysis (same replay as study_draft_sim.py; std policy there reproduces logged PLD
forwards 10/10).  Context pool = full prompt (exemplar HTML etc.) + own output prefix.

SD (K=1):  m = longest suffix of ctx[:L] that re-occurs earlier (match ending <= L-2, m <= MMAX).  Occurrences
           c (continuation positions) of that suffix form the candidate set; the draft is the greedy most-frequent
           path through their continuations (at each depth the token with the largest (weighted) count among
           still-alive occurrences; stop when its conditional share < MIN_P or length reaches
           min(GMAX, ceil(ALPHA*m))).
SD tree:   top-K (K=4) distinct paths by beam over the same weighted counts; accepted = best branch (lossless
           tree verification of a greedy target); verification cost = total draft tokens (reported).
SD+DOM:    identical, but each occurrence carries weight 1 + lambda*sim(c), sim = structural agreement of the
           annotation at c with the current position (lexical state, attr/CSS key, tag-path suffix, depth,
           sibling index, last event; study_draft_sim.sim_score / 11.5).  lambda chosen by sample-level 5-fold CV.

  .venv/Scripts/python.exe -I scripts/study_suffix_sim.py
Output: runs/study/suffix_sim.json, runs/study/figures/suffix_sim.png
"""
from __future__ import annotations

import math, os, sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

import study_draft_sim as S
from study_common import FIG, OUT, ROOT, check_rss

GMAX, MMAX, ALPHA, MIN_P, KTREE = 16, 32, float(os.environ.get("SD_ALPHA", "2.0")), 0.1, 4
LAMBDAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
SIMMAX = 11.5


class SuffixIndex:
    """n-gram (n<=MMAX) -> list of continuation positions; grows incrementally."""

    def __init__(self):
        self.d, self.L = {}, 0

    def extend(self, ctx, L):
        for c in range(max(1, self.L), L):
            for n in range(1, MMAX + 1):
                j = c - n
                if j < 0:
                    break
                self.d.setdefault(tuple(ctx[j:c].tolist()), []).append(c)
        self.L = L

    def longest(self, ctx, L):
        lo = None
        for n in range(min(MMAX, L - 1), 0, -1):
            h = self.d.get(tuple(ctx[L - n:L].tolist()))
            if h:
                h = [c for c in h if c <= L - 1]
                if h:
                    return n, h
        return 0, []


def paths(ctx, L, occ, w, glen, K):
    """Beam over weighted continuation counts. Returns up to K drafts (lists)."""
    beams = [([], list(range(len(occ))), 1.0)]
    done = []
    for j in range(glen):
        nb = []
        for draft, alive, sc in beams:
            cnt = {}
            tot = 0.0
            for i in alive:
                c = occ[i] + j
                if c >= L:
                    continue
                t = int(ctx[c]); cnt.setdefault(t, [0.0, []]); cnt[t][0] += w[i]; cnt[t][1].append(i); tot += w[i]
            if not cnt:
                done.append((draft, sc)); continue
            exts = sorted(cnt.items(), key=lambda kv: -kv[1][0])
            ext_any = False
            for t, (cw, idx) in exts[:K]:
                p = cw / tot
                if p < MIN_P:
                    continue
                nb.append((draft + [t], idx, sc * p)); ext_any = True
            if not ext_any:
                done.append((draft, sc))
        if not nb:
            beams = []
            break
        nb.sort(key=lambda b: -b[2])
        beams = nb[:K]
    done += [(d, s) for d, _, s in beams]
    done = [d for d in done if d[0]]
    done.sort(key=lambda x: -x[1])
    out, seen = [], set()
    for d, _ in done:
        key = tuple(d)
        if key not in seen:
            seen.add(key); out.append(d)
        if len(out) == K:
            break
    return out


def run_sd(prompt, y, ann, lam, K=1, record=False, max_new=2048):
    ctx = np.concatenate([prompt, y]).astype(np.int64)
    P, T = len(prompt), len(y)
    idx = SuffixIndex()
    pos, nf, cost, steps = 1, 1, 0, []
    while pos < T:
        L = P + pos
        idx.extend(ctx, L)
        m, occ = idx.longest(ctx, L)
        a = 0
        if m:
            occ = occ[-400:]
            cur = ann[L]
            if lam > 0 and cur is not None:
                w = [1.0 + lam * max(0.0, S.sim_score(ann[c], cur)) / SIMMAX for c in occ]
            else:
                w = [1.0] * len(occ)
            glen = min(GMAX, max_new - pos - 1, max(1, math.ceil(ALPHA * m)))
            drafts = paths(ctx, L, occ, w, glen, K)
            for d in drafts:
                cost += len(d)
                aa = 0
                while aa < len(d) and pos + aa < T and d[aa] == y[pos + aa]:
                    aa += 1
                a = max(a, aa)
        if record:
            steps.append((pos, a))
        nf += 1
        pos += a + 1
    return nf, cost, steps


def cat_of(i, P, ann, strs):
    a_ = ann[P + i]; ts = strs[P + i]
    c = S.lex_of_token(a_)
    if "<" in ts or ">" in ts or c == "boundary":
        return "boundary"
    return {"attrname": "attr", "attrval": "attr", "cssprop": "css_prop", "csssel": "css_prop",
            "cssval": "css_val", "tagname": "tagname", "text": "text"}.get(c, "other")


def stats(a, b):
    """paired a-b"""
    d = [x - y for x, y in zip(a, b)]
    return {"mean": float(np.mean(d)), "median": float(np.median(d)), "ci95": S.boot_ci(d)}


def main():
    S.guard(int(os.environ.get("SIM_THREADS", "2")))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    sim = json.loads((OUT / "draft_sim.json").read_text(encoding="utf-8"))
    base = {(r["set"], r["id"]): r for r in sim["rows"]}
    targets = [t for t in S.load_targets(tok) if t["set"] != "C2pld"]
    prep = []
    for t in targets:
        prompt = np.array(t["prompt"], np.int64); y = np.array(t["y"], np.int64)
        strs = [tok.decode([i]) for i in list(prompt) + list(y)]
        ann, _ = S.annotate(strs); ann.append(None)
        prep.append((prompt, y, ann, strs))
    print("targets", len(targets), flush=True)
    t0 = time.time()
    # lambda sweep (K=1)
    fwd = {lam: [] for lam in LAMBDAS}
    for i, (prompt, y, ann, strs) in enumerate(prep):
        for lam in LAMBDAS:
            fwd[lam].append(run_sd(prompt, y, ann, lam)[0])
        if i % 10 == 0:
            print(i, targets[i]["set"], {l: fwd[l][-1] for l in LAMBDAS}, f"{time.time() - t0:.0f}s rss={check_rss():.2f}", flush=True)
    sets = [t["set"] for t in targets]
    res = {"config": {"GMAX": GMAX, "MMAX": MMAX, "ALPHA": ALPHA, "MIN_P": MIN_P, "KTREE": KTREE, "LAMBDAS": LAMBDAS},
           "groups": {}}
    groups = {"poc_greedy": [i for i, s in enumerate(sets) if s in ("C1", "C0", "C4g", "C7g")],
              "gemma_fs_approx": [i for i, s in enumerate(sets) if s.startswith("gemma")]}
    per = []
    for g, ids in groups.items():
        # sample-level 5-fold CV for lambda
        folds = {i: k % 5 for k, i in enumerate(ids)}
        chosen, sdd = {}, {}
        for f in range(5):
            tr = [i for i in ids if folds[i] != f]
            best = min(LAMBDAS[1:], key=lambda l: np.mean([fwd[l][i] / fwd[0.0][i] for i in tr]))
            for i in ids:
                if folds[i] == f:
                    chosen[i] = best; sdd[i] = fwd[best][i]
        sd = [fwd[0.0][i] for i in ids]
        sdd_l = [sdd[i] for i in ids]
        b = [base[(sets[i], targets[i]["id"])] for i in ids]
        d = {"n": len(ids), "lambda_chosen": {str(k): v for k, v in zip(*np.unique([chosen[i] for i in ids], return_counts=True))}}
        cols = {"std": [x["fwd_std"] for x in b], "n8": [x["fwd_n8"] for x in b], "dom_rule": [x["fwd_dom"] for x in b],
                "oracle": [x["fwd_oracle"] for x in b], "SD": sd, "SD+DOM(cv)": sdd_l}
        for k, v in cols.items():
            d[f"fwd_{k}"] = float(np.mean(v))
            d[f"fwd_{k}_median"] = float(np.median(v))
            d[f"red_vs_std_samplemean_{k}"] = float(np.mean([1 - x / s for x, s in zip(v, cols["std"])]))
            d[f"red_vs_std_sum_{k}"] = float(1 - np.sum(v) / np.sum(cols["std"]))
        d["red_SDDOM_vs_SD_samplemean"] = float(np.mean([1 - a / s for a, s in zip(sdd_l, sd)]))
        d["red_SDDOM_vs_SD_sum"] = float(1 - np.sum(sdd_l) / np.sum(sd))
        d["paired_SD_minus_SDDOM"] = stats(sd, sdd_l)
        d["paired_SD_minus_domrule"] = stats(sd, cols["dom_rule"])
        d["paired_SDDOM_minus_oracle"] = stats(sdd_l, cols["oracle"])
        d["gate1_pass"] = bool(d["paired_SD_minus_SDDOM"]["ci95"][0] > 0)
        d["lambda_sweep_fwd_mean"] = {str(l): float(np.mean([fwd[l][i] for i in ids])) for l in LAMBDAS}
        res["groups"][g] = d
        for i in ids:
            per.append({"set": sets[i], "id": targets[i]["id"], "T": int(len(prep[i][1])), "SD": fwd[0.0][i],
                        "SDDOM": sdd[i], "lambda": chosen[i], **{f"SD_l{l}": fwd[l][i] for l in LAMBDAS}})
    # tree version (K=4) and context attribution, poc only (time)
    lam_star = float(max(res["groups"]["poc_greedy"]["lambda_chosen"].items(), key=lambda kv: kv[1])[0])
    tree, ctxgain = [], {}
    for i in groups["poc_greedy"]:
        prompt, y, ann, strs = prep[i]
        P = len(prompt)
        f0, c0, st0 = run_sd(prompt, y, ann, 0.0, K=1, record=True)
        f1, c1, st1 = run_sd(prompt, y, ann, lam_star, K=1, record=True)
        for tag, st in (("SD", st0), ("SDDOM", st1)):
            for pos, a in st:
                for j in range(pos, min(len(y), pos + a)):
                    c = cat_of(j, P, ann, strs)
                    ctxgain.setdefault(c, {"SD": 0, "SDDOM": 0})[tag] += 1
        ft0, ct0, _ = run_sd(prompt, y, ann, 0.0, K=KTREE)
        ft1, ct1, _ = run_sd(prompt, y, ann, lam_star, K=KTREE)
        tree.append({"id": targets[i]["id"], "set": sets[i], "SDtree": ft0, "SDtree_cost": ct0, "SDDOMtree": ft1,
                     "SDDOMtree_cost": ct1, "SD_cost": c0, "SDDOM_cost": c1})
    tg = res["groups"]["poc_greedy"]
    tg["lambda_for_tree_and_ctx"] = lam_star
    tg["fwd_SDtree"] = float(np.mean([t["SDtree"] for t in tree]))
    tg["fwd_SDDOMtree"] = float(np.mean([t["SDDOMtree"] for t in tree]))
    tg["paired_SDtree_minus_SDDOMtree"] = stats([t["SDtree"] for t in tree], [t["SDDOMtree"] for t in tree])
    tg["draft_tokens_per_fwd"] = {k: float(np.sum([t[k + "_cost"] for t in tree]) / np.sum([t[k.replace("_cost", "")] if k in ("SDtree", "SDDOMtree") else 1 for t in tree]))
                                  for k in ()}
    for k, fk in (("SD", None), ("SDDOM", None), ("SDtree", "SDtree"), ("SDDOMtree", "SDDOMtree")):
        cost = np.sum([t[k + "_cost"] for t in tree])
        nfw = np.sum([t[k] for t in tree]) if k in ("SDtree", "SDDOMtree") else np.sum([p[k] for p in per if p["set"] in ("C1", "C0", "C4g", "C7g")])
        tg[f"verify_tokens_per_fwd_{k}"] = float(cost / nfw)
    tg["gain_ctx_SDDOM_minus_SD"] = {c: {"SD": v["SD"], "SDDOM": v["SDDOM"], "extra": v["SDDOM"] - v["SD"]} for c, v in ctxgain.items()}
    res["per_sample"] = per
    res["tree"] = tree
    (OUT / "suffix_sim.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    for g, d in res["groups"].items():
        print(g, json.dumps({k: v for k, v in d.items() if not k.startswith("red_vs_std_samplemean")}, default=float)[:3000], flush=True)
    fig(res)


def fig(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    keys = ["std", "n8", "dom_rule", "SD", "SD+DOM(cv)", "oracle"]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4", "#b9b8b2"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, g in zip(axes, ("poc_greedy", "gemma_fs_approx")):
        d = res["groups"][g]
        vals = [d[f"fwd_{k}"] for k in keys]
        ax.bar(range(len(keys)), vals, color=cols, edgecolor="#fcfcfb", hatch=None)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        p = d["paired_SD_minus_SDDOM"]
        ax.set_xticks(range(len(keys)), keys, rotation=20)
        ax.set_title(f"{g} (n={d['n']})  SD−SD+DOM={p['mean']:.1f} [{p['ci95'][0]:.1f},{p['ci95'][1]:.1f}]", loc="left", fontsize=9)
        ax.set_ylabel("forwards per sample (mean)")
    fig.tight_layout(); fig.savefig(FIG / "suffix_sim.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
