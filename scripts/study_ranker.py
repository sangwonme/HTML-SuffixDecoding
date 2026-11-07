"""Stage 1-2: learned ranker over DOM top-8 draft candidates (ideal greedy-target acceptance replay, CPU).

Targets: poc greedy C1/C0/C4g/C7g, gate2_p50 PLD+DOM measured outputs (gen_ids), cands6 300 outputs (1 exemplar,
re-tokenized response), gemma rich_fs (approx).  Duplicates (same prompt+output) removed.
Candidates J_t: positions of the longest matching n-gram (n=3..1, most recent 300) ranked by the DOM rule score
(study_draft_sim.sim_score; ties -> most recent); continuations deduplicated; top 8.  Label A_j = accepted length.
States: visited by the DOM-rule policy (+ optional DAgger round from the learned policy).
Models: HistGradientBoosting regression on A (argmax), 2-layer MLP with pairwise logistic loss.
Split: GroupKFold(5) by app (main) and by request id.  Ablations: -structure, -history.

  .venv/Scripts/python.exe -I scripts/study_ranker.py
Output: runs/study/ranker.json, runs/study/figures/ranker.png
"""
from __future__ import annotations

import os, sys, pathlib, json, time, math
os.environ.setdefault("SIM_THREADS", "4")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

import study_draft_sim as S
from study_common import FIG, OUT, ROOT, check_rss, load_jsonl

GAMMA, TOPK, NHIT = 16, 8, 300
FEATS = ["dom_score", "dom_rank", "lex_eq", "key_eq", "path_suf", "parent_eq", "depth_cur", "depth_cand", "depth_diff",
         "sib_diff", "event_eq", "relpos_diff",                                   # structure (0..11)
         "hit_n", "bm_len", "next_freq", "occ_rank", "is_self", "log_dist",       # lexical (12..17)
         "is_cont", "log_dist_prev", "same_reg_prev", "acc_last", "acc_mean3"]   # history (18..22)
STRUCT = list(range(0, 12))
LEXI = list(range(12, 18))
HIST = list(range(18, 23))


# ------------------------------------------------------------------ targets
def load_all(tok):
    pilot = {r["id"]: r for r in load_jsonl(ROOT / "data" / "processed" / "pilot.jsonl")}
    base = [t for t in S.load_targets(tok) if t["set"] != "C2pld"]
    out = []
    for t in base:
        t["app"] = pilot[t["id"]]["app"]
        out.append(t)
    mid = ROOT / "data" / "processed" / "mid"
    mid_pool = {r["id"]: r for r in load_jsonl(mid / "train_pool.jsonl")}
    mid_pilot = {r["id"]: r for r in load_jsonl(mid / "pilot.jsonl")}
    html = {r["id"]: r["gt_html_flow"] for r in load_jsonl(ROOT / "data" / "processed" / "train_pool.jsonl")}

    def prompt_for(qid, ex_ids):
        msgs = [{"role": "system", "content": S.SYSTEM}]
        for e in ex_ids:
            msgs.append({"role": "user", "content": mid_pool[e]["query_mid"]})
            msgs.append({"role": "assistant", "content": "```html\n" + html[e] + "\n```"})
        msgs.append({"role": "user", "content": mid_pilot[qid]["query_mid"]})
        return list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])

    for fn, tag in (("gate2_p50/E2B_pld.jsonl", "g2pld"), ("gate2_p50/E2B_dom.jsonl", "g2dom")):
        for r in load_jsonl(ROOT / "runs" / "poc" / fn):
            if "id" not in r:
                continue
            p = prompt_for(r["id"], r["example_ids"])
            out.append({"set": tag, "id": r["id"], "app": pilot[r["id"]]["app"], "prompt": p, "y": list(r["gen_ids"]),
                        "prompt_ok": len(p) == r["prompt_tokens"], "max_new": 2048})
    for r in load_jsonl(ROOT / "runs" / "poc" / "E2B_L2_cands6_raw.jsonl"):
        if "id" not in r:
            continue
        p = prompt_for(r["id"], [r["cand_id"]])
        y = tok(r["response"], add_special_tokens=False)["input_ids"]
        out.append({"set": "cands6", "id": r["id"], "app": pilot[r["id"]]["app"], "prompt": p, "y": list(y),
                    "prompt_ok": len(p) == r["prompt_tokens"], "max_new": 2048})
    seen, ded = set(), []
    for t in out:
        k = (hash(tuple(t["prompt"])), hash(tuple(t["y"])))
        if k in seen:
            continue
        seen.add(k); ded.append(t)
    return out, ded


# ------------------------------------------------------------------ per-target context
class Ctx:
    def __init__(self, t, dec):
        self.prompt = np.array(t["prompt"], np.int64)
        self.y = np.array(t["y"], np.int64)
        self.ctx = np.concatenate([self.prompt, self.y])
        self.P = len(self.prompt)
        strs = [dec(i) for i in self.ctx.tolist()]
        self.strs = strs
        ann, reg = S.annotate(strs)
        ann.append(None)
        self.ann = ann
        self.reg = [(-2 if i >= self.P else r) for i, r in enumerate(reg)] + [-2]
        # region start offsets
        self.rstart = {}
        for i, r in enumerate(self.reg):
            self.rstart.setdefault(r, i)
        self.max_new = t["max_new"]


def candidates(cx, L, idx):
    """DOM-rule ordered unique candidates (list of dicts) + hit_n + all hits."""
    ctx, ann = cx.ctx, cx.ann
    hit_n, hits = 0, []
    for n in range(3, 0, -1):
        if L <= n:
            continue
        h = idx.hits(ctx, L, n)
        if h:
            hit_n, hits = n, h
            break
    if not hits:
        return hit_n, [], []
    cur = ann[L]
    hs = hits[-NHIT:][::-1]   # most recent first
    sc = [S.sim_score(ann[c], cur) for c in hs]
    order = sorted(range(len(hs)), key=lambda i: (-sc[i], i))   # score desc, recency
    out, seen = [], set()
    nxt = {}
    for c in hs:
        nxt[int(ctx[c])] = nxt.get(int(ctx[c]), 0) + 1
    for rank, i in enumerate(order):
        c = hs[i]
        key = tuple(ctx[c:min(L, c + GAMMA)].tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append({"c": c, "score": sc[i], "occ_rank": i, "next_freq": nxt[int(ctx[c])] / len(hs)})
        if len(out) == TOPK:
            break
    return hit_n, out, hs


def featurize(cx, L, pos, cands, hit_n, hist):
    ann, ctx = cx.ann, cx.ctx
    cur = ann[L]
    F = []
    prev_src, prev_a, accs = hist
    for r, cd in enumerate(cands):
        c = cd["c"]
        a = ann[c]
        if a is None or cur is None:
            st = [0, 0, 0, 0, 0, 0, 0]
            dcur = cur[3] if cur else 0
            dc = a[3] if a else 0
        else:
            pa, pb = a[2], cur[2]
            k = 0
            while k < min(len(pa), len(pb)) and pa[len(pa) - 1 - k] == pb[len(pb) - 1 - k]:
                k += 1
            st = [int(a[0] == cur[0]), int(a[1] == cur[1] and a[1] != ""), k,
                  int(len(pa) > 0 and len(pb) > 0 and pa[-1] == pb[-1]), abs(a[4] - cur[4]), int(a[5] == cur[5]), 0]
            dcur, dc = cur[3], a[3]
        rc = cx.reg[c]
        rel_c = (c - cx.rstart.get(rc, 0)) / 1000.0
        rel_cur = pos / 1000.0
        bm = 0
        while bm < 16 and c - 1 - bm >= 0 and ctx[c - 1 - bm] == ctx[L - 1 - bm]:
            bm += 1
        cont = int(prev_src is not None and c == prev_src + prev_a + 1)
        dprev = abs(c - (prev_src + prev_a + 1)) if prev_src is not None else 5000
        same = int(prev_src is not None and cx.reg[prev_src] == rc)
        F.append([cd["score"], r, st[0], st[1], st[2], st[3], dcur, dc, abs(dcur - dc), st[4], st[5], abs(rel_c - rel_cur),
                  hit_n, bm, cd["next_freq"], cd["occ_rank"], int(rc == -2), math.log1p(L - c),
                  cont, math.log1p(dprev), same, accs[-1] if accs else 0, float(np.mean(accs[-3:])) if accs else 0])
    return np.array(F, float)


def replay(cx, chooser, collect=False):
    """chooser(F, cands) -> index, or 'rule' / 'top8oracle'.  Returns n_fwd, rows(optional), timing."""
    idx = S.Index(3)
    y, P = cx.y, cx.P
    T = len(y)
    pos, nf = 1, 1
    prev_src, prev_a, accs = None, 0, []
    rows = []
    tsum, nst = 0.0, 0
    while pos < T:
        L = P + pos
        g = min(GAMMA, cx.max_new - pos - 1)
        idx.extend(cx.ctx, L)
        t0 = time.perf_counter()
        hit_n, cands, hs = candidates(cx, L, idx)
        a, src = 0, None
        if cands:
            A = [S.accept_len(cx.ctx, cd["c"], L, y, pos, g) for cd in cands]
            if chooser == "rule":
                j = 0
            elif chooser == "top8oracle":
                j = int(np.argmax(A))
            else:
                F = featurize(cx, L, pos, cands, hit_n, (prev_src, prev_a, accs))
                j = chooser(F)
                if collect:
                    pass
            tsum += time.perf_counter() - t0; nst += 1
            if collect and len(cands) > 1:
                F = featurize(cx, L, pos, cands, hit_n, (prev_src, prev_a, accs))
                rows.append((F, np.array(A, float)))
            src = cands[j]["c"]
            a = A[j]
        else:
            # structure-only fallback (same as DOM rule in study_draft_sim)
            cur = cx.ann[L]
            if cur is not None:
                best = None
                for c in range(L - 1, 0, -1):
                    if cx.ann[c] is None:
                        continue
                    sc = S.sim_score(cx.ann[c], cur)
                    if best is None or sc > best:
                        best, src = sc, c
                if src is not None:
                    a = S.accept_len(cx.ctx, src, L, y, pos, g)
        nf += 1
        prev_src, prev_a = src, a
        accs.append(a)
        pos += a + 1
    return nf, rows, (tsum / max(1, nst)) * 1000


# ------------------------------------------------------------------ models
def fit_hgb(rows, cols):
    from sklearn.ensemble import HistGradientBoostingRegressor
    X = np.concatenate([F[:, cols] for F, A in rows]); Y = np.concatenate([A for F, A in rows])
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=0).fit(X, Y)
    return lambda F: int(np.argmax(m.predict(F[:, cols]) + 1e-6 * -np.arange(len(F))))


def fit_mlp(rows, cols, epochs=6):
    import torch
    torch.set_num_threads(int(os.environ.get("SIM_THREADS", "4")))
    torch.manual_seed(0)
    X = np.concatenate([F[:, cols] for F, A in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    pi, pj = [], []
    off = 0
    for F, A in rows:
        n = len(A)
        for i in range(n):
            for j in range(n):
                if A[i] > A[j]:
                    pi.append(off + i); pj.append(off + j)
        off += n
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    pi, pj = torch.tensor(pi), torch.tensor(pj)
    net = torch.nn.Sequential(torch.nn.Linear(len(cols), 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    B = 4096
    for ep in range(epochs):
        perm = torch.randperm(len(pi))
        for k in range(0, len(pi), B):
            b = perm[k:k + B]
            s = net(Xt).squeeze(-1) if False else None
            si = net(Xt[pi[b]]).squeeze(-1); sj = net(Xt[pj[b]]).squeeze(-1)
            loss = torch.nn.functional.softplus(-(si - sj)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()

    def choose(F):
        with torch.no_grad():
            s = net(torch.tensor((F[:, cols] - mu) / sd, dtype=torch.float32)).squeeze(-1).numpy()
        return int(np.argmax(s + 1e-6 * -np.arange(len(F))))
    return choose


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    return {"mean": float(np.mean(d)), "median": float(np.median(d)), "ci95": S.boot_ci(d)}


def main():
    S.guard(int(os.environ["SIM_THREADS"]))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    cache = {}

    def dec(i):
        if i not in cache:
            cache[i] = tok.decode([i])
        return cache[i]
    raw, targets = load_all(tok)
    from collections import Counter
    print("targets raw", len(raw), "dedup", len(targets), Counter(t["set"] for t in targets),
          "prompt_ok", Counter((t["set"], t["prompt_ok"]) for t in targets), flush=True)
    t0 = time.time()
    cxs = [Ctx(t, dec) for t in targets]
    print("annotated", f"{time.time() - t0:.0f}s rss={check_rss():.2f}", flush=True)
    # baseline replays + DOM-rule states
    base = []
    rows_rule = []
    for i, cx in enumerate(cxs):
        nf_rule, rows, ms_rule = replay(cx, "rule", collect=True)
        nf_o8, _, _ = replay(cx, "top8oracle")
        nf_std, _ = S.run_policy("std", cx.prompt, cx.y, cx.ann, cx.reg)
        nf_or, _ = S.run_policy("oracle", cx.prompt, cx.y, cx.ann, cx.reg)
        base.append({"rule": nf_rule, "top8oracle": nf_o8, "std": nf_std, "oracle": nf_or, "rule_ms": ms_rule})
        rows_rule.append(rows)
        if i % 50 == 0:
            print(i, targets[i]["set"], base[-1], f"{time.time() - t0:.0f}s rss={check_rss():.2f}", flush=True)
    apps = np.array([t["app"] for t in targets])
    reqs = np.array([t["id"] for t in targets])
    from sklearn.model_selection import GroupKFold
    results = {}
    variants = {"hgb_full": ("hgb", STRUCT + LEXI + HIST), "mlp_full": ("mlp", STRUCT + LEXI + HIST),
                "hgb_noStruct": ("hgb", LEXI + HIST), "hgb_noHist": ("hgb", STRUCT + LEXI)}
    for split, groups in (("app", apps), ("request", reqs)):
        gkf = GroupKFold(n_splits=5)
        fwd = {v: np.zeros(len(cxs), int) for v in variants}
        ms = {v: np.zeros(len(cxs)) for v in variants}
        vlist = variants if split == "app" else {"hgb_full": variants["hgb_full"]}
        for f, (tr, te) in enumerate(gkf.split(np.zeros(len(cxs)), groups=groups)):
            trrows = [r for i in tr for r in rows_rule[i]]
            for v, (kind, cols) in vlist.items():
                ch = fit_hgb(trrows, cols) if kind == "hgb" else fit_mlp(trrows, cols)
                for i in te:
                    nf, _, m = replay(cxs[i], ch)
                    fwd[v][i] = nf; ms[v][i] = m
            print(split, "fold", f, f"{time.time() - t0:.0f}s", flush=True)
        results[split] = {v: {"fwd": fwd[v].tolist(), "ms": ms[v].tolist()} for v in vlist}
    # DAgger 1 round (app split, hgb_full)
    gkf = GroupKFold(n_splits=5)
    fwd_d = np.zeros(len(cxs), int)
    for f, (tr, te) in enumerate(gkf.split(np.zeros(len(cxs)), groups=apps)):
        trrows = [r for i in tr for r in rows_rule[i]]
        ch = fit_hgb(trrows, STRUCT + LEXI + HIST)
        extra = []
        for i in tr:
            _, rows, _ = replay(cxs[i], ch, collect=True)
            extra += rows
        ch2 = fit_hgb(trrows + extra, STRUCT + LEXI + HIST)
        for i in te:
            fwd_d[i] = replay(cxs[i], ch2)[0]
        print("dagger fold", f, f"{time.time() - t0:.0f}s", flush=True)
    results["app"]["hgb_dagger"] = {"fwd": fwd_d.tolist()}
    summarize(targets, base, results)


def summarize(targets, base, results):
    sets = [t["set"] for t in targets]
    out = {"n_targets": len(targets), "sets": {s: sets.count(s) for s in set(sets)}, "groups": {}}
    R = results["app"]
    rule = [b["rule"] for b in base]; o8 = [b["top8oracle"] for b in base]
    std = [b["std"] for b in base]; orc = [b["oracle"] for b in base]
    grp = {"all": list(range(len(targets))),
           "poc_greedy": [i for i, s in enumerate(sets) if s in ("C1", "C0", "C4g", "C7g")],
           "gate2_measured": [i for i, s in enumerate(sets) if s in ("g2pld", "g2dom")],
           "cands6": [i for i, s in enumerate(sets) if s == "cands6"],
           "gemma_approx": [i for i, s in enumerate(sets) if s.startswith("gemma")]}
    best = min(("hgb_full", "mlp_full", "hgb_dagger"), key=lambda v: np.sum(R[v]["fwd"]))
    out["main_model"] = best
    for g, ids in grp.items():
        if not ids:
            continue
        sel = lambda v: [v[i] for i in ids]
        d = {"n": len(ids)}
        cols = {"std": sel(std), "dom_rule": sel(rule), "top8oracle": sel(o8), "oracle": sel(orc)}
        for v in R:
            cols[v] = sel(R[v]["fwd"])
        for k, v in cols.items():
            d[f"fwd_{k}"] = float(np.mean(v)); d[f"fwd_{k}_median"] = float(np.median(v))
        d["paired_rule_minus_main"] = paired(cols["dom_rule"], cols[best])
        gap = np.sum(cols["dom_rule"]) - np.sum(cols["top8oracle"])
        d["top8_gap_closed_sum"] = float((np.sum(cols["dom_rule"]) - np.sum(cols[best])) / max(1, gap))
        d["rule_to_top8_gap_sum"] = float(gap)
        d["ablation_noStruct_minus_full"] = paired(cols["hgb_noStruct"], cols["hgb_full"])
        d["ablation_noHist_minus_full"] = paired(cols["hgb_noHist"], cols["hgb_full"])
        d["paired_rule_minus_hgb_full"] = paired(cols["dom_rule"], cols["hgb_full"])
        d["paired_rule_minus_mlp_full"] = paired(cols["dom_rule"], cols["mlp_full"])
        d["paired_rule_minus_dagger"] = paired(cols["dom_rule"], cols["hgb_dagger"])
        if "request" in results:
            d["request_split_rule_minus_hgb_full"] = paired(cols["dom_rule"], sel(results["request"]["hgb_full"]["fwd"]))
        out["groups"][g] = d
    ms_rule = float(np.mean([b["rule_ms"] for b in base]))
    ms_h = float(np.mean(R["hgb_full"]["ms"])); ms_m = float(np.mean(R["mlp_full"]["ms"]))
    a = out["groups"]["all"]
    fwd_ms = 60.0
    out["cost"] = {"rule_ms_per_step": ms_rule, "hgb_ms_per_step": ms_h, "mlp_ms_per_step": ms_m, "forward_ms": fwd_ms,
                   "net_gain_ms_per_request_hgb": float(a["paired_rule_minus_hgb_full"]["mean"] * fwd_ms
                                                        - (ms_h - ms_rule) * a["fwd_hgb_full"]),
                   "note": "python feature+predict time on CPU; draft runs sequentially before each forward"}
    out["verdict_learned_beats_rule"] = bool(a["paired_rule_minus_main"]["ci95"][0] > 0)
    out["verdict_structure_contributes"] = bool(a["ablation_noStruct_minus_full"]["ci95"][0] > 0)
    out["per_target"] = [{"set": t["set"], "id": t["id"], "app": t["app"], **b, **{v: R[v]["fwd"][i] for v in R}}
                         for i, (t, b) in enumerate(zip(targets, base))]
    (OUT / "ranker.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    for g, d in out["groups"].items():
        print(g, json.dumps(d, default=float)[:2500], flush=True)
    print("cost", out["cost"], "verdicts", out["verdict_learned_beats_rule"], out["verdict_structure_contributes"])
    fig(out)


def fig(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    keys = ["std", "dom_rule", "hgb_full", "mlp_full", "hgb_dagger", "hgb_noStruct", "hgb_noHist", "top8oracle", "oracle"]
    cols = ["#2a78d6", "#1baf7a", "#eb6834", "#eda100", "#e87ba4", "#6250d6", "#008300", "#b9b8b2", "#d6d5cf"]
    gs = [g for g in ("all", "poc_greedy", "gate2_measured", "cands6", "gemma_approx") if g in out["groups"]]
    fig, axes = plt.subplots(1, len(gs), figsize=(3.2 * len(gs), 3.6))
    for ax, g in zip(axes, gs):
        d = out["groups"][g]
        vals = [d[f"fwd_{k}"] for k in keys]
        ax.bar(range(len(keys)), vals, color=cols, edgecolor="#fcfcfb")
        ax.set_xticks(range(len(keys)), keys, rotation=60, fontsize=7)
        p = d["paired_rule_minus_main"]
        ax.set_title(f"{g} n={d['n']}\nrule−learned {p['mean']:.1f} [{p['ci95'][0]:.1f},{p['ci95'][1]:.1f}]", loc="left", fontsize=8)
    axes[0].set_ylabel("forwards per request (mean)")
    fig.tight_layout(); fig.savefig(FIG / "ranker.png", dpi=150); plt.close(fig)


# =========================================================================== deployable ranker (numpy only)
FULL_COLS = STRUCT + LEXI + HIST


def train_mlp_np(rows, cols=FULL_COLS, epochs=6, seed=0):
    """Train the 2-layer MLP (pairwise logistic) with torch on CPU and return numpy weights."""
    import torch
    torch.set_num_threads(int(os.environ.get("SIM_THREADS", "4")))
    torch.manual_seed(seed)
    X = np.concatenate([F[:, cols] for F, A in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    pi, pj, off = [], [], 0
    for F, A in rows:
        n = len(A)
        for i in range(n):
            for j in range(n):
                if A[i] > A[j]:
                    pi.append(off + i); pj.append(off + j)
        off += n
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    pi, pj = torch.tensor(pi), torch.tensor(pj)
    net = torch.nn.Sequential(torch.nn.Linear(len(cols), 64), torch.nn.ReLU(), torch.nn.Linear(64, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(epochs):
        perm = torch.randperm(len(pi))
        for k in range(0, len(pi), 4096):
            b = perm[k:k + 4096]
            loss = torch.nn.functional.softplus(-(net(Xt[pi[b]]).squeeze(-1) - net(Xt[pj[b]]).squeeze(-1))).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    W = [p.detach().numpy().astype(np.float32) for p in net.parameters()]
    return {"W1": W[0].T.copy(), "b1": W[1], "W2": W[2].T.copy(), "b2": W[3], "W3": W[4].T.copy(), "b3": W[5],
            "mu": mu.astype(np.float32), "sd": sd.astype(np.float32), "cols": np.array(cols, np.int64)}


def load_weights(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def mlp_scores(F, w):
    x = (F[:, w["cols"]] - w["mu"]) / w["sd"]
    h = np.maximum(x @ w["W1"] + w["b1"], 0)
    h = np.maximum(h @ w["W2"] + w["b2"], 0)
    return (h @ w["W3"] + w["b3"]).ravel()


def rank_candidates(state, candidates_, weights):
    """state = (cx, L, pos, hit_n, (prev_src, prev_a, accs)); candidates_ = DOM top-8 list from candidates().
    Returns index of the chosen candidate (ties -> earlier = higher DOM-rule rank)."""
    cx, L, pos, hit_n, hist = state
    if len(candidates_) == 1:
        return 0
    F = featurize(cx, L, pos, candidates_, hit_n, hist)
    s = mlp_scores(F, weights)
    return int(np.argmax(s - 1e-6 * np.arange(len(s))))


class Annotator:
    """Incremental version of study_draft_sim.annotate (identical output)."""

    def __init__(self):
        self.text, self.inhtml, self.lx, self.k, self.pend = "", False, None, -1, ""

    def peek(self):
        return (self.lx.state() if self.inhtml and self.lx else None), (self.k if self.inhtml else -1)

    def push(self, s):
        a, r = self.peek()
        for ch in s:
            self.text += ch
            if not self.inhtml:
                if self.text.endswith("```html\n") or self.text.endswith("```html"):
                    self.pend = "open"
                if self.pend == "open" and ch == "\n" and self.text.rstrip("\n").endswith("```html"):
                    self.inhtml, self.lx, self.k, self.pend = True, S.Lexer(), self.k + 1, ""
            else:
                if self.text.endswith("```"):
                    self.inhtml = False
                    continue
                self.lx.feed(ch)
        if len(self.text) > 64:
            self.text = self.text[-64:]
        return a, r


class RankerDrafter:
    """Drop-in drafter for poc_rnsd.py.

        dr = RankerDrafter(prompt_ids, decode_one, weights, max_new=2048)   # decode_one(id) -> str
        draft = dr.draft(gen_ids, gamma)      # call before every verify forward with ALL generated ids so far
    Same candidate generation (DOM top-8), features and history as the simulator; weights=None -> DOM rule.
    """

    def __init__(self, prompt_ids, decode_one, weights=None, max_new=2048):
        self.dec, self.w, self.max_new = decode_one, weights, max_new
        self.P = len(prompt_ids)
        self.ctx = np.zeros(self.P + max_new + 1, np.int64)
        self.ctx[:self.P] = prompt_ids
        self.n = self.P
        self.ann, self.reg = [], []
        self.an = Annotator()
        for t in prompt_ids:
            a, r = self.an.push(self.dec(int(t)))
            self.ann.append(a); self.reg.append(r)
        self.rstart = {}
        for i, r in enumerate(self.reg):
            self.rstart.setdefault(r, i)
        self.rstart.setdefault(-2, self.P)
        self.idx = S.Index(3)
        self.prev_src, self.prev_a, self.accs, self.last_k, self.last_len = None, 0, [], None, None
        self.t_ms = []

    def _append(self, toks):
        for t in toks:
            a, _ = self.an.push(self.dec(int(t)))
            self.ann.append(a); self.reg.append(-2)
            self.ctx[self.n] = t; self.n += 1

    def draft(self, gen_ids, gamma=GAMMA):
        t0 = time.perf_counter()
        new = gen_ids[self.n - self.P:]
        if self.last_len is not None:
            a = len(gen_ids) - self.last_len - 1
            self.prev_a = max(0, a) if self.last_k else 0
            self.accs.append(self.prev_a)
        self._append(new)
        L, pos = self.n, len(gen_ids)
        self.ann_cur = self.an.peek()[0]
        g = min(gamma, self.max_new - pos - 1)
        # expose Ctx-like view for candidates()/featurize()
        view = _View(self)
        self.idx.extend(self.ctx, L)
        hit_n, cands, _ = candidates(view, L, self.idx)
        src = None
        if cands:
            j = 0 if self.w is None else rank_candidates((view, L, pos, hit_n, (self.prev_src, self.prev_a, self.accs)), cands, self.w)
            src = cands[j]["c"]
        else:
            cur = self.ann_cur
            if cur is not None:
                best = None
                for c in range(L - 1, 0, -1):
                    if self.ann[c] is None:
                        continue
                    sc = S.sim_score(self.ann[c], cur)
                    if best is None or sc > best:
                        best, src = sc, c
        d = [] if src is None or g <= 0 else self.ctx[src:min(L, src + g)].tolist()
        self.prev_src, self.last_k, self.last_len = src, len(d), len(gen_ids)
        self.t_ms.append((time.perf_counter() - t0) * 1000)
        return d


class _View:
    """Ctx-compatible view (ctx, ann incl. current position, reg, rstart) over a RankerDrafter."""

    def __init__(self, dr):
        self.ctx = dr.ctx
        self.ann = _AnnList(dr)
        self.reg = _RegList(dr)
        self.rstart = dr.rstart


class _AnnList:
    def __init__(self, dr):
        self.dr = dr

    def __getitem__(self, i):
        return self.dr.ann_cur if i == self.dr.n else self.dr.ann[i]


class _RegList:
    def __init__(self, dr):
        self.dr = dr

    def __getitem__(self, i):
        return -2 if i >= self.dr.n else self.dr.reg[i]


def export():
    """Cross-fit export: app-level GroupKFold(5) MLP weights + fold map + pilot-50 predicted forwards."""
    S.guard(int(os.environ["SIM_THREADS"]))
    from transformers import AutoTokenizer
    from sklearn.model_selection import GroupKFold
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    cache = {}

    def dec(i):
        if i not in cache:
            cache[i] = tok.decode([i])
        return cache[i]
    raw, targets = load_all(tok)
    cxs = [Ctx(t, dec) for t in targets]
    rows_rule = [replay(cx, "rule", collect=True)[1] for cx in cxs]
    apps = np.array([t["app"] for t in targets])
    gkf = GroupKFold(n_splits=5)
    app_fold = {}
    for f, (tr, te) in enumerate(gkf.split(np.zeros(len(cxs)), groups=apps)):
        w = train_mlp_np([r for i in tr for r in rows_rule[i]])
        np.savez(OUT / f"ranker_mlp_fold{f}.npz", **w)
        for i in te:
            app_fold[apps[i]] = f
        print("fold", f, "saved", flush=True)
    pilot = load_jsonl(ROOT / "data" / "processed" / "pilot.jsonl")
    fmap = {r["id"]: app_fold.get(r["app"]) for r in pilot}
    (OUT / "ranker_fold_map.json").write_text(json.dumps(fmap, indent=1), encoding="utf-8")
    # pilot-50 predicted forwards on the gate2 PLD outputs (rep 0) with the cross-fit model; check drafter == replay
    W = {f: load_weights(OUT / f"ranker_mlp_fold{f}.npz") for f in range(5)}
    mid = ROOT / "data" / "processed" / "mid"
    pred = {}
    seen = set()
    tg = [t for t in raw if t["set"] == "g2pld"]
    for t in tg:
        if t["id"] in seen:
            continue
        seen.add(t["id"])
        f = fmap[t["id"]]
        cx = Ctx(t, dec)
        nf_m, _, ms = replay(cx, lambda F, w=W[f]: int(np.argmax(mlp_scores(F, w) - 1e-6 * np.arange(len(F)))))
        nf_r = replay(cx, "rule")[0]
        nf_s, _ = S.run_policy("std", cx.prompt, cx.y, cx.ann, cx.reg)
        # drafter path (as poc_rnsd would call it) on the same target
        dr = RankerDrafter(t["prompt"], dec, W[f])
        y = t["y"]; pos, nf = 1, 1
        while pos < len(y):
            d = dr.draft(y[:pos])
            a = 0
            while a < len(d) and pos + a < len(y) and d[a] == y[pos + a]:
                a += 1
            nf += 1; pos += a + 1
        pred[t["id"]] = {"fold": f, "fwd_mlp_crossfit": nf_m, "fwd_drafter": nf, "fwd_dom_rule": nf_r, "fwd_std": nf_s,
                         "T": len(y), "drafter_ms_per_step_mean": float(np.mean(dr.t_ms)),
                         "drafter_ms_per_step_p95": float(np.percentile(dr.t_ms, 95)), "target": "gate2 PLD output rep0"}
    agg = {k: float(np.mean([v[k] for v in pred.values()])) for k in ("fwd_mlp_crossfit", "fwd_drafter", "fwd_dom_rule", "fwd_std",
                                                                      "drafter_ms_per_step_mean", "drafter_ms_per_step_p95")}
    agg["drafter_equals_replay"] = int(sum(v["fwd_mlp_crossfit"] == v["fwd_drafter"] for v in pred.values()))
    agg["n"] = len(pred)
    agg["paired_rule_minus_mlp"] = paired([v["fwd_dom_rule"] for v in pred.values()], [v["fwd_mlp_crossfit"] for v in pred.values()])
    (OUT / "ranker_pred_p50.json").write_text(json.dumps({"summary": agg, "per_request": pred}, indent=1), encoding="utf-8")
    print(agg)


if __name__ == "__main__":
    if "--export" in sys.argv:
        export()
    else:
        main()
