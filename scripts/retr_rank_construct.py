"""Ranking vs construction gap of draft-position choice (greedy replay, CPU, 2 threads, BelowNormal).

Imports scripts/study_draft_sim.py and scripts/study_suffix_sim.py (not modified). gamma=16, acceptance = LCP + 1 bonus,
n_fwd = 1 prefill + #verify steps. In greedy replay the context at output position pos is fixed, so per-pos
quantities are computed once and each myopic oracle is a sequence of jumps pos -> pos + a + 1.

Layers (myopic oracle restricted to the layer = take the candidate with the largest true acceptance):
  SD   : study_suffix_sim.run_sd (lambda=0, K=1), executable baseline
  O1   : lexical candidates SD can see = every start c < L whose previous token equals the current last token
         (suffix match n>=1), draft = ctx[c:c+16]
  O2   : any start position in ctx (= study "oracle")
  O3   : O2 + generated structural drafts not present in ctx:
         (i)  closing-tag sequence from the full open-tag stack (innermost first), tokenized
         (ii) structure jumps: top-8 element-start positions in ctx ranked by structural agreement
              (study_draft_sim.sim_score) with the current position, used without any suffix match.
              These copy from ctx, so they are a subset of O2 and add nothing to O3 by construction;
              they matter for the executable policy below.
  SD+jump (executable, no oracle): SD, but when the longest suffix match m <= MSHORT (0 = none), draft from
         the top-1 structure-jump position instead (16 tokens). MSHORT in {0,1,2}; main = 1 (fixed a priori).
Gap decomposition of std -> O2: std -> SD, SD -> O1 (ranking), O1 -> O2 (construction).
Usage: .venv/Scripts/python.exe -I scripts/retr_rank_construct.py   ->  runs/retrieval/rank_construct.json
"""
from __future__ import annotations

import os

os.environ["SIM_THREADS"] = "2"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_k] = "2"

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import study_draft_sim as S  # noqa: E402
import study_suffix_sim as SS  # noqa: E402
from retr_seq_oracle import accept_all, boot  # noqa: E402

OUT = ROOT / "runs" / "retrieval" / "rank_construct.json"
G = S.GAMMA
NJUMP = 8
MSHORTS = (0, 1, 2)
MAIN_MSHORT = 1


def annotate_full(strs):
    """Same as study_draft_sim.annotate, plus the full open-tag stack and lexer mode before each token."""
    ann, stk = [], []
    text = ""
    inhtml, lx, pend = False, None, ""
    for s in strs:
        ann.append(lx.state() if inhtml and lx else None)
        stk.append((tuple(lx.stack), lx.mode) if inhtml and lx else None)
        for ch in s:
            text += ch
            if not inhtml:
                if text.endswith("```html\n") or text.endswith("```html"):
                    pend = "open"
                if pend == "open" and ch == "\n" and text.rstrip("\n").endswith("```html"):
                    inhtml, lx, pend = True, S.Lexer(), ""
            else:
                if text.endswith("```"):
                    inhtml = False
                    continue
                lx.feed(ch)
        if len(text) > 64:
            text = text[-64:]
    return ann, stk


def lcp(d, y, pos, g):
    a = 0
    while a < min(g, len(d)) and pos + a < len(y) and d[a] == y[pos + a]:
        a += 1
    return a


def sample(t, tok, close_cache):
    prompt = np.array(t["prompt"], np.int64)
    y = np.array(t["y"], np.int64)
    P, T = len(prompt), len(y)
    ctx = np.concatenate([prompt, y]).astype(np.int64)
    max_new = t["max_new"]
    strs = [tok.decode([i]) for i in list(prompt) + list(y)]
    ann, stk = annotate_full(strs)
    ann.append(None)
    stk.append(None)
    # element-start positions (token begins a start tag) usable as structure-jump targets
    starts = [c for c in range(len(ctx)) if ann[c] is not None and strs[c].lstrip().startswith("<")
              and not strs[c].lstrip().startswith("</")]

    def gcap(pos):
        return min(G, max_new - pos - 1)

    a1 = np.zeros(T + 1, np.int64); a2 = np.zeros(T + 1, np.int64); a3 = np.zeros(T + 1, np.int64)
    aclose = np.zeros(T + 1, np.int64); ajump = np.zeros(T + 1, np.int64)
    jump_top = {}
    for pos in range(1, T):
        L, g = P + pos, gcap(pos)
        cand, lens = accept_all(ctx, L, y, pos, g)
        a2[pos] = lens.max() if len(lens) else 0
        if len(lens):
            ok = (cand >= 1) & (ctx[np.maximum(cand - 1, 0)] == ctx[L - 1])
            a1[pos] = lens[ok].max() if ok.any() else 0
        # (i) closing-tag sequence
        st = stk[L]
        ac = 0
        if st is not None and st[0] and st[1] == "text":
            key = st[0]
            if key not in close_cache:
                s = "".join(f"</{n}>" for n in reversed(key))
                close_cache[key] = tok(s, add_special_tokens=False)["input_ids"]
            ac = lcp(close_cache[key], y, pos, g)
        aclose[pos] = ac
        # (ii) structure jumps
        cur = ann[L]
        js = []
        if cur is not None:
            pool = [c for c in starts if c < L - 1]
            sc = [S.sim_score(ann[c], cur) for c in pool]
            order = sorted(range(len(pool)), key=lambda i: (-sc[i], -pool[i]))[:NJUMP]  # ties: most recent
            js = [pool[i] for i in order]
        jump_top[pos] = js
        ajump[pos] = max([S.accept_len(ctx, c, L, y, pos, g) for c in js], default=0)
        a3[pos] = max(a2[pos], ac, ajump[pos])

    def run(av):
        pos, nf = 1, 1
        while pos < T:
            pos += int(av[pos]) + 1
            nf += 1
        return nf

    def run_o1sd(sd_len):
        """O1 split: best of SD's own candidate set (occurrences of the longest suffix, last 400), evaluated
        only on visited positions (incremental index as in run_sd). sd_len: cap at SD's draft length."""
        ix = SS.SuffixIndex()
        pos, nf = 1, 1
        t_start = time.time()
        while pos < T:
            if time.time() - t_start > 20:  # degenerate repetitive output: skip this diagnostic for the sample
                return None
            L, g = P + pos, gcap(pos)
            ix.extend(ctx, L)
            m, occ = ix.longest(ctx, L)
            a = 0
            if m:
                occ = np.array(occ[-400:])
                cand, lens = accept_all(ctx, L, y, pos, g)
                if len(lens):
                    hit = np.isin(cand, occ)
                    a = int(lens[hit].max()) if hit.any() else 0
                if sd_len:
                    a = min(a, min(SS.GMAX, max_new - pos - 1, max(1, math.ceil(SS.ALPHA * m))))
            pos += a + 1
            nf += 1
        return nf

    res = {"T": T, "fwd_O1a_SDset_SDlen": run_o1sd(True), "fwd_O1b_SDset_g16": run_o1sd(False), "fwd_O1": run(a1), "fwd_O2": run(a2), "fwd_O3": run(a3)}
    # context of steps where the O3 extra draft beats O2 on the O3 trajectory
    o3ctx = {}
    pos = 1
    while pos < T:
        if a3[pos] > a2[pos]:
            k = SS.cat_of(pos, P, ann, strs)
            o3ctx[k] = o3ctx.get(k, 0) + 1
        pos += int(a3[pos]) + 1
    res["o3_gain_ctx"] = o3ctx
    res["o3_close_gain_steps"] = int(sum(1 for p in range(1, T) if aclose[p] > a2[p]))

    # ---------------- SD (re-implemented loop identical to SS.run_sd with lam=0, K=1) + structure jump
    def run_sd_jump(mshort):
        idx = SS.SuffixIndex()
        pos, nf = 1, 1
        used, gain_ctx, extra = 0, {}, 0
        while pos < T:
            L = P + pos
            idx.extend(ctx, L)
            m, occ = idx.longest(ctx, L)
            a_sd = 0
            if m:
                occ = occ[-400:]
                glen = min(SS.GMAX, max_new - pos - 1, max(1, math.ceil(SS.ALPHA * m)))
                drafts = SS.paths(ctx, L, occ, [1.0] * len(occ), glen, 1)
                for d in drafts:
                    a_sd = max(a_sd, lcp(d, y, pos, len(d)))
            a = a_sd
            if mshort is not None and m <= mshort and jump_top.get(pos):
                c = jump_top[pos][0]
                a = S.accept_len(ctx, c, L, y, pos, gcap(pos))
                used += 1
                if a > a_sd:
                    k = SS.cat_of(pos, P, ann, strs)
                    gain_ctx[k] = gain_ctx.get(k, 0) + 1
                    extra += a - a_sd
            nf += 1
            pos += a + 1
        return nf, used, gain_ctx, extra

    res["fwd_SD"] = run_sd_jump(None)[0]
    for ms in MSHORTS:
        nf, used, gc, ex = run_sd_jump(ms)
        res[f"fwd_SDjump_m{ms}"] = nf
        res[f"SDjump_m{ms}_used"] = used
        res[f"SDjump_m{ms}_gain_ctx"] = gc
        res[f"SDjump_m{ms}_extra_tokens"] = ex
    res["study_SD"] = SS.run_sd(prompt, y, ann, 0.0)[0]  # validation of the re-implemented loop
    return res


def main():
    S.guard(2)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    sim = json.loads((ROOT / "runs" / "study" / "draft_sim.json").read_text(encoding="utf-8"))
    std = {(r["set"], r["id"]): r["fwd_std"] for r in sim["rows"]}
    targets = [t for t in S.load_targets(tok) if t["set"] != "C2pld"]
    targets.sort(key=lambda t: (t["set"].startswith("gemma"), len(t["y"])))
    print("targets", len(targets), flush=True)
    rows, cc, t0 = [], {}, time.time()
    for i, t in enumerate(targets):
        r = sample(t, tok, cc)
        r.update({"set": t["set"], "id": t["id"], "fwd_std": std[(t["set"], t["id"])]})
        rows.append(r)
        print(f"[{i + 1}/{len(targets)}] {t['set']} {t['id']} T={r['T']} std={r['fwd_std']} SD={r['fwd_SD']}"
              f"(study {r['study_SD']}) O1={r['fwd_O1']} O2={r['fwd_O2']} O3={r['fwd_O3']} "
              f"SDjump1={r['fwd_SDjump_m1']} {time.time() - t0:.0f}s rss={S.check_rss():.2f}", flush=True)
        if (i + 1) % 10 == 0 or i + 1 == len(targets):
            OUT.write_text(json.dumps({"rows": rows, "summary": summarize(rows)}, indent=1), encoding="utf-8")
    print(json.dumps(summarize(rows), indent=1)[:8000])


def summarize(rows):
    out = {}
    groups = {"poc_greedy": [r for r in rows if not r["set"].startswith("gemma")],
              "gemma_fs(approx)": [r for r in rows if r["set"].startswith("gemma")]}
    keys = ["fwd_std", "fwd_SD", "fwd_O1a_SDset_SDlen", "fwd_O1b_SDset_g16", "fwd_O1", "fwd_O2", "fwd_O3"] + [f"fwd_SDjump_m{m}" for m in MSHORTS]
    for g, rs in groups.items():
        if not rs:
            continue
        d = {"n": len(rs), "SD_matches_study": int(sum(r["fwd_SD"] == r["study_SD"] for r in rs))}
        for k in keys:
            d[k] = float(np.mean([r[k] for r in rs if r[k] is not None]))
        split = [r for r in rs if r["fwd_O1a_SDset_SDlen"] is not None and r["fwd_O1b_SDset_g16"] is not None]
        d["n_split"] = len(split)
        sm = {k: float(np.mean([r[k] for r in split])) for k in ("fwd_std", "fwd_SD", "fwd_O1a_SDset_SDlen",
                                                                   "fwd_O1b_SDset_g16", "fwd_O1", "fwd_O2")}
        d["split_subset_means"] = sm
        tot = d["fwd_std"] - d["fwd_O2"]
        d["decomp_std_to_O2"] = {"total": tot,
                                 "std->SD": (d["fwd_std"] - d["fwd_SD"]) / tot,
                                 "ranking SD->O1": (d["fwd_SD"] - d["fwd_O1"]) / tot,
                                 "construction O1->O2": (d["fwd_O1"] - d["fwd_O2"]) / tot}
        tots = sm["fwd_std"] - sm["fwd_O2"]
        d["decomp_ranking_split(subset, share of std->O2)"] = {
            "path choice within SD set (SD->O1a)": (sm["fwd_SD"] - sm["fwd_O1a_SDset_SDlen"]) / tots,
            "SD length cap (O1a->O1b)": (sm["fwd_O1a_SDset_SDlen"] - sm["fwd_O1b_SDset_g16"]) / tots,
            "shorter-suffix occurrences (O1b->O1)": (sm["fwd_O1b_SDset_g16"] - sm["fwd_O1"]) / tots}
        tot2 = d["fwd_SD"] - d["fwd_O2"]
        d["decomp_SD_to_O2"] = {"total": tot2, "ranking": (d["fwd_SD"] - d["fwd_O1"]) / tot2,
                                "construction": (d["fwd_O1"] - d["fwd_O2"]) / tot2}
        for a, b in (("fwd_SD", "fwd_O1"), ("fwd_O1", "fwd_O2"), ("fwd_O2", "fwd_O3")):
            x = [r[a] - r[b] for r in rs]
            d[f"paired_{a[4:]}-{b[4:]}"] = {"mean": float(np.mean(x)), "ci95": boot(x), "median": float(np.median(x))}
        for m in MSHORTS:
            x = [r["fwd_SD"] - r[f"fwd_SDjump_m{m}"] for r in rs]  # > 0: jump policy uses fewer forwards
            d[f"SD-SDjump_m{m}"] = {"mean": float(np.mean(x)), "ci95": boot(x), "median": float(np.median(x)),
                                    "red_mean": float(np.mean([1 - r[f"fwd_SDjump_m{m}"] / r["fwd_SD"] for r in rs])),
                                    "red_total": float(1 - sum(r[f"fwd_SDjump_m{m}"] for r in rs) / sum(r["fwd_SD"] for r in rs)),
                                    "n_better": int(sum(v > 0 for v in x)), "n_worse": int(sum(v < 0 for v in x)),
                                    "steps_used": int(sum(r[f"SDjump_m{m}_used"] for r in rs))}
            gc = {}
            for r in rs:
                for k, v in r[f"SDjump_m{m}_gain_ctx"].items():
                    gc[k] = gc.get(k, 0) + v
            d[f"SDjump_m{m}_gain_ctx"] = dict(sorted(gc.items(), key=lambda kv: -kv[1]))
        oc = {}
        for r in rs:
            for k, v in r["o3_gain_ctx"].items():
                oc[k] = oc.get(k, 0) + v
        d["O3_gain_ctx"] = dict(sorted(oc.items(), key=lambda kv: -kv[1]))
        d["verdict_construction_gt_ranking"] = bool(d["fwd_O1"] - d["fwd_O2"] > d["fwd_SD"] - d["fwd_O1"])
        out[g] = d
    return out


if __name__ == "__main__":
    main()
