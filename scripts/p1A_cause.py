"""P1-A cause analysis (CPU, BelowNormal, 2 cores): why does the learned ranker lose to the DOM rule under some
transforms?  Replays the RANKER trajectory on the measured RANKER outputs (exact: replay == measured forwards) and
records per step: n-gram hit availability (hit_n), #candidates, accepted length of the MLP choice vs the DOM-rule
choice (top-1) at the SAME state, top-8 oracle, structural feature values (FEATS 0..11), and the HTML context of
steps where MLP < rule.
Sets: orig (gate2b, rep 0) and each transform in runs/poc/p1A/<t> (prompt = transformed exemplars).
Usage: .venv/Scripts/python.exe -I scripts/p1A_cause.py fmt [dom cls]   -> runs/poc/p1A/cause_<sets>.md / .json
"""
from __future__ import annotations

import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
try:
    p = psutil.Process()
    p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    p.cpu_affinity([0, 1])
except Exception:
    pass
os.environ["SIM_THREADS"] = "2"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402


def load_mod(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


R = load_mod("poc_rnsd")
SR = R._ranker_mod()
S = R._sim()
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized")
tc = {}


def dec(i):
    if i not in tc:
        tc[i] = tok.decode([i])
    return tc[i]


fmap = json.load(open(ROOT / "runs" / "study" / "ranker_fold_map.json", encoding="utf-8"))
W = {f: SR.load_weights(ROOT / "runs" / "study" / f"ranker_mlp_fold{f}.npz") for f in set(fmap.values())}
GAMMA = 16


def analyze_set(name, run_dir, html_file):
    prompts = {p["row"]["id"]: p for p in R.build_prompts(50, exemplar_html=html_file)}
    rows = [json.loads(l) for l in open(run_dir / "E2B_ranker.jsonl", encoding="utf-8")]
    rows = [r for r in rows if "id" in r and r.get("rep", 0) == 0]
    steps, worse = [], []
    for r in rows:
        ids = list(tok.apply_chat_template(prompts[r["id"]]["msgs"], add_generation_prompt=True, tokenize=True,
                                           return_dict=True)["input_ids"])
        cx = SR.Ctx({"prompt": ids, "y": r["gen_ids"], "max_new": 2048}, dec)
        w = W[fmap[r["id"]]]
        idx = S.Index(3)
        y, P = cx.y, cx.P
        pos, nf = 1, 1
        prev_src, prev_a, accs = None, 0, []
        while pos < len(y):
            L = P + pos
            g = min(GAMMA, 2048 - pos - 1)
            idx.extend(cx.ctx, L)
            hit_n, cands, _ = SR.candidates(cx, L, idx)
            a, src = 0, None
            if cands:
                A = [S.accept_len(cx.ctx, cd["c"], L, y, pos, g) for cd in cands]
                j = SR.rank_candidates((cx, L, pos, hit_n, (prev_src, prev_a, accs)), cands, w)
                F = SR.featurize(cx, L, pos, cands, hit_n, (prev_src, prev_a, accs))
                src, a = cands[j]["c"], A[j]
                st = {"hit_n": hit_n, "ncand": len(cands), "a_mlp": a, "a_rule": A[0], "a_or": max(A), "j": j,
                      "F_all": F[:, :12].mean(0).tolist(), "F_ch": F[j, :12].tolist(),
                      "lex": (cx.ann[L][0] if cx.ann[L] else "none")}
                steps.append(st)
                if a < A[0] and len(worse) < 400:
                    worse.append({"id": r["id"], "pos": pos, "lex": st["lex"], "a_mlp": a, "a_rule": A[0],
                                  "ctx": "".join(cx.strs[max(P, L - 12):L])[-80:],
                                  "rule_next": "".join(cx.strs[cands[0]["c"]:cands[0]["c"] + 6]),
                                  "mlp_next": "".join(cx.strs[src:src + 6]), "truth": "".join(cx.strs[L:L + 6])})
            else:
                steps.append({"hit_n": 0, "ncand": 0, "a_mlp": 0, "a_rule": 0, "a_or": 0, "j": -1, "lex": "nohit"})
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
                        steps[-1]["a_mlp"] = steps[-1]["a_rule"] = a
            nf += 1
            prev_src, prev_a = src, a
            accs.append(a)
            pos += a + 1
    return steps, worse


def summ(steps):
    n = len(steps)
    wc = [s for s in steps if s["ncand"] > 0]
    multi = [s for s in wc if s["ncand"] > 1]
    out = {"steps": n, "hit_rate": len(wc) / n, "hit_n3": sum(s["hit_n"] == 3 for s in steps) / n,
           "mean_ncand": statistics.mean(s["ncand"] for s in steps),
           "multi_cand_share": len(multi) / n,
           "a_mlp": statistics.mean(s["a_mlp"] for s in steps), "a_rule_same_state": statistics.mean(s["a_rule"] for s in steps),
           "mlp_lt_rule": sum(s["a_mlp"] < s["a_rule"] for s in multi) / max(1, len(multi)),
           "mlp_gt_rule": sum(s["a_mlp"] > s["a_rule"] for s in multi) / max(1, len(multi)),
           "mlp_not_top1": sum(s["j"] > 0 for s in multi) / max(1, len(multi)),
           "oracle_gap": statistics.mean(s["a_or"] - s["a_mlp"] for s in wc) if wc else 0}
    if multi:
        out["F_all_mean"] = np.mean([s["F_all"] for s in multi], 0).round(3).tolist()
    return out


def main():
    sets = sys.argv[1:] or ["fmt"]
    res = {"orig": analyze_set("orig", ROOT / "runs" / "poc" / "gate2b", None)}
    for t in sets:
        res[t] = analyze_set(t, ROOT / "runs" / "poc" / "p1A" / t, ROOT / "runs" / "retrieval" / f"pilot_exemplar_html_{t}.json")
    feats = SR.FEATS[:12]
    lines = ["# P1-A cause analysis (RANKER trajectory replay on measured RANKER outputs)", "",
             "| set | steps | n-gram hit rate | hit n=3 | mean #cand | multi-cand share | acc MLP | acc rule (same state) | MLP<rule | MLP>rule | MLP != top1 | oracle gap |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for k, (st, _) in res.items():
        s = summ(st)
        summary[k] = s
        lines.append(f"| {k} | {s['steps']} | {s['hit_rate']:.3f} | {s['hit_n3']:.3f} | {s['mean_ncand']:.2f} | {s['multi_cand_share']:.3f} | "
                     f"{s['a_mlp']:.2f} | {s['a_rule_same_state']:.2f} | {s['mlp_lt_rule']:.3f} | {s['mlp_gt_rule']:.3f} | "
                     f"{s['mlp_not_top1']:.3f} | {s['oracle_gap']:.2f} |")
    lines += ["", "Structural feature means over candidates at multi-candidate steps (shift vs orig):", "",
              "| feature | " + " | ".join(res) + " |", "|---|" + "---|" * len(res)]
    for i, f in enumerate(feats):
        lines.append(f"| {f} | " + " | ".join(f"{summary[k].get('F_all_mean', [float('nan')] * 12)[i]:.3f}" for k in res) + " |")
    for k, (st, wo) in res.items():
        if k == "orig":
            continue
        lex = {}
        for x in wo:
            lex[x["lex"]] = lex.get(x["lex"], 0) + 1
        lines += ["", f"## {k}: steps where MLP choice < DOM-rule choice (n={len(wo)}), lexical state counts: {json.dumps(lex)}", ""]
        for x in wo[:8]:
            lines.append(f"- `{x['id']}` pos {x['pos']} lex={x['lex']} a_mlp={x['a_mlp']} a_rule={x['a_rule']} | ctx `{x['ctx']!r}` | "
                         f"truth `{x['truth']!r}` | rule `{x['rule_next']!r}` | mlp `{x['mlp_next']!r}`")
    tag = "_".join(sets)
    (ROOT / "runs" / "poc" / "p1A" / f"cause_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "runs" / "poc" / "p1A" / f"cause_{tag}.json").write_text(json.dumps({"summary": summary}, indent=1))
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
