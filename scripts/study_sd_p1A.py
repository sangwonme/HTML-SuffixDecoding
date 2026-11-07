"""SD vs SD+learned under HTML representation changes (P1-A replay, CPU).

Targets: P1-A actual PLD outputs per perturbation (runs/poc/p1A/<cond>/E2B_pld.jsonl, rep 0) and, for "orig",
gate2_p50 PLD outputs.  Local context = the request's (perturbed) exemplar prompt + output.  Global cache = outputs
(PLD/DOM/ranker runs of the same condition, deduplicated) of OTHER apps' requests (fold = ranker_fold_map, app-level).
Policies: PLD, DOM rule, learned MLP (cross-fit, original-format weights), formal SD f1/f16, SD + learned
(frequency + DOM features; HGB trained on ORIGINAL-format data per app fold, no retraining).

  .venv/Scripts/python.exe -I scripts/study_sd_p1A.py [cond ...]
Output: runs/study/sd_p1A.json
"""
from __future__ import annotations

import os, sys, pathlib, json, time
os.environ.setdefault("SIM_THREADS", "2")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

import study_draft_sim as S
import study_ranker as R
import study_sd_formal as F
from study_common import OUT, ROOT, load_jsonl

CONDS = ["orig", "fmt", "css", "dom", "cls"]
COLS = [int(x) for x in os.environ.get("SDL_COLS", "0,1,2,3,4,5,6,7,8,9").split(",")]   # 0-6 frequency, 7 sim max, 8 sim mean, 9 high-sim share
FACS = [float(x) for x in os.environ.get("SDL_FACTORS", "1,16").split(",")]


def main():
    S.guard(int(os.environ["SIM_THREADS"]))
    conds = sys.argv[1:] or CONDS
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    cache = {}

    def dec(i):
        if i not in cache:
            cache[i] = tok.decode([i])
        return cache[i]
    pilot = {r["id"]: r for r in load_jsonl(ROOT / "data" / "processed" / "pilot.jsonl")}
    mid = ROOT / "data" / "processed" / "mid"
    mid_pool = {r["id"]: r for r in load_jsonl(mid / "train_pool.jsonl")}
    mid_pilot = {r["id"]: r for r in load_jsonl(mid / "pilot.jsonl")}
    html = {r["id"]: r["gt_html_flow"] for r in load_jsonl(ROOT / "data" / "processed" / "train_pool.jsonl")}
    fmap = json.loads((OUT / "ranker_fold_map.json").read_text(encoding="utf-8"))
    W = {f: R.load_weights(OUT / f"ranker_mlp_fold{f}.npz") for f in range(5)}
    t0 = time.time()

    # ---- SD+learned models trained on ORIGINAL-format data, per pilot-app fold (fmap folds)
    raw, targets = R.load_all(tok)
    train_t = [t for t in targets if t["set"] in ("C1", "C0", "C4g", "C7g", "g2pld", "g2dom")]
    tcx = [R.Ctx(t, dec) for t in train_t]
    tresp = []
    for cx in tcx:
        yt = cx.y.tolist(); ann, _ = S.annotate([dec(i) for i in yt]); ann.append(None); tresp.append(yt + [-1])
        tresp[-1] = (yt, ann)
    models = {}
    for f in range(5):
        test_apps = {pilot[q]["app"] for q, ff in fmap.items() if ff == f}
        tr = [i for i, t in enumerate(train_t) if t["app"] not in test_apps]
        toks, anns = [], []
        for i in tr:
            yt, ann = tresp[i]; toks += yt + [-1]; anns += ann[:len(yt)] + [None]
        glob = F.Seq(toks, anns)
        for fac in FACS:
            rows = []
            for i in tr:
                rows += F.run(tcx[i], glob, fac, "path", collect=True)[2]
            models[(f, fac)] = F.fit(rows, COLS)
        print("trained fold", f, f"{time.time() - t0:.0f}s", flush=True)

    def prompt_for(qid, ex_ids, ov):
        msgs = [{"role": "system", "content": S.SYSTEM}]
        for e in ex_ids:
            msgs.append({"role": "user", "content": mid_pool[e]["query_mid"]})
            msgs.append({"role": "assistant", "content": "```html\n" + ov.get(e, html[e]) + "\n```"})
        msgs.append({"role": "user", "content": mid_pilot[qid]["query_mid"]})
        return list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])

    out = {"seed_policy": "perturbations applied per exemplar with seed = exemplar index in the pilot-50 exemplar list "
                          "(scripts/study_perturb_exemplars.py); for cls each exemplar has its own random class names, "
                          "and the model output invents/copies its own", "conds": {}}
    for cond in conds:
        if cond == "orig":
            ov, base = {}, ROOT / "runs" / "poc" / "gate2_p50"
        elif cond.startswith("B_"):   # P1-B: draft source perturbed, outputs (and global cache) original
            ov = json.loads((ROOT / "runs" / "retrieval" / f"pilot_exemplar_html_{cond[2:]}.json").read_text(encoding="utf-8"))
            base = ROOT / "runs" / "poc" / "gate2_p50"
        else:
            ov = json.loads((ROOT / "runs" / "retrieval" / f"pilot_exemplar_html_{cond}.json").read_text(encoding="utf-8"))
            base = ROOT / "runs" / "poc" / "p1A" / cond
        if not (base / "E2B_pld.jsonl").exists():
            print("missing", cond); continue
        tg = {}
        for r in load_jsonl(base / "E2B_pld.jsonl"):
            if "id" in r and r.get("rep", 0) == 0 and r["id"] not in tg:
                tg[r["id"]] = r
        pool = []   # global cache material: all outputs of this condition
        seen = set()
        for fn in ("E2B_pld.jsonl", "E2B_dom.jsonl", "E2B_ranker.jsonl"):
            if (base / fn).exists():
                for r in load_jsonl(base / fn):
                    if "id" in r and "gen_ids" in r:
                        k = tuple(r["gen_ids"])
                        if k not in seen:
                            seen.add(k); pool.append((pilot[r["id"]]["app"], r["gen_ids"]))
        pool_ann = [(app, yt, (lambda a: a + [None])(S.annotate([dec(i) for i in yt])[0])) for app, yt in pool]
        rows = []
        pok = 0
        for qid, r in tg.items():
            p = prompt_for(qid, r["example_ids"], ov)
            pok += len(p) == r["prompt_tokens"]   # for B_* the draft prompt differs from the model prompt by design
            cx = R.Ctx({"prompt": p, "y": list(r["gen_ids"]), "max_new": 2048}, dec)
            f = fmap[qid]; app = pilot[qid]["app"]
            test_apps = {pilot[q]["app"] for q, ff in fmap.items() if ff == f}
            toks, anns = [], []
            for a, yt, ann in pool_ann:
                if a not in test_apps:
                    toks += list(yt) + [-1]; anns += ann[:len(yt)] + [None]
            glob = F.Seq(toks, anns)
            w = W[f]
            row = {"id": qid, "T": len(r["gen_ids"]), "measured_pld_fwd": r["n_fwd"],
                   "std": S.run_policy("std", cx.prompt, cx.y, cx.ann, cx.reg)[0],
                   "dom_rule": R.replay(cx, "rule")[0],
                   "mlp": R.replay(cx, lambda Fm, w=w: int(np.argmax(R.mlp_scores(Fm, w) - 1e-6 * np.arange(len(Fm)))))[0],
                   "sd_f1": F.run(cx, glob, 1.0, "path")[0], "sd_f16": F.run(cx, glob, 16.0, "path")[0],
                   "sdl_f1": F.run(cx, glob, 1.0, "learned", chooser=models[(f, 1.0)])[0] if (f, 1.0) in models else -1,
                   "sdl_f16": F.run(cx, glob, 16.0, "learned", chooser=models[(f, 16.0)])[0],
                   "oracle": S.run_policy("oracle", cx.prompt, cx.y, cx.ann, cx.reg)[0]}
            rows.append(row)
        d = {"n": len(rows), "prompt_len_match": pok, "global_pool_outputs": len(pool),
             "std_sim_equals_measured": int(sum(x["std"] == x["measured_pld_fwd"] for x in rows))}
        for k in ("std", "dom_rule", "mlp", "sd_f1", "sd_f16", "sdl_f1", "sdl_f16", "oracle"):
            d[k] = float(np.mean([x[k] for x in rows]))
        for a, b in (("std", "sd_f1"), ("std", "sd_f16"), ("sd_f1", "sdl_f1"), ("sd_f16", "sdl_f16"), ("dom_rule", "mlp"),
                     ("std", "mlp"), ("sd_f16", "mlp")):
            d[f"{a}_minus_{b}"] = R.paired([x[a] for x in rows], [x[b] for x in rows])
        d["sd_f16_gain_vs_pld_rel"] = 1 - d["sd_f16"] / d["std"]
        d["sd_f1_gain_vs_pld_rel"] = 1 - d["sd_f1"] / d["std"]
        out["conds"][cond] = {"summary": d, "rows": rows}
        print(cond, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if not isinstance(v, dict)}),
              {k: (round(v["mean"], 2), [round(x, 2) for x in v["ci95"]]) for k, v in d.items() if isinstance(v, dict)},
              f"{time.time() - t0:.0f}s", flush=True)
        (OUT / os.environ.get("SD_P1A_OUT", "sd_p1A.json")).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
