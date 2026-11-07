"""E: train the MLP ranker ONLY on train-split PLD outputs (runs/poc/E/train150/E2B_pld.jsonl, 150 train apps,
disjoint from pilot/dev/test) and save numpy weights runs/study/ranker_mlp_trainsplit.npz.
Then evaluate the same model by replay on the existing pilot data (n=444 targets of study_ranker.load_all).

  .venv/Scripts/python.exe -I scripts/study_ranker_trainsplit.py
Output: runs/study/ranker_mlp_trainsplit.npz, runs/study/ranker_trainsplit.json
"""
from __future__ import annotations

import os, sys, pathlib, json, time
os.environ.setdefault("SIM_THREADS", "2")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

import study_draft_sim as S
import study_ranker as R
from study_common import OUT, ROOT, load_jsonl

SRC = ROOT / "runs" / "poc" / "E" / "train150" / "E2B_pld.jsonl"


def main():
    S.guard(int(os.environ["SIM_THREADS"]))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    cache = {}

    def dec(i):
        if i not in cache:
            cache[i] = tok.decode([i])
        return cache[i]
    mid = ROOT / "data" / "processed" / "mid"
    q = {}
    for fn in ("train_pool.jsonl", "pilot.jsonl", "dev.jsonl", "test.jsonl"):
        if (mid / fn).exists():
            for r in load_jsonl(mid / fn):
                q.setdefault(r["id"], r["query_mid"])
    html = {r["id"]: r["gt_html_flow"] for r in load_jsonl(ROOT / "data" / "processed" / "train_pool.jsonl")}
    alt = {}
    rows_src = [r for r in load_jsonl(SRC) if "id" in r]
    meta = [r for r in load_jsonl(SRC) if "_meta" in r]
    if meta and meta[0]["_meta"].get("exemplar_html"):
        alt = json.loads((ROOT / meta[0]["_meta"]["exemplar_html"]).read_text(encoding="utf-8"))
    seen, tg = set(), []
    pok = 0
    for r in rows_src:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        msgs = [{"role": "system", "content": S.SYSTEM}]
        for e in r["example_ids"]:
            msgs.append({"role": "user", "content": q[e]})
            msgs.append({"role": "assistant", "content": "```html\n" + alt.get(e, html[e]) + "\n```"})
        msgs.append({"role": "user", "content": q[r["id"]]})
        p = list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])
        pok += len(p) == r.get("prompt_tokens", -1)
        tg.append({"id": r["id"], "prompt": p, "y": list(r["gen_ids"]), "max_new": 2048, "measured": r.get("n_fwd")})
    print("train targets", len(tg), "prompt_len_match", pok, flush=True)
    t0 = time.time()
    cxs = [R.Ctx(t, dec) for t in tg]
    rows, rule_tr, std_match = [], [], 0
    for cx, t in zip(cxs, tg):
        nf, rw, _ = R.replay(cx, "rule", collect=True)
        rows += rw; rule_tr.append(nf)
        std_match += S.run_policy("std", cx.prompt, cx.y, cx.ann, cx.reg)[0] == t["measured"]
    print("states", len(rows), "std sim == measured", std_match, f"{time.time() - t0:.0f}s", flush=True)
    w = R.train_mlp_np(rows, R.STRUCT + R.LEXI + R.HIST)
    np.savez(OUT / "ranker_mlp_trainsplit.npz", **w)
    print("saved ranker_mlp_trainsplit.npz", flush=True)
    # evaluation on existing pilot data (no pilot data used in training)
    raw, targets = R.load_all(tok)
    pcx = [R.Ctx(t, dec) for t in targets]
    ab = {(r["set"], r["id"], ): r for r in json.loads((OUT / "ranker_mlp_ablation.json").read_text(encoding="utf-8"))["per_target"]}
    res = []
    for t, cx in zip(targets, pcx):
        nf = R.replay(cx, lambda Fm: int(np.argmax(R.mlp_scores(Fm, w) - 1e-6 * np.arange(len(Fm)))))[0]
        rule = R.replay(cx, "rule")[0]
        res.append({"set": t["set"], "id": t["id"], "trainsplit_mlp": nf, "dom_rule": rule,
                    "crossfit_mlp": ab.get((t["set"], t["id"]), {}).get("mlp_full")})
    sets = [r["set"] for r in res]
    groups = {"all": list(range(len(res))), "poc_greedy": [i for i, s in enumerate(sets) if s in ("C1", "C0", "C4g", "C7g")],
              "gate2_measured": [i for i, s in enumerate(sets) if s in ("g2pld", "g2dom")],
              "gemma_approx": [i for i, s in enumerate(sets) if s.startswith("gemma")], "cands6": [i for i, s in enumerate(sets) if s == "cands6"]}
    out = {"n_train_requests": len(tg), "n_train_states": len(rows), "prompt_len_match": pok, "std_sim_equals_measured": std_match, "groups": {}}
    for g, ids in groups.items():
        d = {"n": len(ids)}
        for k in ("dom_rule", "trainsplit_mlp"):
            d[k] = float(np.mean([res[i][k] for i in ids]))
        d["rule_minus_trainsplit"] = R.paired([res[i]["dom_rule"] for i in ids], [res[i]["trainsplit_mlp"] for i in ids])
        cf = [i for i in ids if res[i]["crossfit_mlp"] is not None]
        if cf:
            d["crossfit_mlp"] = float(np.mean([res[i]["crossfit_mlp"] for i in cf]))
            d["crossfit_minus_trainsplit"] = R.paired([res[i]["crossfit_mlp"] for i in cf], [res[i]["trainsplit_mlp"] for i in cf])
        out["groups"][g] = d
        print(g, json.dumps(d)[:600], flush=True)
    out["rows"] = res
    (OUT / "ranker_trainsplit.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
