"""P1 prep: render-preserving perturbations (fmt/css/dom/cls) of the pilot-50 TF-IDF exemplars used by poc_rnsd
(query_mid top-3, max_html_chars=4000), pixel-verified, + simulator pre-check of condition B (reference-only
perturbation: prompt & model output unchanged, draft pool built from perturbed exemplar HTML).

  .venv/Scripts/python.exe -I scripts/study_perturb_exemplars.py
Outputs: runs/retrieval/pilot_exemplar_html_{fmt,css,dom,cls}.json, runs/study/p1_perturb_report.json
"""
from __future__ import annotations

import os, sys, pathlib, json, io, time
os.environ.setdefault("SIM_THREADS", "2")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from PIL import Image

from study_common import ROOT, OUT, Renderer, load_jsonl
from genui.retrieval import Retriever
from genui.perturb import PERTURBATIONS


def full_shot(page, html):
    page.set_content(html, wait_until="load", timeout=15000)
    return np.asarray(Image.open(io.BytesIO(page.screenshot(full_page=True, animations="disabled", caret="hide"))).convert("RGB"))


def main():
    import study_draft_sim as S
    S.guard(int(os.environ["SIM_THREADS"]))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    mid = ROOT / "data" / "processed" / "mid"
    pilot = load_jsonl(mid / "pilot.jsonl")[:50]
    html_by_id = {r["id"]: r["gt_html_flow"] for r in load_jsonl(ROOT / "data" / "processed" / "train_pool.jsonl")}
    pool = [dict(r, gt_html_flow=html_by_id[r["id"]]) for r in load_jsonl(mid / "train_pool.jsonl") if r["id"] in html_by_id]
    ret = Retriever(pool, field="query_mid")
    ids = []
    for r in pilot:
        for e in ret.topk(r["query_mid"], 3, max_html_chars=4000):
            if e["id"] not in ids:
                ids.append(e["id"])
    rend = Renderer()
    rep = {"n_exemplars": len(ids)}
    out_html = {}
    try:
        for name, f in PERTURBATIONS.items():
            ok, changed, dtok, rows = 0, 0, [], []
            res = {}
            for k, i in enumerate(ids):
                h = html_by_id[i]
                hp = f(h, seed=k, page=rend.page)
                if hp == h:
                    rows.append({"id": i, "changed": False}); continue
                changed += 1
                a, b = full_shot(rend.page, h), full_shot(rend.page, hp)
                same = bool(a.shape == b.shape and (a == b).all())
                t0, t1 = len(tok(h, add_special_tokens=False)["input_ids"]), len(tok(hp, add_special_tokens=False)["input_ids"])
                rows.append({"id": i, "changed": True, "pixel_identical": same, "tok": t0, "tok_p": t1})
                if same:
                    ok += 1; res[i] = hp; dtok.append(t1 / t0 - 1)
            (ROOT / "runs" / "retrieval" / f"pilot_exemplar_html_{name}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
            out_html[name] = res
            rep[name] = {"changed": changed, "pixel_identical": ok, "kept_rate": ok / len(ids),
                         "tok_change_mean": float(np.mean(dtok)) if dtok else None,
                         "tok_change_range": [float(np.min(dtok)), float(np.max(dtok))] if dtok else None,
                         "example": next(iter(res.values()))[:600] if res else None, "rows": rows}
            print(name, {k: v for k, v in rep[name].items() if k not in ("rows", "example")}, flush=True)
    finally:
        rend.close()
    # ---------------- simulator pre-check, condition B (reference-only): target = gate2 PLD outputs (original)
    import study_ranker as R
    cache = {}

    def dec(i):
        if i not in cache:
            cache[i] = tok.decode([i])
        return cache[i]
    mid_pool = {r["id"]: r for r in load_jsonl(mid / "train_pool.jsonl")}
    mid_pilot = {r["id"]: r for r in load_jsonl(mid / "pilot.jsonl")}
    fmap = json.loads((OUT / "ranker_fold_map.json").read_text(encoding="utf-8"))
    W = {f: R.load_weights(OUT / f"ranker_mlp_fold{f}.npz") for f in range(5)}
    g2 = {}
    for r in load_jsonl(ROOT / "runs" / "poc" / "gate2_p50" / "E2B_pld.jsonl"):
        if "id" in r and r["id"] not in g2:
            g2[r["id"]] = r

    def prompt_for(qid, ex_ids, override):
        msgs = [{"role": "system", "content": S.SYSTEM}]
        for e in ex_ids:
            msgs.append({"role": "user", "content": mid_pool[e]["query_mid"]})
            msgs.append({"role": "assistant", "content": "```html\n" + override.get(e, html_by_id[e]) + "\n```"})
        msgs.append({"role": "user", "content": mid_pilot[qid]["query_mid"]})
        return list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])

    sim = {}
    for cond in ["orig"] + list(PERTURBATIONS):
        ov = {} if cond == "orig" else out_html[cond]
        per = {}
        for qid, r in g2.items():
            t = {"prompt": prompt_for(qid, r["example_ids"], ov), "y": list(r["gen_ids"]), "max_new": 2048}
            cx = R.Ctx(t, dec)
            w = W[fmap[qid]]
            per[qid] = {"std": S.run_policy("std", cx.prompt, cx.y, cx.ann, cx.reg)[0],
                        "rule": R.replay(cx, "rule")[0],
                        "mlp": R.replay(cx, lambda F, w=w: int(np.argmax(R.mlp_scores(F, w) - 1e-6 * np.arange(len(F)))))[0],
                        "oracle": S.run_policy("oracle", cx.prompt, cx.y, cx.ann, cx.reg)[0]}
        sim[cond] = per
        print("simB", cond, {k: float(np.mean([v[k] for v in per.values()])) for k in ("std", "rule", "mlp", "oracle")}, flush=True)
    summ = {}
    for cond, per in sim.items():
        q = list(per)
        d = {k: float(np.mean([per[i][k] for i in q])) for k in ("std", "rule", "mlp", "oracle")}
        d["rule_minus_mlp"] = R.paired([per[i]["rule"] for i in q], [per[i]["mlp"] for i in q])
        d["std_minus_rule"] = R.paired([per[i]["std"] for i in q], [per[i]["rule"] for i in q])
        d["std_minus_mlp"] = R.paired([per[i]["std"] for i in q], [per[i]["mlp"] for i in q])
        if cond != "orig":
            for k in ("std", "rule", "mlp"):
                d[f"delta_vs_orig_{k}"] = R.paired([per[i][k] for i in q], [sim["orig"][i][k] for i in q])
        summ[cond] = d
    rep["simB"] = {"summary": summ, "per_request": sim,
                   "note": "condition B: prompt examples perturbed in the draft pool only; target = gate2 PLD output (original prompt)."
                           " Approximation: the simulator's ctx uses the perturbed prompt as the draft pool."}
    (OUT / "p1_perturb_report.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    for c, d in summ.items():
        print(c, json.dumps(d)[:700])


if __name__ == "__main__":
    main()
