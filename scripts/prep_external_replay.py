"""Pipeline prep for external HTML (WebSight subset): exemplar retrieval -> prompts -> GT-as-target replay proxy.
CPU only (2 threads, BelowNormal). Imports study_draft_sim (not modified). No GPU, no rendering.

1. Retrieval: TF-IDF over the pool queries (genui.retrieval.Retriever, same settings as poc_rnsd: top-3,
   max_html_chars=4000), for each eval row (split == "eval").
2. Prompts: the same chat layout as poc_rnsd (SYSTEM, 3 x (query, ```html exemplar```), query), written to
   runs/retrieval/external_websight_prompts.jsonl for a later GPU run (`msgs` + `example_ids`).
3. Proxy replay (no model output yet): the GT HTML stands in for the model output, so this measures how much
   of a target-like document can be copied from the prompt by each draft policy
   (std PLD, DOM rule, any-position oracle; gamma=16, LCP + 1). The same proxy is run on S2W pilot 50
   (GT flow, exemplars = poc_rnsd TF-IDF top-3) for a like-for-like comparison.
   CAVEAT: a proxy only. Real outputs differ from GT; forward counts of real runs must come from the GPU run.
Usage: .venv/Scripts/python.exe -I scripts/prep_external_replay.py -> runs/retrieval/external_replay.json
"""
from __future__ import annotations

import os

os.environ["SIM_THREADS"] = "2"
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "2"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import study_draft_sim as S  # noqa: E402
from genui.retrieval import Retriever  # noqa: E402
from retr_seq_oracle import boot  # noqa: E402

EXT = ROOT / "data" / "processed" / "external"
OUTD = ROOT / "runs" / "retrieval"


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def msgs_for(exs, q, qkey):
    m = [{"role": "system", "content": S.SYSTEM}]
    for e in exs:
        m.append({"role": "user", "content": e[qkey]})
        m.append({"role": "assistant", "content": "```html\n" + e["html"] + "\n```"})
    m.append({"role": "user", "content": q})
    return m


def replay(tok, msgs, target_html):
    ids = list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])
    y = tok("```html\n" + target_html + "\n```", add_special_tokens=False)["input_ids"]
    prompt, y = np.array(ids, np.int64), np.array(y, np.int64)
    strs = [tok.decode([i]) for i in list(prompt) + list(y)]
    ann, reg = S.annotate(strs)
    reg = [(-2 if i >= len(prompt) else r) for i, r in enumerate(reg)]
    ann.append(None)
    out = {"T": int(len(y)), "P": int(len(prompt))}
    for pol in ("std", "dom", "oracle"):
        out[pol] = S.run_policy(pol, prompt, y, ann, reg, max_new=4096)[0]
    return out


def main():
    S.guard(2)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    res = {}
    # ---- WebSight
    ws = jl(EXT / "websight.jsonl")
    pool = [r for r in ws if r["split"] == "pool"]
    ev = [r for r in ws if r["split"] == "eval"]
    ret = Retriever([dict(r, gt_html_flow=r["html"]) for r in pool], field="query")
    rows, top3 = [], {}
    with open(OUTD / "external_websight_prompts.jsonl", "w", encoding="utf-8") as f:
        for r in ev:
            exs = ret.topk(r["query"], 3, max_html_chars=4000)
            top3[r["id"]] = [e["id"] for e in exs]
            m = msgs_for(exs, r["query"], "query")
            f.write(json.dumps({"id": r["id"], "msgs": m, "example_ids": top3[r["id"]]}, ensure_ascii=False) + "\n")
            x = replay(tok, m, r["html"])
            rows.append({"set": "websight", "id": r["id"], **x})
            print("ws", r["id"], x, flush=True)
    (OUTD / "external_websight_top3.json").write_text(json.dumps(top3, indent=1), encoding="utf-8")
    # ---- S2W pilot 50, same proxy (exemplars = poc_rnsd TF-IDF top-3 over query_mid)
    mid = ROOT / "data" / "processed" / "mid"
    pilot = jl(mid / "pilot.jsonl")
    gt = {r["id"]: r["gt_html_flow"] for r in jl(ROOT / "data" / "processed" / "pilot.jsonl")}
    html_by_id = {}
    with open(ROOT / "data" / "processed" / "train_pool.jsonl", encoding="utf-8") as f:
        for l in f:
            x = json.loads(l)
            html_by_id[x["id"]] = x["gt_html_flow"]
    spool = [dict(r, gt_html_flow=html_by_id[r["id"]], html=html_by_id[r["id"]]) for r in jl(mid / "train_pool.jsonl")]
    sret = Retriever(spool, field="query_mid")
    for r in pilot:
        exs = sret.topk(r["query_mid"], 3, max_html_chars=4000)
        x = replay(tok, msgs_for(exs, r["query_mid"], "query_mid"), gt[r["id"]])
        rows.append({"set": "s2w_pilot", "id": r["id"], **x})
        print("s2w", r["id"], x, flush=True)
    for s in ("websight", "s2w_pilot"):
        rs = [r for r in rows if r["set"] == s]
        d = {"n": len(rs), "T_mean": float(np.mean([r["T"] for r in rs])), "P_mean": float(np.mean([r["P"] for r in rs]))}
        for pol in ("std", "dom", "oracle"):
            d[f"fwd_{pol}"] = float(np.mean([r[pol] for r in rs]))
            d[f"tok_per_fwd_{pol}"] = float(sum(r["T"] for r in rs) / sum(r[pol] for r in rs))
        x = [r["std"] - r["dom"] for r in rs]
        d["std-dom"] = {"mean": float(np.mean(x)), "ci95": boot(x), "rel": float(sum(x) / sum(r["std"] for r in rs))}
        x = [r["std"] - r["oracle"] for r in rs]
        d["std-oracle_rel"] = float(sum(x) / sum(r["std"] for r in rs))
        res[s] = d
    (OUTD / "external_replay.json").write_text(json.dumps({"summary": res, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
