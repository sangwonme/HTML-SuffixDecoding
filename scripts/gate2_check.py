"""Gate 2 consistency check (CPU only): does DomDrafter (incremental, in poc_rnsd) reproduce the simulator's
'dom' policy step-for-step?  Replays a greedy target y with both and compares forward counts.
Also used after the GPU run: sim prediction for each measured output (same y) vs measured n_fwd.

Usage:
  .venv/Scripts/python.exe -I scripts/gate2_check.py replay            # replay C1 greedy outputs (pilot 10)
  .venv/Scripts/python.exe -I scripts/gate2_check.py measured DIR      # compare runs in DIR (E2B_pld/dom.jsonl)
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    from transformers import AutoTokenizer
    R = load("poc_rnsd", ROOT / "scripts" / "poc_rnsd.py")
    S = R._sim()
    tok = AutoTokenizer.from_pretrained(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized")
    mode = sys.argv[1]
    if mode == "replay":
        files = [("dom", ROOT / "runs" / "poc" / "E2B_greedy.jsonl")]
    else:
        d = Path(sys.argv[2])
        files = [("pld", d / "E2B_pld.jsonl"), ("dom", d / "E2B_dom.jsonl"), ("ranker", d / "E2B_ranker.jsonl")]
    RM = R._ranker_mod() if (mode != "replay" and (Path(sys.argv[2]) / "E2B_ranker.jsonl").exists()) else None
    if RM:
        fmap = json.load(open(ROOT / "runs" / "study" / "ranker_fold_map.json", encoding="utf-8"))
        W = {f: RM.load_weights(ROOT / "runs" / "study" / f"ranker_mlp_fold{f}.npz") for f in set(fmap.values())}
    tcache = {}

    def dec1(t):
        if t not in tcache:
            tcache[t] = tok.decode([t])
        return tcache[t]
    prompts = {p["row"]["id"]: p for p in R.build_prompts(50)}
    out = []
    for pol, f in files:
        if not f.exists():
            continue
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if "id" not in r:
                continue
            ids = list(tok.apply_chat_template(prompts[r["id"]]["msgs"], add_generation_prompt=True, tokenize=True,
                                               return_dict=True)["input_ids"])
            prompt = np.array(ids, np.int64)
            y = np.array(r["gen_ids"], np.int64)
            strs = [tok.decode([i]) for i in list(prompt) + list(y)]
            ann, reg = S.annotate(strs)
            ann.append(None)
            if pol == "ranker":
                nf_sim = None  # simulator for the ranker = RankerDrafter replay (study_ranker uses the same class)
            else:
                nf_sim, _ = S.run_policy("std" if pol == "pld" else "dom", prompt, y, ann, reg, max_new=2048)
            # replay with the incremental drafter (no model): accept = LCP with y
            P = len(prompt)
            dd = R.DomDrafter(tok) if pol == "dom" else None
            rk = RM.RankerDrafter(prompt, dec1, W[fmap[r["id"]]], max_new=2048) if pol == "ranker" else None
            pos, nf = 1, 1
            ctx = np.concatenate([prompt, y]).astype(np.int64)
            while pos < len(y):
                L = P + pos
                g = min(16, 2048 - pos - 1)
                if rk is not None:
                    d = rk.draft(list(y[:pos]), 16)
                else:
                    d = dd.draft(ctx[:L], g) if dd else R.ngram_draft(ctx[:L], g)
                a = 0
                while a < len(d) and pos + a < len(y) and d[a] == y[pos + a]:
                    a += 1
                nf += 1
                pos += a + 1
            rec = {"pol": pol, "id": r["id"], "rep": r.get("rep"), "T": len(y), "fwd_sim": nf_sim, "fwd_replay": nf,
                   "fwd_measured": r["n_fwd"] if mode == "measured" else None}
            out.append(rec)
            print(rec, flush=True)
    for o in out:
        if o["fwd_sim"] is None:
            o["fwd_sim"] = o["fwd_replay"]
    agree = sum(o["fwd_sim"] == o["fwd_replay"] for o in out)
    print(f"sim == incremental replay: {agree}/{len(out)}")
    if mode == "measured":
        m = sum(o["fwd_sim"] == o["fwd_measured"] for o in out)
        print(f"sim == measured: {m}/{len(out)}")
        (Path(sys.argv[2]) / "sim_vs_measured.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n")


if __name__ == "__main__":
    main()
