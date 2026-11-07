"""Latency oracle gap: same request, different single exemplar (k=1), RNSD-L2 decoding.

For each pilot id (in file order) the 6 candidate train ids from runs/retrieval/pilot_cands6.json are run in a
seeded shuffled order (interleaving candidates within a request avoids time drift confounding). The single
exemplar is both the few-shot example and the draft source. Stops starting new samples at --stop HHMM.
GPU run writes runs/poc/E2B_L2_cands6_raw.jsonl; scoring (CPU) with --score writes runs/poc/E2B_L2_cands6.jsonl.

Usage:
  .venv/Scripts/python.exe -I scripts/poc_cands.py            # GPU run (holds ../.gpu_busy)
  .venv/Scripts/python.exe -I scripts/poc_cands.py --score    # CPU scoring after the GPU run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
POC = ROOT / "runs" / "poc"
CANDS = ROOT / "runs" / "retrieval" / "pilot_cands6.json"
RAW = POC / "E2B_L2_cands6_raw.jsonl"
OUT = POC / "E2B_L2_cands6.jsonl"


def load_rnsd():
    spec = importlib.util.spec_from_file_location("poc_rnsd", ROOT / "scripts" / "poc_rnsd.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gpu_run(stop_hhmm: int, warmup: int):
    R = load_rnsd()
    import psutil
    import torch
    import transformers
    from transformers import AutoTokenizer, BitsAndBytesConfig

    if psutil.virtual_memory().available / 2**30 < 8:
        raise SystemExit("free RAM < 8 GB")
    cands = json.load(open(CANDS, encoding="utf-8"))
    mid = ROOT / "data" / "processed" / "mid"
    pilot = R.load_jsonl(mid / "pilot.jsonl")
    pool_mid = {r["id"]: r for r in R.load_jsonl(mid / "train_pool.jsonl")}
    html = {}
    with open(ROOT / "data" / "processed" / "train_pool.jsonl", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            html[r["id"]] = r["gt_html_flow"]
    mpath = ROOT / "models" / R.MODELS["E2B"]
    tok = AutoTokenizer.from_pretrained(mpath)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        mpath, quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                      bnb_4bit_compute_dtype=torch.bfloat16),
        dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation={"text_config": "sdpa", "vision_config": "sdpa", "audio_config": "eager"})
    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    memo = {}

    def dc(ids):
        key = (len(ids), ids[-1] if ids else None)
        if key not in memo:
            memo[key] = tok.decode(ids, skip_special_tokens=True)
        return memo[key]

    def enc_for(row, cid):
        msgs = [{"role": "system", "content": R.SYSTEM},
                {"role": "user", "content": pool_mid[cid]["query_mid"]},
                {"role": "assistant", "content": "```html\n" + html[cid] + "\n```"},
                {"role": "user", "content": row["query_mid"]}]
        e = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        return {k: v.to(0) for k, v in e.items()}

    rows = [r for r in pilot if r["id"] in cands]
    for r in rows[:warmup]:  # warm-up (not logged), short generations
        memo.clear()
        R.decode(model, tok, enc_for(r, cands[r["id"]][0]), "L2", 32, eos_ids, dc)
    R.GPU_BUSY.write_text(f"GenerativeUI poc_cands pid={os.getpid()} since {datetime.now().isoformat()}\n")
    done_req = 0
    try:
        with open(RAW, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": {"model": "E2B", "cond": "L2", "k": 1, "started_utc":
                                          datetime.now(timezone.utc).isoformat(), "stop_hhmm": stop_hhmm}}) + "\n")
            for r in rows:
                if int(time.strftime("%H%M")) >= stop_hhmm:
                    print(f"stop time reached after {done_req} requests", flush=True)
                    break
                order = list(cands[r["id"]])
                random.Random(f"{r['id']}-20261107").shuffle(order)
                for cid in order:
                    if cid not in html or cid not in pool_mid:
                        continue
                    R.check_rss()
                    memo.clear()
                    o = R.decode(model, tok, enc_for(r, cid), "L2", 2048, eos_ids, dc)
                    n_out = len(o["gen"])
                    rec = {"id": r["id"], "cand_id": cid, "total_s": o["total_s"], "ttft_s": o["ttft_s"],
                           "prompt_tokens": o["n_prompt"], "output_tokens": n_out, "n_fwd": o["n_fwd"],
                           "tokens_per_fwd": n_out / o["n_fwd"], "decode_fwd_s_mean": o["decode_fwd_s_mean"],
                           "hit_max_new": n_out >= 2048, "fires": o["fires"],
                           "response": tok.decode(o["gen"], skip_special_tokens=True)}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                done_req += 1
                print(f"[{done_req}/{len(rows)}] {r['id']} done {time.strftime('%H:%M:%S')}", flush=True)
    finally:
        try:
            R.GPU_BUSY.unlink()
        except FileNotFoundError:
            pass
    print(f"requests completed: {done_req}/{len(rows)}", flush=True)


def score():
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "ms-playwright"))
    from playwright.sync_api import sync_playwright

    from genui.checklist import ChecklistScorer
    from genui.metrics import Scorer
    mid = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/mid/pilot.jsonl", encoding="utf-8")}
    gt = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/pilot.jsonl", encoding="utf-8")}
    rows = [json.loads(l) for l in open(RAW, encoding="utf-8")]
    meta, rows = rows[0], rows[1:]
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=True)
        cs, es = ChecklistScorer(br), Scorer(br)
        gcache = {}
        with open(OUT, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
            for r in rows:
                c = cs.score(r["response"], mid[r["id"]]["checklist"])
                if r["id"] not in gcache:
                    gcache[r["id"]] = es.analyze(gt[r["id"]]["gt_html_flow"])
                e = es.score(r["response"], gt[r["id"]], gcache[r["id"]])
                out = {k: v for k, v in r.items() if k != "response"}
                out.update({"check_score": c["check_score"], "check_all": c["check_all"], "elem_f1": e["elem_f1"],
                            "type_f1": e["type_f1"], "render_ok": e["render_ok"]})
                f.write(json.dumps(out) + "\n")
        br.close()
    print("scored", len(rows), "->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--stop", type=int, default=2255)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    score() if a.score else gpu_run(a.stop, a.warmup)
