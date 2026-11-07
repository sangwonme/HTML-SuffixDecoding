"""Latency / token benchmark of local Gemma 4 (transformers, bitsandbytes 4-bit) on the query->HTML task.

Per sample: streaming TTFT (time until first generated token), total generation time, output tokens/s,
prompt/output token counts, torch peak VRAM, nvidia-smi snapshot (other GPU processes).
Concurrency 1, greedy, warm-up runs excluded. Raw outputs are logged to JSONL (one line per sample).
Creates C:\\Works\\KSEF26\\.gpu_busy during the measured loop (removed in finally).

Usage:
  .venv/Scripts/python.exe -I scripts/bench_gemma.py --model E2B --cond rich_zs --split pilot [--n 50]
  cond: rich_zs | short_zs | rich_fs (3 retrieved examples, query_rich)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"  # CPU thread cap (shared machine)

import psutil  # noqa: E402

MIN_FREE_RAM_GB = 8.0
SHARED_SM_PCT = 10  # another process using >=10% SM counts as GPU sharing
RSS_LIMIT_GB = 14.0  # process memory guard


def ram_state():
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    return {"free_ram_gb": round(vm.available / 2**30, 2), "total_ram_gb": round(vm.total / 2**30, 1),
            "swap_used_gb": round(sw.used / 2**30, 2),
            "rss_gb": round(psutil.Process().memory_info().rss / 2**30, 2)}


def check_rss():
    rss = psutil.Process().memory_info().rss / 2**30
    if rss > RSS_LIMIT_GB:
        raise SystemExit(f"RSS guard: {rss:.1f} GB > {RSS_LIMIT_GB} GB")

import torch  # noqa: E402

torch.set_num_threads(4)
from transformers import AutoProcessor, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from transformers.generation.streamers import BaseStreamer  # noqa: E402

GPU_BUSY = ROOT.parent / ".gpu_busy"
MODELS = {"E2B": "gemma-4-E2B-it-qat-q4_0-unquantized",
          "E4B": "gemma-4-E4B-it-qat-q4_0-unquantized",
          "12B": "gemma-4-12B-it-qat-q4_0-unquantized"}
SYSTEM = ("You generate mobile app UI screens as HTML. Given a user's request, output ONE complete, "
          "self-contained HTML document for a 360x640 px mobile viewport: inline <style> CSS only, "
          "no JavaScript, no external resources (no CDN, fonts, or remote images). Use semantic elements "
          "(header, button, input, img, ul/li, nav, etc.) and include the texts the user mentions. "
          "Reply with only the HTML inside a single ```html code block, no explanation.")


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def gpu_snapshot():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        mem, util = [int(x) for x in q.split(",")]
        pm = subprocess.run(["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
                            capture_output=True, text=True, timeout=15).stdout
        me = os.getpid()
        others = []
        for line in pm.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4 and parts[1].isdigit() and int(parts[1]) != me:
                sm = parts[3]
                if sm.isdigit() and int(sm) >= SHARED_SM_PCT:
                    others.append({"pid": int(parts[1]), "name": parts[-1], "sm": sm})
        return {"mem_used_mib": mem, "util_pct": util, "other_procs_sm": others}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:100]}


class TTFTStreamer(BaseStreamer):
    """First put() is the prompt; the second put() is the first generated token."""

    def __init__(self):
        self.calls = 0
        self.t_first = None

    def put(self, value):
        self.calls += 1
        if self.calls == 2 and self.t_first is None:
            torch.cuda.synchronize()
            self.t_first = time.perf_counter()

    def end(self):
        pass


def build_messages(row, cond, retriever):
    q = row["query_short"] if cond == "short_zs" else row["query_rich"]
    msgs = [{"role": "system", "content": SYSTEM}]
    if cond == "rich_fs":
        for ex in retriever.topk(row["query_rich"], 3, max_html_chars=4000):
            msgs.append({"role": "user", "content": ex["query_rich"]})
            msgs.append({"role": "assistant", "content": "```html\n" + ex["gt_html_flow"] + "\n```"})
    msgs.append({"role": "user", "content": q})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--cond", default="rich_zs", choices=["rich_zs", "short_zs", "rich_fs"])
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--warmup-max-new", type=int, default=64, help="warm-up only exercises kernels")
    ap.add_argument("--quant", default="nf4", choices=["nf4", "bf16"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--text-only", action="store_true",
                    help="load only the language model (drop vision/audio towers), e.g. after OOM")
    args = ap.parse_args()

    rs = ram_state()
    print("RAM at start:", rs, flush=True)
    if rs["free_ram_gb"] < MIN_FREE_RAM_GB:
        raise SystemExit(f"free RAM {rs['free_ram_gb']} GB < {MIN_FREE_RAM_GB} GB; not loading the model")
    rows = load_jsonl(ROOT / "data" / "processed" / f"{args.split}.jsonl")[: args.n]
    retriever = None
    if args.cond == "rich_fs":
        from genui.retrieval import Retriever
        retriever = Retriever(load_jsonl(ROOT / "data" / "processed" / "train_pool.jsonl"))

    mpath = ROOT / "models" / MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(mpath)
    qcfg = None
    if args.quant == "nf4":
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)
    t0 = time.perf_counter()
    model = None
    errs = []
    import transformers
    # text decoder uses SDPA; audio tower (E2B/E4B) has no SDPA support -> eager for that sub-model only
    attn_options = ["sdpa", {"text_config": "sdpa", "vision_config": "sdpa", "audio_config": "eager"},
                    {"text_config": "sdpa", "vision_config": "eager", "audio_config": "eager"}]
    attn_used = None
    if args.text_only:
        cfg = transformers.AutoConfig.from_pretrained(mpath)
        cls = transformers.Gemma4UnifiedForCausalLM if cfg.model_type == "gemma4_unified" else transformers.Gemma4ForCausalLM
        model = cls.from_pretrained(mpath, config=cfg.text_config, quantization_config=qcfg, dtype=torch.bfloat16,
                                    device_map={"": 0}, attn_implementation="sdpa",
                                    key_mapping={r"^model\.language_model\.": "model."})
        attn_used = "sdpa (text-only)"
        attn_options = []
    for attn in attn_options:
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                mpath, quantization_config=qcfg, dtype=torch.bfloat16, device_map={"": 0},
                attn_implementation=attn)
            attn_used = attn
            break
        except Exception as e:  # noqa: BLE001
            errs.append(f"{attn}: {str(e)[:200]}")
            torch.cuda.empty_cache()
    if model is None:
        raise SystemExit("load failed: " + " | ".join(errs))
    model.eval()
    load_s = time.perf_counter() - t0
    weights_mib = torch.cuda.memory_allocated() / 2**20

    run_id = f"{args.model}_{args.quant}{'_textonly' if args.text_only else ''}_{args.cond}_{args.split}{args.tag}"
    out_dir = ROOT / "runs" / "gemma"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.jsonl"
    meta = {
        "run_id": run_id, "model_dir": MODELS[args.model], "model_class": type(model).__name__,
        "quant": args.quant, "compute_dtype": "bfloat16", "attn_implementation": attn_used,
        "decoding": "greedy", "max_new_tokens": args.max_new, "thinking": "off (no <|think|> token)",
        "concurrency": 1, "warmup": args.warmup, "warmup_max_new": args.warmup_max_new, "torch": torch.__version__,
        "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0),
        "ram_at_start": rs, "ram_after_load": ram_state(), "cpu_threads": 4, "rss_limit_gb": RSS_LIMIT_GB,
        "load_seconds": round(load_s, 1), "weights_mib": round(weights_mib), "started_utc":
        datetime.now(timezone.utc).isoformat(), "gpu_at_start": gpu_snapshot(), "system_prompt": SYSTEM,
    }
    try:
        import bitsandbytes
        meta["bitsandbytes"] = bitsandbytes.__version__
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps({k: v for k, v in meta.items() if k != "system_prompt"}), flush=True)

    def run_one(row, max_new=None):
        max_new = max_new or args.max_new
        msgs = build_messages(row, args.cond, retriever)
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True).to(0)
        n_in = int(enc["input_ids"].shape[1])
        st = TTFTStreamer()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**enc, do_sample=False, max_new_tokens=max_new, streamer=st,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        torch.cuda.synchronize()
        t_end = time.perf_counter()
        gen = out[0, n_in:]
        n_out = int(gen.shape[0])
        text = tok.decode(gen, skip_special_tokens=True)
        ttft = (st.t_first - t_start) if st.t_first else None
        total = t_end - t_start
        dec_tps = (n_out - 1) / (total - ttft) if ttft and n_out > 1 and total > ttft else None
        return {
            "id": row["id"], "prompt_tokens": n_in, "output_tokens": n_out, "ttft_s": ttft, "total_s": total,
            "decode_tok_per_s": dec_tps, "e2e_tok_per_s": n_out / total, "hit_max_new": n_out >= max_new,
            "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
            "response": text,
        }

    check_rss()
    # warm-up (not logged as measurements)
    warm_rows = load_jsonl(ROOT / "data" / "processed" / "dev.jsonl")[: args.warmup]  # disjoint from pilot/test
    for row in warm_rows:
        r = run_one(row, args.warmup_max_new)
        print(f"warmup {row['id']} out={r['output_tokens']} total={r['total_s']:.1f}s", flush=True)

    own_flag = not os.environ.get("GENUI_CHAIN")  # the chain script owns the flag across runs
    if own_flag:
        GPU_BUSY.write_text(f"GenerativeUI bench {run_id} pid={os.getpid()} since {datetime.now().isoformat()}\n")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": meta}) + "\n")
            for i, row in enumerate(rows):
                snap = gpu_snapshot()
                check_rss()
                ram = ram_state()
                if ram["free_ram_gb"] < 2.0:
                    raise SystemExit(f"free RAM dropped to {ram['free_ram_gb']} GB; stopping (swap risk)")
                r = run_one(row)
                r.update({"run_id": run_id, "model": args.model, "cond": args.cond,
                          "ts_utc": datetime.now(timezone.utc).isoformat(), "gpu_before": snap, "ram_before": ram,
                          "gpu_shared": bool(snap.get("other_procs_sm"))})
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{i+1}/{len(rows)}] {row['id']} in={r['prompt_tokens']} out={r['output_tokens']} "
                      f"ttft={r['ttft_s']:.3f}s total={r['total_s']:.1f}s dec={r['decode_tok_per_s'] or 0:.1f}t/s "
                      f"peak={r['peak_reserved_mib']}MiB shared={r['gpu_shared']}", flush=True)
    finally:
        if own_flag:
            try:
                GPU_BUSY.unlink()
            except FileNotFoundError:
                pass
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
