"""Perceptual latency (study's genui.interleave.perceptual_latency) for selected PoC conditions.

Token arrival times are reconstructed from the logged decode trace (no per-token clock was recorded):
  token 0 at ttft; verify step s (s = 1..S) delivers its acc+1 tokens at ttft + s * (total - ttft) / S.
For greedy this is ttft + k*(mean step time), i.e. the study's linear approximation.
head_share = fraction of output tokens emitted before the first '<body' (CSS-first heads -> large).
Usage: .venv/Scripts/python.exe -I scripts/poc_perc.py E2B
Writes runs/poc/<model>_perc.jsonl and runs/poc/<model>_perc.md. CPU + headless Chromium only.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "ms-playwright"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from playwright.sync_api import sync_playwright  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from genui.interleave import perceptual_latency  # noqa: E402

POC = ROOT / "runs" / "poc"
model = sys.argv[1] if len(sys.argv) > 1 else "E2B"
ROWS = [("C0", "greedy_C0"), ("C1", "greedy"), ("C2", "pld"), ("C3", "L2_P50C1"), ("C4", "L2_C4"),
        ("C7g", "greedy_C7"), ("C7", "L2_C7")]
tok = AutoTokenizer.from_pretrained(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized")


def load(cond):
    f = POC / f"{model}_{cond}.jsonl"
    if not f.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in open(f, encoding="utf-8")) if "id" in r}


def token_times(r):
    gen, steps = r["gen_ids"], r["step_acc"]
    T = len(gen)
    S = max(1, len(steps))
    dt = (r["total_s"] - r["ttft_s"]) / S
    times = [r["ttft_s"]]
    for s, (_k, acc) in enumerate(steps, 1):
        times += [r["ttft_s"] + s * dt] * (acc + 1)
    return times[:T] + [r["total_s"]] * max(0, T - len(times))


c1_ids = set(load("greedy"))
recs = []
with sync_playwright() as pw:
    br = pw.chromium.launch(headless=True)
    page = br.new_page(viewport={"width": 360, "height": 640})
    page.route("**/*", lambda route: route.abort())
    for name, cond in ROWS:
        for i, r in load(cond).items():
            if i not in c1_ids:
                continue
            gen = r["gen_ids"]
            ends = [len(tok.decode(gen[: k + 1], skip_special_tokens=True)) for k in range(len(gen))]
            tt = list(zip(ends, token_times(r)))
            res = perceptual_latency(r["response"], tt, page=page)
            b = r["response"].lower().find("<body")
            head_k = next((k for k, e in enumerate(ends) if e > b), len(gen)) if b >= 0 else len(gen)
            recs.append({"row": name, "cond": cond, "id": i, "T_s": res["T_s"], "Lperc_s": res["Lperc_s"],
                         "Lperc_over_T": res["Lperc_over_T"], "t50_s": res["t50_s"], "t50_over_T": res["t50_over_T"],
                         "t90_over_T": res["t90_over_T"], "head_share": head_k / max(1, len(gen))})
    br.close()
with open(POC / f"{model}_perc.jsonl", "w", encoding="utf-8") as f:
    for x in recs:
        f.write(json.dumps(x) + "\n")
c1_t50 = {x["id"]: x["t50_s"] for x in recs if x["row"] == "C1"}
lines = ["| row | cond | n | T p50 (s) | t50 p50 (s) | t50 ratio vs C1 (p50) | Lperc/T | t50/T | head_share |",
         "|---|---|---|---|---|---|---|---|---|"]
for name, cond in ROWS:
    xs = [x for x in recs if x["row"] == name]
    if not xs:
        lines.append(f"| {name} | {cond} | 0 | (not run) |||||")
        continue
    med = lambda k: statistics.median(x[k] for x in xs)  # noqa: E731
    mean = lambda k: statistics.mean(x[k] for x in xs)  # noqa: E731
    ratio = statistics.median(x["t50_s"] / c1_t50[x["id"]] for x in xs if c1_t50.get(x["id"]))
    lines.append(f"| {name} | {cond} | {len(xs)} | {med('T_s'):.1f} | {med('t50_s'):.2f} | {ratio:.3f} | "
                 f"{mean('Lperc_over_T'):.3f} | {mean('t50_over_T'):.3f} | {mean('head_share'):.3f} |")
out = ("Token arrival times reconstructed from the decode trace (per-step uniform); same 10 pilot ids as C1.\n\n"
       + "\n".join(lines) + "\n")
(POC / f"{model}_perc.md").write_text(out, encoding="utf-8")
print(out)
