"""Validate the processed dataset.

1. Render every GT HTML (flow + abs) with headless Chromium (Playwright), network blocked.
   Success = no page error, body has non-zero size, #[data-label] == sum(element_counts).
2. Save 20 side-by-side review images (original | GT abs | GT flow) to data/review/.
3. Length / token statistics (Gemma 4 tokenizer) for queries and GT HTML.

Usage: .venv/Scripts/python.exe -I scripts/validate_dataset.py [--splits pilot,dev,test,train_pool]
Writes data/processed/validation_report.json. Never opens a window (headless only).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "ms-playwright"))

from PIL import Image, ImageDraw  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

PROC = ROOT / "data" / "processed"
REVIEW = ROOT / "data" / "review"
N_REVIEW = 20
TOKENIZER = ROOT / ".cache" / "gemma4_tokenizer" / "tokenizer.json"

CHECK_JS = """() => {
  const b = document.body;
  return {n: document.querySelectorAll('[data-label]').length,
          w: b ? b.scrollWidth : 0, h: b ? b.scrollHeight : 0};
}"""


def load(name):
    with open(PROC / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summarize(xs):
    return {"mean": round(statistics.mean(xs), 1), "p50": pct(xs, 50), "p95": pct(xs, 95),
            "min": min(xs), "max": max(xs)}


def render_check(page, html, expected):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.set_content(html, wait_until="load", timeout=10000)
    r = page.evaluate(CHECK_JS)
    page.remove_listener("pageerror", page.listeners("pageerror")[0]) if hasattr(page, "listeners") else None
    ok = not errors and r["w"] > 0 and r["h"] > 0 and r["n"] == expected
    return ok, r, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="pilot,dev,test,train_pool")
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(TOKENIZER))
    report = {"render": {}, "lengths": {}}
    REVIEW.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 360, "height": 640}, device_scale_factor=1,
                                  java_script_enabled=True)
        ctx.route("**/*", lambda route: route.abort())  # no network at all
        page = ctx.new_page()
        errors_seen = []
        page.on("pageerror", lambda e: errors_seen.append(str(e)))

        for split in args.splits.split(","):
            rows = load(split)
            t0 = time.time()
            res = {"n": len(rows), "flow_ok": 0, "abs_ok": 0, "failures": []}
            for r in rows:
                exp = sum(r["element_counts"].values())
                for kind in ("flow", "abs"):
                    errors_seen.clear()
                    try:
                        page.set_content(r[f"gt_html_{kind}"], wait_until="load", timeout=10000)
                        m = page.evaluate(CHECK_JS)
                        ok = not errors_seen and m["w"] > 0 and m["h"] > 0 and m["n"] == exp
                    except Exception as e:  # noqa: BLE001
                        ok, m = False, {"exc": str(e)[:200]}
                    if ok:
                        res[f"{kind}_ok"] += 1
                    elif len(res["failures"]) < 20:
                        res["failures"].append({"id": r["id"], "kind": kind, "expected": exp, "got": m,
                                                "errors": errors_seen[:2]})
            res["flow_rate"] = round(res["flow_ok"] / len(rows), 4)
            res["abs_rate"] = round(res["abs_ok"] / len(rows), 4)
            res["seconds"] = round(time.time() - t0, 1)
            report["render"][split] = res
            print(split, {k: v for k, v in res.items() if k != "failures"}, file=sys.stderr)

            if split == "test":  # review images: first N_REVIEW test samples (file order is seeded)
                for i, r in enumerate(rows[:N_REVIEW]):
                    shots = []
                    for kind in ("abs", "flow"):
                        page.set_content(r[f"gt_html_{kind}"], wait_until="load")
                        shots.append(Image.open(io.BytesIO(page.screenshot(full_page=False))).convert("RGB"))
                    orig = Image.open(ROOT / r["screenshot_path"]).convert("RGB").resize((360, 640))
                    canvas = Image.new("RGB", (360 * 3 + 40, 640 + 120), "white")
                    for j, (im, cap) in enumerate(zip([orig] + shots, ["original screenshot", "GT abs", "GT flow"])):
                        canvas.paste(im.resize((360, 640)), (10 + j * 370, 30))
                        ImageDraw.Draw(canvas).text((10 + j * 370, 8), cap, fill="black")
                    ImageDraw.Draw(canvas).multiline_text(
                        (10, 680), f"{r['id']} | {r['category']}\n{r['query_short']}", fill="black")
                    canvas.save(REVIEW / f"cmp_{i:02d}_{r['id']}.png")
        browser.close()

    for split in args.splits.split(","):
        rows = load(split)
        L = {}
        for field in ("query_short", "query_rich", "gt_html_flow", "gt_html_abs"):
            L[field] = {
                "chars": summarize([len(r[field]) for r in rows]),
                "gemma4_tokens": summarize([len(tok.encode(r[field]).ids) for r in rows]),
            }
        L["elements"] = summarize([sum(r["element_counts"].values()) for r in rows])
        L["key_texts"] = summarize([len(r["key_texts"]) for r in rows])
        report["lengths"][split] = L

    out = PROC / "validation_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
