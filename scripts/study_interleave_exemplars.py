"""Build interleaved (i') versions of the few-shot exemplar HTML used by scripts/poc_rnsd.py (fewshot=tfidf:
query_mid TF-IDF top-3 over mid train_pool, max_html_chars=4000) for the first N pilot rows.

  .venv/Scripts/python.exe -I scripts/study_interleave_exemplars.py [N=10]
Writes runs/retrieval/pilot_exemplar_html_interleaved.json  {train_id: html}  (only exemplars whose interleaved
version renders pixel-identical, full page) and runs/study/interleave_exemplars_report.json.
"""
from __future__ import annotations

import sys, pathlib, json, io
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from PIL import Image

from study_common import ROOT, OUT, Renderer, encode, guard_setup, load_jsonl
from genui.retrieval import Retriever
from genui.interleave import interleave


def full_shot(page, html):
    page.set_content(html, wait_until="load", timeout=15000)
    return np.asarray(Image.open(io.BytesIO(page.screenshot(full_page=True, animations="disabled", caret="hide"))).convert("RGB"))


def main():
    guard_setup()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    mid = ROOT / "data" / "processed" / "mid"
    pilot = load_jsonl(mid / "pilot.jsonl")[:n]
    html_by_id = {r["id"]: r["gt_html_flow"] for r in load_jsonl(ROOT / "data" / "processed" / "train_pool.jsonl")}
    pool = [dict(r, gt_html_flow=html_by_id[r["id"]]) for r in load_jsonl(mid / "train_pool.jsonl") if r["id"] in html_by_id]
    ret = Retriever(pool, field="query_mid")
    ids, per_pilot = [], {}
    for r in pilot:
        ex = [e["id"] for e in ret.topk(r["query_mid"], 3, max_html_chars=4000)]
        per_pilot[r["id"]] = ex
        ids += [i for i in ex if i not in ids]
    rend = Renderer()
    out, rep = {}, []
    try:
        for i in ids:
            h = html_by_id[i]
            hi = interleave(h, rend.page)
            row = {"id": i, "changed": hi != h, "tok": len(encode(h)[0]), "tok_i": len(encode(hi)[0]),
                   "chars": len(h), "chars_i": len(hi)}
            if hi != h:
                a, b = full_shot(rend.page, h), full_shot(rend.page, hi)
                row["pixel_identical"] = bool(a.shape == b.shape and (a == b).all())
                if row["pixel_identical"]:
                    out[i] = hi
            row["crosses_4000"] = len(h) <= 4000 < len(hi)
            rep.append(row)
    finally:
        rend.close()
    dst = ROOT / "runs" / "retrieval" / "pilot_exemplar_html_interleaved.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    ch = [r for r in rep if r["changed"]]
    summ = {"n_pilot": n, "n_exemplars": len(ids), "n_changed": len(ch),
            "n_pixel_identical": sum(r.get("pixel_identical", False) for r in ch), "n_written": len(out),
            "tok_diff_mean": float(np.mean([r["tok_i"] - r["tok"] for r in ch])) if ch else 0.0,
            "tok_ratio_mean": float(np.mean([r["tok_i"] / r["tok"] for r in ch])) if ch else 1.0,
            "n_crosses_4000_chars": sum(r["crosses_4000"] for r in rep),
            "per_pilot_exemplars": per_pilot, "rows": rep}
    (OUT / "interleave_exemplars_report.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print({k: v for k, v in summ.items() if k not in ("rows", "per_pilot_exemplars")})
    print("example:", next(iter(out.values()))[:700] if out else None)


if __name__ == "__main__":
    main()
