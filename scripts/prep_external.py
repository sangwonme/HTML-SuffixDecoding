"""External HTML data (small subsets) for generalization checks: WebSight v0.2 and Design2Code. CPU only.

Downloaded data is UNTRUSTED: it is only parsed (lxml, no script execution, no rendering here). Raw copies live in
data/raw/<name>/. Run with `python -I`.
  WebSight (HuggingFaceM4/WebSight, cc-by-4.0, config v0.2, 1.92M rows): 200 rows (text fields only, no images)
      via the HF datasets-server rows API, 100 from offset 0 and 100 from offset 1,000,000.
      query = llm_generated_idea (the prompt the HTML was generated from); target = text (HTML, Tailwind 2.2.19 CDN).
  Design2Code (SALT-NLP/Design2Code, odc-by, 484 pages): the .html files only (35.6 MB), via snapshot_download.
      No natural-language description exists, so the query is a deterministic template from the page itself
      (title, headings, short button/link labels), in the spirit of query_mid.
Preprocessing -> data/processed/external/{websight,design2code}.jsonl with
  id, source, query, html, html_chars, n_tokens (Gemma tokenizer), fits_gen (n_tokens <= 1800, max_new 2048),
  fits_exemplar (html_chars <= 4000, same cap as poc_rnsd exemplars), split (pool | eval; seeded),
  key_texts (for a text-recall / checklist-style score), n_elements.
Usage: .venv/Scripts/python.exe -I scripts/prep_external.py [--skip-download]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "external"
TOKENIZER = ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized" / "tokenizer.json"
SEED = 20261108
GEN_TOK, EX_CHARS, N_EVAL = 1800, 4000, 50


def low_priority():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        p.cpu_affinity([0, 1])
    except Exception:  # noqa: BLE001
        pass


def download_websight():
    d = RAW / "WebSight"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for off in (0, 1_000_000):
        url = ("https://datasets-server.huggingface.co/rows?dataset=HuggingFaceM4/WebSight&config=v0.2"
               f"&split=train&offset={off}&length=100")
        with urllib.request.urlopen(url, timeout=60) as r:
            js = json.loads(r.read().decode("utf-8"))
        for x in js["rows"]:
            rows.append({"row_idx": x["row_idx"], "text": x["row"]["text"],
                         "llm_generated_idea": x["row"]["llm_generated_idea"]})
    (d / "websight_v0.2_rows200.json").write_text(json.dumps(rows), encoding="utf-8")
    return rows


def download_design2code():
    from huggingface_hub import snapshot_download
    d = RAW / "Design2Code"
    snapshot_download("SALT-NLP/Design2Code", repo_type="dataset", local_dir=d, allow_patterns=["*.html", "README.md"])
    return d


def page_info(html):
    from lxml import html as lhtml
    try:
        root = lhtml.document_fromstring(html)
    except Exception:  # noqa: BLE001
        return None
    for bad in root.xpath("//script|//style|//noscript"):
        bad.drop_tree()

    def txt(e):
        return re.sub(r"\s+", " ", e.text_content() or "").strip()
    title = txt(root.find(".//title")) if root.find(".//title") is not None else ""
    heads = [txt(e) for e in root.xpath("//h1|//h2|//h3")]
    btns = [txt(e) for e in root.xpath("//button|//a|//nav//li")]
    heads = [h for h in heads if 2 <= len(h) <= 60]
    btns = [b for b in btns if 2 <= len(b) <= 30]
    key = []
    for s in heads + btns:
        if s.lower() not in {k.lower() for k in key}:
            key.append(s)
    n_el = len(root.xpath("//body//*"))
    return {"title": title[:80], "heads": heads, "btns": btns, "key_texts": key[:15], "n_elements": n_el}


def d2c_query(info):
    """Deterministic query from the page (no LLM): purpose from title/h1, plus up to 4 salient items."""
    h1 = info["heads"][0] if info["heads"] else ""
    subject = info["title"] or h1 or "a website"
    parts = []
    if h1 and h1 != subject:
        parts.append(f'a main heading "{h1}"')
    for h in info["heads"][1:3]:
        parts.append(f'a section titled "{h}"')
    if info["btns"]:
        parts.append("links or buttons such as " + ", ".join(f'"{b}"' for b in info["btns"][:3]))
    q = f'Create a web page for "{subject}".'
    if parts:
        q += " Include " + "; ".join(parts[:4]) + "."
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    low_priority()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    OUT.mkdir(parents=True, exist_ok=True)
    if args.skip_download:
        ws = json.loads((RAW / "WebSight" / "websight_v0.2_rows200.json").read_text(encoding="utf-8"))
        d2c_dir = RAW / "Design2Code"
    else:
        ws = download_websight()
        d2c_dir = download_design2code()
    report = {}
    rng = random.Random(SEED)
    sets = {"websight": [], "design2code": []}
    for r in ws:
        info = page_info(r["text"]) or {"key_texts": [], "n_elements": 0}
        sets["websight"].append({"id": f"ws_{r['row_idx']}", "source": "WebSight v0.2", "query": r["llm_generated_idea"],
                                 "html": r["text"], "key_texts": info["key_texts"], "n_elements": info["n_elements"]})
    for p in sorted(d2c_dir.glob("*.html")):
        h = p.read_text(encoding="utf-8", errors="replace")
        info = page_info(h)
        if info is None:
            continue
        sets["design2code"].append({"id": f"d2c_{p.stem}", "source": "Design2Code", "query": d2c_query(info),
                                    "html": h, "key_texts": info["key_texts"], "n_elements": info["n_elements"]})
    for name, rows in sets.items():
        enc = tok.encode_batch([r["html"] for r in rows], add_special_tokens=False)
        for r, e in zip(rows, enc):
            r["html_chars"] = len(r["html"])
            r["n_tokens"] = len(e.ids)
            r["fits_gen"] = r["n_tokens"] <= GEN_TOK
            r["fits_exemplar"] = r["html_chars"] <= EX_CHARS
        fit = [r for r in rows if r["fits_gen"]]
        rng.shuffle(fit)
        ev = {r["id"] for r in fit[:N_EVAL]}
        for r in rows:
            r["split"] = "eval" if r["id"] in ev else "pool"
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        toks = sorted(r["n_tokens"] for r in rows)
        report[name] = {"rows": len(rows), "fits_gen": len(fit), "fits_exemplar": sum(r["fits_exemplar"] for r in rows),
                        "eval": len(ev), "pool": len(rows) - len(ev),
                        "pool_fits_exemplar": sum(r["fits_exemplar"] for r in rows if r["split"] == "pool"),
                        "tokens_p50": toks[len(toks) // 2], "tokens_p90": toks[int(len(toks) * 0.9)],
                        "tokens_max": toks[-1], "query_example": rows[0]["query"][:200] if rows else ""}
    (OUT / "prep_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
