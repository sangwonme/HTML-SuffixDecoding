"""P1 condition A post-hoc (CPU): format adherence + quality per transform and policy.

Format metrics on the extracted output HTML (and on the exemplars shown in the prompt, as reference):
  nl_per_kchar   newlines per 1000 chars                       (fmt -> up)
  inline_style   share of start tags carrying a style= attribute (css -> up)
  contents_wrap  share of start tags with display:contents      (dom -> up)
  std_class      share of class tokens from the original GT vocabulary (row, col, ph, icon, a, ...) (cls -> down)
Quality: checklist score (query_mid), element F1 / type F1 vs GT flow.
Baseline row "orig" uses runs/poc/gate2b (same harness, original exemplars).
Usage: .venv/Scripts/python.exe -I scripts/p1A_eval.py   -> runs/poc/p1A/p1A_eval.md, p1A_eval.jsonl
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "ms-playwright"))
from playwright.sync_api import sync_playwright  # noqa: E402

from genui.checklist import ChecklistScorer  # noqa: E402
from genui.metrics import Scorer, extract_html  # noqa: E402

P1 = ROOT / "runs" / "poc" / "p1A"
TRANS = ["fmt", "css", "dom", "cls"]
CONDS = ["pld", "dom", "ranker"]
STD = {"row", "col", "ph", "icon", "a", "pager", "bgimage", "ad", "webview", "map"}
TAG = re.compile(r"<([a-zA-Z][\w-]*)([^>]*)>")


def fmt_metrics(h: str) -> dict:
    if not h:
        return {}
    tags = TAG.findall(h)
    n = max(1, len(tags))
    cls = [c for m in re.finditer(r"""class\s*=\s*["']([^"']*)["']""", h) for c in m.group(1).split()]
    return {"nl_per_kchar": 1000 * h.count("\n") / max(1, len(h)),
            "inline_style": sum(1 for _, a in tags if re.search(r"\bstyle\s*=", a)) / n,
            "contents_wrap": sum(1 for _, a in tags if "display:contents" in a.replace(" ", "")) / n,
            "std_class": (sum(c in STD for c in cls) / len(cls)) if cls else float("nan")}


def rows(d, c):
    f = d / f"E2B_{c}.jsonl"
    return [r for r in (json.loads(l) for l in open(f, encoding="utf-8")) if "id" in r] if f.exists() else []


def mean(xs):
    xs = [x for x in xs if x == x]
    return statistics.mean(xs) if xs else float("nan")


def main():
    mid = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/mid/pilot.jsonl", encoding="utf-8")}
    gt = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/processed/pilot.jsonl", encoding="utf-8")}
    sets = [("orig", ROOT / "runs" / "poc" / "gate2b", None)] + [
        (t, P1 / t, ROOT / "runs" / "retrieval" / f"pilot_exemplar_html_{t}.json") for t in TRANS]
    recs, ex_rows = [], []
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=True)
        cs, es = ChecklistScorer(br), Scorer(br)
        gcache = {}
        for name, d, exf in sets:
            if exf and exf.exists():
                ex = json.load(open(exf, encoding="utf-8"))
                fm = [fmt_metrics(h) for h in ex.values()]
                ex_rows.append({"set": name, **{k: mean([f[k] for f in fm]) for k in fm[0]}})
            for c in CONDS:
                for r in rows(d, c):
                    if name == "orig" and r.get("rep", 0) != 0:
                        continue
                    h = extract_html(r["response"]) or ""
                    if r["id"] not in gcache:
                        gcache[r["id"]] = es.analyze(gt[r["id"]]["gt_html_flow"])
                    try:
                        ck = cs.score(r["response"], mid[r["id"]]["checklist"])
                    except Exception as e:  # noqa: BLE001  (page crash guard)
                        ck = {"check_score": float("nan")}
                        cs = ChecklistScorer(br)
                    try:
                        e = es.score(r["response"], gt[r["id"]], gcache[r["id"]])
                    except Exception:  # noqa: BLE001
                        e = {"elem_f1": float("nan"), "type_f1": float("nan")}
                        es = Scorer(br)
                    recs.append({"set": name, "cond": c, "id": r["id"], "total_s": r["total_s"], "n_fwd": r["n_fwd"],
                                 "tokens_per_fwd": r["tokens_per_fwd"], "prompt_tokens": r["prompt_tokens"],
                                 "ttft_s": r["ttft_s"], "output_tokens": r["output_tokens"],
                                 "check_score": ck["check_score"], "elem_f1": e["elem_f1"], "type_f1": e["type_f1"],
                                 **fmt_metrics(h)})
        br.close()
    with open(P1 / "p1A_eval.jsonl", "w", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    ks = ["nl_per_kchar", "inline_style", "contents_wrap", "std_class"]
    lines = ["## Format adherence (exemplars shown vs outputs)", "",
             "| set | source | n | " + " | ".join(ks) + " |", "|---|---|---|" + "---|" * len(ks)]
    for e in ex_rows:
        lines.append(f"| {e['set']} | exemplars | 150 | " + " | ".join(f"{e[k]:.3f}" for k in ks) + " |")
    for name, _, _ in sets:
        xs = [x for x in recs if x["set"] == name]
        if xs:
            lines.append(f"| {name} | outputs (all policies) | {len(xs)} | " + " | ".join(f"{mean([x.get(k, float('nan')) for x in xs]):.3f}" for k in ks) + " |")
    lines += ["", "## Quality and latency per set x policy (rep 0 for orig)", "",
              "| set | policy | n | checklist | elem F1 | type F1 | wall-clock mean (s) | forwards | tok/fwd | prompt tok | TTFT (s) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, _, _ in sets:
        for c in CONDS:
            xs = [x for x in recs if x["set"] == name and x["cond"] == c]
            if xs:
                lines.append(f"| {name} | {c.upper()} | {len(xs)} | {mean([x['check_score'] for x in xs]):.3f} | "
                             f"{mean([x['elem_f1'] for x in xs]):.3f} | {mean([x['type_f1'] for x in xs]):.3f} | "
                             f"{mean([x['total_s'] for x in xs]):.2f} | {mean([x['n_fwd'] for x in xs]):.1f} | "
                             f"{mean([x['tokens_per_fwd'] for x in xs]):.2f} | {mean([x['prompt_tokens'] for x in xs]):.0f} | "
                             f"{mean([x['ttft_s'] for x in xs]):.3f} |")
    out = "\n".join(lines) + "\n"
    (P1 / "p1A_eval.md").write_text(out, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
