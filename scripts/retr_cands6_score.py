"""Score runs/poc/E2B_L2_cands6_raw.jsonl (read only) -> runs/retrieval/cands6_scored.jsonl. CPU, headless.

The GPU-side scorer crashed at row 61 (Page crashed). This re-scores every row with the same scorers
(checklist with the target's query_mid checklist, element/type F1 vs target GT flow), recreating the browser
page after a crash. A row that crashes twice gets render_error and score 0.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from retr_struct import limit, load_split, rss_check  # noqa: E402
from genui.checklist import ChecklistScorer  # noqa: E402
from genui.metrics import Scorer  # noqa: E402

RAW = ROOT / "runs" / "poc" / "E2B_L2_cands6_raw.jsonl"
OUT = ROOT / "runs" / "retrieval" / "cands6_scored.jsonl"


def main():
    limit()
    from playwright.sync_api import sync_playwright
    tgt = {r["id"]: r for r in load_split("pilot")}
    rows = [json.loads(l) for l in open(RAW, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if "_meta" not in r]
    with sync_playwright() as pw:
        kw = {}
        hs = sorted(Path(os.environ["LOCALAPPDATA"], "ms-playwright").glob(
            "chromium_headless_shell-*/chrome-headless-shell-win64/chrome-headless-shell.exe"))
        if hs and not Path(pw.chromium.executable_path).exists():
            kw["executable_path"] = str(hs[-1])
        br = pw.chromium.launch(headless=True, args=["--disable-gpu", "--renderer-process-limit=1"], **kw)

        def fresh():
            return ChecklistScorer(br), Scorer(br)

        cs, ms = fresh()
        gcache = {}
        n_err = 0
        with open(OUT, "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                t = tgt[r["id"]]
                res = None
                for attempt in range(2):
                    try:
                        if r["id"] not in gcache:
                            gcache[r["id"]] = ms.analyze(t["gt_html_flow"])
                        c = cs.score(r.get("response") or "", t["checklist"])
                        e = ms.score(r.get("response") or "", t, gt_analysis=gcache[r["id"]])
                        if "render_error" in c or "render_error" in e:
                            raise RuntimeError(c.get("render_error") or e.get("render_error"))
                        res = {"check_score": c["check_score"], "check_all": c["check_all"],
                               "elem_f1": e["elem_f1"], "type_f1": e["type_f1"], "render_ok": e["render_ok"]}
                        break
                    except Exception as ex:  # noqa: BLE001
                        for o in (cs, ms):
                            try:
                                o.ctx.close()
                            except Exception:  # noqa: BLE001
                                pass
                        cs, ms = fresh()
                        err = str(ex)[:200]
                if res is None:
                    n_err += 1
                    res = {"check_score": 0.0, "check_all": False, "elem_f1": 0.0, "type_f1": 0.0,
                           "render_ok": False, "render_error": err}
                out = {k: v for k, v in r.items() if k != "response"}
                f.write(json.dumps({**out, **res}) + "\n")
                if i % 50 == 0:
                    print(f"{i}/{len(rows)} rss {rss_check():.2f}GB", flush=True)
        br.close()
    print(f"scored {len(rows)}, render errors {n_err}")


if __name__ == "__main__":
    main()
