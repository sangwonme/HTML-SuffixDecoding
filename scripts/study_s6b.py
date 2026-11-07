"""S6b: CSS interleaving (UI-specific reorder that keeps styled frames).

  i   interleaved: each CSS rule is emitted in a <style> right before the first top-level body block it applies
      to (assignment via querySelectorAll in the final DOM). Rules hitting elements outside the blocks
      (html/body/container, @-rules) stay in a global <style> at the original position; rules matching nothing
      (dead) go to the end of <body>. Blocks keep their original order. Rule text = browser cssText, so the token
      count changes slightly (reported). Final render verified against the original.
  si  i + Smith block order (w/p desc), slot bound like S6 (s): block appears at its final position once its
      rules+tokens are complete; frame = final page with not-yet-generated blocks hidden.
Prefix frames of i are rendered literally (honest). Output: runs/study/s6b_interleave.json
"""
from __future__ import annotations

import sys, pathlib, json, io
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
from PIL import Image

from study_common import OUT, Renderer, check_rss, encode, gray, guard_setup, load_runs, ssim
from study_s6 import CONTAINER_JS, HIDE_JS, NCP, STYLE_RE, container_children_spans, lperc, t_at, vc
from genui.metrics import extract_html

ASSIGN_JS = r"""() => { let c = document.body; if (!c) return null;
 while (c.children.length === 1 && c.children[0].children.length > 0) c = c.children[0];
 const ch = [...c.children]; const out = [];
 for (const sh of document.styleSheets) { let rules; try { rules = sh.cssRules; } catch (e) { continue; }
   for (const r of rules) { let idx = -1;
     if (r.selectorText) { let els = [];
       try { els = document.querySelectorAll(r.selectorText); } catch (e) { idx = -2; }
       for (const e of els) { const i = ch.findIndex(b => b === e || b.contains(e));
         if (i < 0) { idx = -2; break; } idx = (idx < 0) ? i : Math.min(idx, i); } }
     else idx = -2;
     out.push({css: r.cssText, idx}); } }
 return out; }"""


def interleave(y, rules, spans):
    import re as _re
    from study_s2_eta import minify_css
    orig = []
    for m in STYLE_RE.finditer(y):
        body = _re.sub(r"^<style[^>]*>|</style\s*>$", "", m.group(), flags=_re.I)
        orig += [x.group() for x in _re.finditer(r"[^{}]+\{[^{}]*\}", body)]  # verbatim (leading ws/comments kept)
    if len(orig) == len(rules) and "@" not in "".join(orig):
        rules = [dict(r, css=o) for r, o in zip(rules, orig)]   # original rule text (same tokens)
        interleave.used_orig = True
    else:
        rules = [dict(r, css=minify_css(r["css"])) for r in rules]
        interleave.used_orig = False
    glob = "".join(r["css"] for r in rules if r["idx"] == -2)
    dead = "".join(r["css"] for r in rules if r["idx"] == -1)
    per = {}
    for r in rules:
        if r["idx"] >= 0:
            per.setdefault(r["idx"], []).append(r["css"])
    edits = []  # (pos, end, text)
    sb = list(STYLE_RE.finditer(y))
    for j, m in enumerate(sb):
        edits.append((m.start(), m.end(), f"<style>{glob}</style>" if (j == 0 and glob) else ""))
    if not sb and glob:
        edits.append((spans[0][0], spans[0][0], f"<style>{glob}</style>"))
    for i, s in enumerate(spans):
        if i in per:
            edits.append((s[0], s[0], "<style>" + "".join(per[i]) + "</style>"))
    endpos = spans[-1][1]
    if dead:
        edits.append((endpos, endpos, f"<style>{dead}</style>"))
    out, pos = [], 0
    for a, b, t in sorted(edits, key=lambda e: (e[0], e[1])):
        out.append(y[pos:a]); out.append(t); pos = max(pos, b)
    out.append(y[pos:])
    per = {i: [x for x in v] for i, v in per.items()}
    return "".join(out), per


def shot_hidden(rend, html, hide):
    rend.page.set_content(html, wait_until="load", timeout=15000)
    rend.page.evaluate(HIDE_JS, hide)
    return np.asarray(Image.open(io.BytesIO(rend.page.screenshot(animations="disabled", caret="hide"))).convert("RGB"))


def main():
    guard_setup()
    pilot, runs = load_runs()
    rend = Renderer()
    res = {}
    try:
        blank = rend.shot("<html><body></body></html>")
        for c in ["E2B_nf4_rich_fs", "E4B_nf4_rich_fs", "E2B_nf4_rich_zs", "E4B_nf4_rich_zs", "12B_nf4_rich_fs",
                  "12B_nf4_rich_zs", "E2B_nf4_short_zs"]:
            rows = []
            for r in runs[c]:
                y = extract_html(r["response"])
                final = rend.shot(y)
                fg = gray(final)
                s0 = ssim(gray(blank), fg)
                rend.page.set_content(y, wait_until="load", timeout=15000)
                rules = rend.page.evaluate(ASSIGN_JS) or []
                blocks = rend.page.evaluate(CONTAINER_JS) or []
                spans = container_children_spans(y)
                if not rules or not spans or len(spans) != len(blocks) or len(blocks) < 2:
                    rows.append({"id": r["id"], "ok": False})
                    continue
                yi, per = interleave(y, rules, spans)
                used_orig = interleave.used_orig
                fi = rend.shot(yi)
                ids_o, _ = encode(y)
                ids_i, off_i = encode(yi)
                T, Ti = len(ids_o), len(ids_i)
                row = {"id": r["id"], "ok": True, "T": T, "T_i": Ti, "tok_ratio_i": Ti / T, "extra_tok": Ti - T,
                       "ssim_i_vs_o": ssim(gray(fi), fg), "pix_identical_i": bool((fi == final).all()),
                       "pix_diff_ratio_i": float((np.abs(fi.astype(int) - final.astype(int)).max(-1) > 0).mean()),
                       "n_rules": len(rules), "used_orig_rule_text": used_orig, "global_rules": sum(1 for x in rules if x["idx"] == -2),
                       "dead_rules": sum(1 for x in rules if x["idx"] == -1)}
                # Smith on interleaved units
                unit_tok = []
                for i, s in enumerate(spans):
                    css = "<style>" + "".join(per[i]) + "</style>" if i in per else ""  # per already minified
                    unit_tok.append(len(encode(css + y[s[0]:s[1]])[0]))
                first_block_pos = yi.find(y[spans[0][0]:spans[0][1]])
                pre = yi[:first_block_pos]
                if 0 in per:
                    pre = pre[:pre.rfind("<style>")]
                H = len(encode(pre)[0])
                order = sorted(range(len(blocks)), key=lambda i: -(blocks[i]["area"] / max(1, unit_tok[i])))
                ends, acc = {}, H
                for i in order:
                    acc += unit_tok[i]; ends[i] = acc
                cps = []
                for k_i in range(1, NCP + 1):
                    f = k_i / NCP
                    k = max(1, round(f * Ti))
                    cp = {"f": f}
                    cp["i"] = vc(rend.shot(yi[:off_i[k - 1][1]] if k_i < NCP else yi), fg, s0)
                    if k < H and k_i < NCP:
                        cp["si"] = cp["i"]
                    else:
                        hide = [i for i in range(len(blocks)) if ends[i] > k] if k_i < NCP else []
                        cp["si"] = vc(shot_hidden(rend, y, hide), fg, s0)
                    cps.append(cp)
                row.update({"Lperc_i_absT": lperc(cps, "i") * Ti / T, "Lperc_si_absT": lperc(cps, "si") * Ti / T,
                            "Lperc_i": lperc(cps, "i"), "Lperc_si": lperc(cps, "si"),
                            "t50_i": t_at(cps, "i", 0.5), "t50_si": t_at(cps, "si", 0.5),
                            "t90_i": t_at(cps, "i", 0.9), "t50_i_absT": t_at(cps, "i", 0.5) * Ti / T, "head_share_i": H / Ti, "checkpoints": cps})
                rows.append(row)
                check_rss()
            ok = [x for x in rows if x["ok"]]
            agg = {"n_ok": len(ok)}
            for k in ("extra_tok", "Lperc_i_absT", "Lperc_si_absT", "Lperc_i", "Lperc_si", "t50_i", "t50_i_absT", "t50_si", "t90_i", "ssim_i_vs_o", "pix_diff_ratio_i", "tok_ratio_i", "head_share_i"):
                agg[k] = float(np.mean([x[k] for x in ok])) if ok else None
            agg["orig_text_rate"] = float(np.mean([x["used_orig_rule_text"] for x in ok])) if ok else None
            agg["pix_identical_i_rate"] = float(np.mean([x["pix_identical_i"] for x in ok])) if ok else None
            agg["render_equal_rate_ssim999"] = float(np.mean([x["ssim_i_vs_o"] >= 0.999 for x in ok])) if ok else None
            res[c] = {"agg": agg, "samples": rows}
            print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items()}, flush=True)
            (OUT / "s6b_interleave_fair.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    finally:
        rend.close()


if __name__ == "__main__":
    main()
