"""S6: reorder generation (same tokens, different serialization order) -> perceptual latency.

Orders (same token multiset up to re-tokenization; uniform time per token, t/T axis):
  o  original order
  b  body-first: every <style>...</style> block moved to just before </body> (cascade => same final render;
     verified: final SSIM / identical-pixel check against original). Intermediate frames are rendered literally,
     i.e. UNSTYLED until the CSS arrives.
  s  Smith order (theoretical bound): head+CSS first (as in o), then top-level body blocks in decreasing
     w_i/p_i (w = rendered area inside the 360x640 viewport, p = tokens of the block). Absolute-slot assumption:
     each block appears at its final position when its last token is generated (rendered as the final page with
     not-yet-generated blocks set to visibility:hidden).
  bs b+s combined bound: CSS last (unstyled body blocks in Smith order are NOT simulated - see notes); reported
     only as o/b/s.
VC = SSIM-normalized against blank (same as S3), 20 checkpoints, VC=0 before first checkpoint, step-hold.

  .venv/Scripts/python.exe -I scripts/study_s6.py
Output: runs/study/s6_reorder.json, figures/s6_reorder.png
"""
from __future__ import annotations

import sys, pathlib, json, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from study_common import CONDS, FIG, OUT, Renderer, check_rss, encode, gray, guard_setup, load_runs, ssim
from study_s2_pld import element_spans
from genui.metrics import extract_html

NCP = 20
STYLE_RE = re.compile(r"<style[^>]*>.*?</style\s*>", re.S | re.I)

CONTAINER_JS = r"""() => { let c = document.body; if (!c) return null;
  while (c.children.length === 1 && c.children[0].children.length > 0) c = c.children[0];
  const out = [];
  for (const e of c.children) { const r = e.getBoundingClientRect();
    const w = Math.max(0, Math.min(r.right, 360) - Math.max(r.left, 0)), h = Math.max(0, Math.min(r.bottom, 640) - Math.max(r.top, 0));
    out.push({tag: e.tagName.toLowerCase(), area: w * h}); }
  return out; }"""

HIDE_JS = r"""(hide) => { let c = document.body;
  while (c.children.length === 1 && c.children[0].children.length > 0) c = c.children[0];
  const ch = [...c.children]; for (const i of hide) if (ch[i]) ch[i].style.visibility = 'hidden'; return ch.length; }"""


def body_first(html):
    styles = STYLE_RE.findall(html)
    if not styles:
        return html
    rest = STYLE_RE.sub("", html)
    i = rest.lower().rfind("</body")
    if i < 0:
        i = rest.lower().rfind("</html")
    if i < 0:
        i = len(rest)
    return rest[:i] + "".join(styles) + rest[i:]


def container_children_spans(html):
    """Text spans of the children of the first element (from <body>) that has >1 element child."""
    spans = sorted(element_spans(html))
    body = [s for s in spans if s[2] == "body"]
    if body:
        a, b = body[0][0], body[0][1]
        cur = (a, b, body[0][3])
    else:
        h = [s for s in spans if s[2] == "html"]
        if not h:
            return None
        cur = (h[0][0], h[0][1], h[0][3])
    def children(par):
        a, b, _ = par
        inside = [s for s in spans if a < s[0] and s[1] <= b and (s[0], s[1]) != (a, b)]
        direct = [s for s in inside if not any(t[0] < s[0] and s[1] <= t[1] for t in inside if t is not s)]
        return [s for s in direct if s[2] not in ("head", "style", "meta", "title", "link", "script")]
    for _ in range(10):
        ch = children(cur)
        if len(ch) == 1:
            s = ch[0]
            sub = children((s[0], s[1], s[3]))
            if sub:
                cur = (s[0], s[1], s[3])
                continue
        return ch
    return None


def vc(img, final_g, s0):
    return float(np.clip((ssim(gray(img), final_g) - s0) / max(1e-9, 1 - s0), 0, 1))


def lperc(cps, key):
    L, pt, pv = 0.0, 0.0, 0.0
    for cp in cps:
        L += (1 - pv) * (cp["f"] - pt)
        pt, pv = cp["f"], cp[key]
    return L


def t_at(cps, key, th):
    for cp in cps:
        if cp[key] >= th:
            return cp["f"]
    return 1.0


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
                yb = body_first(y)
                final = rend.shot(y)
                fg = gray(final)
                s0 = ssim(gray(blank), fg)
                fb = rend.shot(yb)
                ver = {"ssim_b_vs_o": ssim(gray(fb), fg),
                       "pix_identical_b": bool((fb == final).all()),
                       "pix_diff_ratio_b": float((np.abs(fb.astype(int) - final.astype(int)).max(-1) > 0).mean())}
                ids_o, off_o = encode(y)
                ids_b, off_b = encode(yb)
                T = len(ids_o)
                cps = []
                # Smith setup
                rend.page.set_content(y, wait_until="load", timeout=15000)
                blocks = rend.page.evaluate(CONTAINER_JS) or []
                spans = container_children_spans(y)
                smith_ok = bool(blocks) and spans is not None and len(spans) == len(blocks) and len(blocks) >= 2
                if smith_ok:
                    p = [len(encode(y[s[0]:s[1]])[0]) for s in spans]
                    H = len(encode(y[:spans[0][0]])[0])
                    order = sorted(range(len(blocks)), key=lambda i: -(blocks[i]["area"] / max(1, p[i])))
                    ends, acc = {}, H
                    for i in order:
                        acc += p[i]
                        ends[i] = acc
                for k_i in range(1, NCP + 1):
                    f = k_i / NCP
                    k = max(1, round(f * T))
                    cp = {"f": f}
                    po = y[:off_o[k - 1][1]] if k_i < NCP else y
                    cp["o"] = vc(rend.shot(po), fg, s0)
                    kb = max(1, round(f * len(ids_b)))
                    pb = yb[:off_b[kb - 1][1]] if k_i < NCP else yb
                    cp["b"] = vc(rend.shot(pb), fg, s0)
                    if smith_ok:
                        if k < H and k_i < NCP:
                            cp["s"] = cp["o"]
                        else:
                            hide = [i for i in range(len(blocks)) if ends[i] > k] if k_i < NCP else []
                            rend.page.set_content(y, wait_until="load", timeout=15000)
                            rend.page.evaluate(HIDE_JS, hide)
                            from PIL import Image
                            import io
                            img = np.asarray(Image.open(io.BytesIO(rend.page.screenshot(animations="disabled", caret="hide"))).convert("RGB"))
                            cp["s"] = vc(img, fg, s0)
                    cps.append(cp)
                row = {"id": r["id"], "T": T, "smith_ok": smith_ok, "n_blocks": len(blocks), **ver,
                       "Lperc_o": lperc(cps, "o"), "Lperc_b": lperc(cps, "b"),
                       "t50_o": t_at(cps, "o", 0.5), "t50_b": t_at(cps, "b", 0.5),
                       "t90_o": t_at(cps, "o", 0.9), "t90_b": t_at(cps, "b", 0.9), "checkpoints": cps}
                if smith_ok:
                    row.update({"Lperc_s": lperc(cps, "s"), "t50_s": t_at(cps, "s", 0.5), "t90_s": t_at(cps, "s", 0.9)})
                rows.append(row)
                check_rss()
            agg = {k: float(np.mean([x[k] for x in rows])) for k in ("Lperc_o", "Lperc_b", "t50_o", "t50_b", "t90_o", "t90_b", "ssim_b_vs_o", "pix_diff_ratio_b")}
            agg["pix_identical_b_rate"] = float(np.mean([x["pix_identical_b"] for x in rows]))
            sm = [x for x in rows if x["smith_ok"]]
            agg["smith_n"] = len(sm)
            if sm:
                for k in ("Lperc_s", "t50_s", "t90_s"):
                    agg[k] = float(np.mean([x[k] for x in sm]))
                agg["Lperc_o_on_smith_subset"] = float(np.mean([x["Lperc_o"] for x in sm]))
            res[c] = {"agg": agg, "samples": rows}
            print(c, {k: round(v, 3) for k, v in agg.items()}, flush=True)
            (OUT / "s6_reorder.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    finally:
        rend.close()
    fig(res)


def fig(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PAL = {"o": "#2a78d6", "b": "#eb6834", "s": "#1baf7a"}
    NAME = {"o": "(o) original", "b": "(b) body-first, CSS last", "s": "(s) Smith, slot bound"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    c = "E2B_nf4_rich_fs"
    rows = res[c]["samples"]
    pick = sorted(rows, key=lambda x: x["Lperc_o"] - x["Lperc_b"], reverse=True)
    pick = [pick[0], pick[len(pick) // 2], pick[-1]]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4), sharey=True)
    for ax, row in zip(axes[:3], pick):
        f = [0] + [cp["f"] for cp in row["checkpoints"]]
        for key in ("o", "b", "s"):
            if key in row["checkpoints"][0]:
                ax.step(f, [0] + [cp[key] for cp in row["checkpoints"]], where="post", lw=2, color=PAL[key], label=NAME[key])
        ax.set_title(f"{row['id']}  Lperc/T o={row['Lperc_o']:.2f} b={row['Lperc_b']:.2f}", loc="left", fontsize=9)
        ax.set_xlabel("decoding progress t/T")
    axes[0].set_ylabel("VC(t)")
    ax = axes[3]
    fr = [0] + [cp["f"] for cp in rows[0]["checkpoints"]]
    for key in ("o", "b", "s"):
        sub = [r for r in rows if key in r["checkpoints"][0]]
        if sub:
            m = np.mean([[cp[key] for cp in r["checkpoints"]] for r in sub], axis=0)
            ax.step(fr, [0] + list(m), where="post", lw=2, color=PAL[key], label=f"{NAME[key]} (n={len(sub)})")
    ax.set_title(f"mean, {c.replace('_nf4_', ' ')}", loc="left", fontsize=9)
    ax.set_xlabel("decoding progress t/T")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "s6_reorder.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
