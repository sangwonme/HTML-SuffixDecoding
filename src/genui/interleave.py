"""CSS interleaving (i') and perceptual-latency helpers.  See POC_study.md section 6b.

interleave(html, page=None) -> html
    Moves every CSS rule of the document's <style> blocks, VERBATIM (original text, incl. its leading whitespace /
    comments), into a new <style> placed right before the first top-level body block it applies to.
    - top-level blocks = children of <body>, descending while there is a single wrapper child that has children
    - rule -> block assignment: document.querySelectorAll(selector) in the rendered DOM (headless Chromium);
      rules that hit html/body/wrapper or anything outside the blocks, @-rules and unparsable selectors stay in a
      global <style> at the position of the first original <style>; rules matching nothing go to the end of the
      container.
    - block order is unchanged; only extra tokens are the added <style></style> tags.
    Returns the input unchanged on any failure (no <style>, <2 blocks, rule-count mismatch between source text
    and CSSOM, @-rules present, parse problems).
    page: optional Playwright page (360x640, JS may be disabled - page.evaluate still works). If None, a
    private headless Chromium is started and closed per call (slow; pass a page for batches).

perceptual_latency(html, token_times, n_checkpoints=20, page=None) -> dict
    html: generated output (raw response text incl. ```html fence is fine) ; token_times: list of
    (char_end_offset, t_seconds) per generated token, in order (char_end_offset = len(text) after that token,
    t = wall-clock time since request start). Renders 20 prefixes, VC = SSIM-normalized visual completeness vs
    the final render (VC=0 before the first checkpoint, step-hold).  Returns
    {"Lperc_s", "T_s", "Lperc_over_T", "t50_s", "t50_over_T", "t90_over_T", "checkpoints": [...]}
"""
from __future__ import annotations

import io
import re

import numpy as np

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
STYLE_RE = re.compile(r"<style[^>]*>.*?</style\s*>", re.S | re.I)
RULE_RE = re.compile(r"[^{}]+\{[^{}]*\}")

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
     out.push(idx); } }
 return {idx: out, n_blocks: ch.length}; }"""


# ------------------------------------------------------------------ text-level element spans
def _element_spans(t):
    skip = [(m.start(), m.end()) for m in re.finditer(r"<!--.*?-->|<style[^>]*>.*?</style\s*>|<script[^>]*>.*?</script\s*>", t, re.S | re.I)]
    out, stack = [], []
    for m in re.finditer(r"<(/?)([a-zA-Z][\w-]*)[^<>]*>", t):
        if any(a < m.start() < b for a, b in skip):
            continue
        name = m.group(2).lower()
        if m.group(1):
            if any(s[2] == name for s in stack):
                while stack:
                    s = stack.pop()
                    out.append((s[0], m.end() if s[2] == name else m.start(), s[2]))
                    if s[2] == name:
                        break
            continue
        if name in VOID or m.group().endswith("/>"):
            out.append((m.start(), m.end(), name))
        else:
            stack.append((m.start(), m.end(), name))
    for s in stack:
        out.append((s[0], len(t), s[2]))
    return sorted(out)


def _blocks(t):
    spans = _element_spans(t)
    top = [s for s in spans if s[2] == "body"] or [s for s in spans if s[2] == "html"]
    if not top:
        return None
    cur = top[0]

    def children(par):
        a, b = par[0], par[1]
        inside = [s for s in spans if a < s[0] and s[1] <= b and (s[0], s[1]) != (a, b)]
        direct = [s for s in inside if not any(u[0] < s[0] and s[1] <= u[1] for u in inside if u is not s)]
        return [s for s in direct if s[2] not in ("head", "style", "meta", "title", "link", "script")]

    for _ in range(12):
        ch = children(cur)
        if len(ch) == 1 and children(ch[0]):
            cur = ch[0]
            continue
        return ch
    return None


# ------------------------------------------------------------------ browser helpers
class _Browser:
    def __init__(self):
        from playwright.sync_api import sync_playwright
        self.pw = sync_playwright().start()
        self.br = self.pw.chromium.launch(headless=True)
        self.ctx = self.br.new_context(viewport={"width": 360, "height": 640}, java_script_enabled=False)
        self.ctx.route("**/*", lambda route: route.abort())
        self.page = self.ctx.new_page()

    def close(self):
        self.br.close()
        self.pw.stop()


def _with_page(fn, page):
    if page is not None:
        return fn(page)
    b = _Browser()
    try:
        return fn(b.page)
    finally:
        b.close()


# ------------------------------------------------------------------ interleave
def _interleave_with(html, page):
    styles = list(STYLE_RE.finditer(html))
    if not styles:
        return html
    orig = []
    for m in styles:
        body = re.sub(r"^<style[^>]*>|</style\s*>$", "", m.group(), flags=re.I)
        orig += [x.group() for x in RULE_RE.finditer(body)]
    if not orig or "@" in "".join(orig):
        return html
    spans = _blocks(html)
    if not spans or len(spans) < 2:
        return html
    page.set_content(html, wait_until="load", timeout=15000)
    info = page.evaluate(ASSIGN_JS)
    if not info or len(info["idx"]) != len(orig) or info["n_blocks"] != len(spans):
        return html
    glob = "".join(o for o, i in zip(orig, info["idx"]) if i == -2)
    dead = "".join(o for o, i in zip(orig, info["idx"]) if i == -1)
    per = {}
    for o, i in zip(orig, info["idx"]):
        if i >= 0:
            per.setdefault(i, []).append(o)
    edits = []
    for j, m in enumerate(styles):
        edits.append((m.start(), m.end(), f"<style>{glob}</style>" if (j == 0 and glob) else ""))
    for i, s in enumerate(spans):
        if i in per:
            edits.append((s[0], s[0], "<style>" + "".join(per[i]) + "</style>"))
    if dead:
        edits.append((spans[-1][1], spans[-1][1], f"<style>{dead}</style>"))
    out, pos = [], 0
    for a, b, txt in sorted(edits, key=lambda e: (e[0], e[1])):
        out.append(html[pos:a]); out.append(txt); pos = max(pos, b)
    out.append(html[pos:])
    return "".join(out)


def interleave(html: str, page=None) -> str:
    try:
        return _with_page(lambda p: _interleave_with(html, p), page)
    except Exception:
        return html


# ------------------------------------------------------------------ perceptual latency
def _ssim(a, b):
    from scipy.ndimage import uniform_filter
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = uniform_filter(a, 7), uniform_filter(b, 7)
    saa = uniform_filter(a * a, 7) - mu_a ** 2
    sbb = uniform_filter(b * b, 7) - mu_b ** 2
    sab = uniform_filter(a * b, 7) - mu_a * mu_b
    return float((((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))).mean())


def _gray_shot(page, html):
    from PIL import Image
    page.set_content(html or "", wait_until="load", timeout=15000)
    img = np.asarray(Image.open(io.BytesIO(page.screenshot(animations="disabled", caret="hide"))).convert("RGB"))
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def _strip_fence(s):
    i = s.find("<")
    s = s[i:] if i >= 0 else ""
    j = s.find("```")
    return s[:j] if j >= 0 else s


def perceptual_latency(html: str, token_times, n_checkpoints: int = 20, page=None) -> dict:
    def run(p):
        T = len(token_times)
        final = _gray_shot(p, _strip_fence(html))
        blank = _gray_shot(p, "<html><body></body></html>")
        s0 = _ssim(blank, final)
        cps = []
        for k in range(1, n_checkpoints + 1):
            idx = max(1, round(k / n_checkpoints * T)) - 1
            end, t = token_times[idx]
            g = _gray_shot(p, _strip_fence(html[:end]) if k < n_checkpoints else _strip_fence(html))
            vc = float(np.clip((_ssim(g, final) - s0) / max(1e-9, 1 - s0), 0, 1))
            cps.append({"t": float(t), "vc": vc})
        Ts = float(token_times[-1][1])
        L, pt, pv = 0.0, 0.0, 0.0
        for cp in cps:
            L += (1 - pv) * (cp["t"] - pt); pt, pv = cp["t"], cp["vc"]
        t50 = next((cp["t"] for cp in cps if cp["vc"] >= 0.5), Ts)
        t90 = next((cp["t"] for cp in cps if cp["vc"] >= 0.9), Ts)
        return {"Lperc_s": L, "T_s": Ts, "Lperc_over_T": L / Ts if Ts else None, "t50_s": t50,
                "t50_over_T": t50 / Ts if Ts else None, "t90_over_T": t90 / Ts if Ts else None, "checkpoints": cps}
    return _with_page(run, page)
