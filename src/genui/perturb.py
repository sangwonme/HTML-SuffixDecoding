"""Render-preserving HTML perturbations for the representation-robustness study (P1).

Each f(html, seed, page=None) -> html returns the input unchanged on failure.  Render identity is NOT checked
here; callers verify with a full-page screenshot (see scripts/study_perturb_exemplars.py).

fmt  Formatting: newline + indentation between tags (outside <style>/<title>), attribute order shuffled, some
     attribute values switched to single quotes.
css  CSS representation: every <style> rule whose selector can be matched is expanded into inline style="..."
     on the matching elements (cascade order: specificity, then source order; existing inline style last);
     fully-inlined rules are removed. Needs a Playwright page (DOM serialization of the result).
dom  DOM structure: wraps a random ~40% of the elements inside <body> (non-void, not style/script) in
     <div style="display:contents">.
cls  Class names: consistent random renaming in class attributes and .selectors in <style>.
"""
from __future__ import annotations

import random
import re

from genui.interleave import _element_spans, _with_page

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
STYLE_RE = re.compile(r"<style[^>]*>.*?</style\s*>", re.S | re.I)
ATTR_RE = re.compile(r"""\s+([^\s=/>]+)(?:\s*=\s*("[^"]*"|'[^']*'|[^\s>]+))?""")


def _protected(html):
    return [(m.start(), m.end()) for m in re.finditer(r"<style[^>]*>.*?</style\s*>|<title[^>]*>.*?</title\s*>|<!--.*?-->",
                                                      html, re.S | re.I)]


# ------------------------------------------------------------------ fmt
def fmt(html: str, seed: int = 0, page=None) -> str:
    try:
        rng = random.Random(seed)
        prot = _protected(html)

        def in_prot(i):
            return any(a <= i < b for a, b in prot)

        def retag(m):
            if in_prot(m.start()):
                return m.group()
            tag = m.group(1)
            body = m.group(2)
            end = m.group(3)
            attrs = ATTR_RE.findall(body)
            if not attrs or body.strip() != body.strip():
                pass
            parts = []
            for name, val in attrs:
                if val and val.startswith('"') and "'" not in val and rng.random() < 0.5:
                    val = "'" + val[1:-1] + "'"
                parts.append(f"{name}={val}" if val else name)
            rng.shuffle(parts)
            return "<" + tag + "".join(" " + p for p in parts) + end
        out = re.sub(r"<([a-zA-Z][\w-]*)((?:\s+[^\s=/>]+(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s>]+))?)*)\s*(/?>)", retag, html)
        # newline + indentation between adjacent tags (depth-based), outside protected regions
        prot = _protected(out)
        res, depth, i = [], 0, 0
        for m in re.finditer(r"<(/?)([a-zA-Z][\w-]*)[^<>]*>", out):
            if any(a <= m.start() < b for a, b in prot):
                continue
            s = m.start()
            if s > 0 and out[s - 1] == ">":
                d = depth - 1 if m.group(1) else depth
                res.append(out[i:s]); res.append("\n" + "  " * max(0, d)); i = s
            name = m.group(2).lower()
            if m.group(1):
                depth = max(0, depth - 1)
            elif name not in VOID and not m.group().endswith("/>"):
                depth += 1
        res.append(out[i:])
        return "".join(res)
    except Exception:
        return html


# ------------------------------------------------------------------ css
INLINE_JS = r"""() => {
 function spec(sel){ const s = sel.replace(/::?[a-zA-Z-]+(\([^)]*\))?/g, m => { if (m.startsWith('::')) return ' '; return ' .p'; });
   const a = (s.match(/#[\w-]+/g)||[]).length, b = (s.match(/\.[\w-]+|\[[^\]]+\]/g)||[]).length,
         c = (s.replace(/#[\w-]+|\.[\w-]+|\[[^\]]+\]/g,' ').match(/(^|[\s>+~])[a-zA-Z][\w-]*/g)||[]).length; return a*10000+b*100+c; }
 const decl = new Map(); const keepRules = []; let order = 0;
 const sheets = [...document.styleSheets];
 for (const sh of sheets) { for (const r of sh.cssRules) { order++;
   if (!(r instanceof CSSStyleRule) || /::|:hover|:focus|:active|:visited|:checked|:before|:after|:placeholder/.test(r.selectorText)) { keepRules.push(r.cssText); continue; }
   let ok = true; const parts = r.selectorText.split(',').map(x=>x.trim());
   for (const p of parts) { let els; try { els = document.querySelectorAll(p); } catch(e) { ok = false; break; }
     for (const e of els) { if (e.closest('head')) continue; if (!decl.has(e)) decl.set(e, []); decl.get(e).push([spec(p), order, r.style.cssText]); } }
   if (!ok) { keepRules.push(r.cssText); } } }
 for (const [e, list] of decl) { list.sort((x,y)=> x[0]-y[0] || x[1]-y[1]);
   const own = e.getAttribute('style') || ''; e.setAttribute('style', list.map(x=>x[2]).join(' ') + (own ? ' ' + own : '')); }
 for (const s of [...document.querySelectorAll('style')]) s.remove();
 if (keepRules.length) { const st = document.createElement('style'); st.textContent = keepRules.join(''); document.head.appendChild(st); }
 return '<!DOCTYPE html>' + document.documentElement.outerHTML; }"""


def css(html: str, seed: int = 0, page=None) -> str:
    try:
        def run(p):
            p.set_content(html, wait_until="load", timeout=15000)
            return p.evaluate(INLINE_JS)
        out = _with_page(run, page)
        return out or html
    except Exception:
        return html


# ------------------------------------------------------------------ dom
def dom(html: str, seed: int = 0, page=None, frac: float = 0.4) -> str:
    try:
        rng = random.Random(seed)
        spans = _element_spans(html)
        body = [s for s in spans if s[2] == "body"]
        if not body:
            return html
        b0, b1 = body[0][0], body[0][1]
        cand = [s for s in spans if b0 < s[0] and s[1] <= b1 and s[2] not in VOID | {"style", "script", "body"}]
        pick = [s for s in cand if rng.random() < frac]
        if not pick:
            return html
        edits = []
        for a, b, _ in pick:
            edits.append((a, 0, '<div style="display:contents">'))
            edits.append((b, 1, "</div>"))
        # apply: opening tags before closing at same position must nest correctly -> sort by pos, closes first
        edits.sort(key=lambda e: (e[0], -e[1]))
        out, pos = [], 0
        for p, kind, txt in edits:
            out.append(html[pos:p]); out.append(txt); pos = p
        out.append(html[pos:])
        return "".join(out)
    except Exception:
        return html


# ------------------------------------------------------------------ cls
def cls(html: str, seed: int = 0, page=None) -> str:
    try:
        rng = random.Random(seed)
        names = []
        for m in re.finditer(r"""class\s*=\s*["']?([^"'>]*)""", html):
            for n in m.group(1).split():
                if n not in names:
                    names.append(n)
        if not names:
            return html
        letters = "abcdefghijklmnopqrstuvwxyz"
        new = {}
        for n in names:
            while True:
                c = "".join(rng.choice(letters) for _ in range(rng.randint(3, 8)))
                if c not in new.values() and c not in names:
                    new[n] = c; break
        out = re.sub(r"""(class\s*=\s*["']?)([^"'>]*)""", lambda m: m.group(1) + " ".join(new.get(x, x) for x in m.group(2).split()), html)

        def ren_style(m):
            body = m.group()
            return re.sub(r"([^{}]+)(\{[^{}]*\})",
                          lambda r: re.sub(r"\.(-?[_a-zA-Z][\w-]*)", lambda q: "." + new.get(q.group(1), q.group(1)), r.group(1)) + r.group(2),
                          body)
        return STYLE_RE.sub(ren_style, out)
    except Exception:
        return html


PERTURBATIONS = {"fmt": fmt, "css": css, "dom": dom, "cls": cls}
