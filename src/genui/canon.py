"""Render-quotient canonical form kappa(y), copy coverage omega_n, and UI structure signature phi(y).

kappa(y): parse HTML leniently (html5lib if installed, else lxml.html), drop comments and
  whitespace-only text, drop data-* attributes, sort attributes, rename CSS classes c0, c1, ... by
  order of first appearance (class attributes AND <style> selectors), serialize compactly, then
  tokenize with the Gemma tokenizer (tokenizer.json only, CPU).
omega_n(e, y): fraction of kappa(y) tokens covered by an n-gram (default n=4) that also occurs in kappa(e).
  (start>0 restricts the fraction to tokens from `start` on, e.g. body only via body_offset.)
phi(y): from gt_html_abs (data-label + absolute positions, no rendering). Normalized histogram over
  (category x vertical band: top <20%, middle, bottom >80% of 640px, by element center) concatenated
  with parent->child category bigrams (parent = nearest labeled ancestor, else ROOT). Each block is
  L1-normalized, then the concatenation is scaled to unit L2 norm (cosine-ready).
"""
from __future__ import annotations

import re
from html import escape
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_PATH = ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized" / "tokenizer.json"

try:  # spec: html5lib preferred, lxml.html fallback
    import html5lib  # type: ignore  # noqa: F401
    _PARSER = "html5lib"
except ImportError:
    _PARSER = "lxml"
from lxml import etree, html as lhtml  # noqa: E402

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
_CSS_CLASS = re.compile(r"\.(-?[A-Za-z_][\w-]*)")
_CSS_BLOCK = re.compile(r"\{[^{}]*\}")


def _parse(html: str):
    if _PARSER == "html5lib":
        import html5lib
        return html5lib.parse(html, treebuilder="lxml", namespaceHTMLElements=False).getroot()
    return lhtml.document_fromstring(html)


def canon_html(html: str) -> str:
    """kappa(y) as a string (before tokenization)."""
    root = _parse(html)
    names: dict[str, str] = {}

    def nm(c: str) -> str:
        if c not in names:
            names[c] = f"c{len(names)}"
        return names[c]

    def css(text: str) -> str:
        # rename only in selectors (outside {...} blocks), keep declarations as-is
        out, last = [], 0
        for m in _CSS_BLOCK.finditer(text):
            out.append(_CSS_CLASS.sub(lambda k: "." + nm(k.group(1)), text[last:m.start()]))
            out.append(m.group(0))
            last = m.end()
        out.append(_CSS_CLASS.sub(lambda k: "." + nm(k.group(1)), text[last:]))
        return "".join(out)

    def txt(s: str | None) -> str:
        return escape(s, quote=False) if s and s.strip() else ""

    def ser(el) -> str:
        if not isinstance(el.tag, str):  # comment / PI
            return txt(el.tail)
        tag = el.tag.lower()
        attrs = []
        for k, v in el.attrib.items():
            k = k.lower()
            if k.startswith("data-"):
                continue
            if k == "class":
                v = " ".join(nm(c) for c in v.split())
            attrs.append((k, v))
        attrs.sort()
        a = "".join(f' {k}="{escape(v)}"' if v != "" else f" {k}" for k, v in attrs)
        if tag in VOID:
            return f"<{tag}{a}>" + txt(el.tail)
        if tag == "style":
            inner = css(el.text or "")
        elif tag == "script":
            inner = el.text or ""
        else:
            inner = txt(el.text) + "".join(ser(c) for c in el)
        return f"<{tag}{a}>{inner}</{tag}>" + txt(el.tail)

    return ser(root)


_TOK = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(str(TOKENIZER_PATH))
    return _TOK


def kappa_ids(htmls: list[str]) -> list[np.ndarray]:
    encs = tokenizer().encode_batch([canon_html(h) for h in htmls], add_special_tokens=False)
    return [np.asarray(e.ids, dtype=np.int64) for e in encs]


_P = np.uint64(1000003)


def ngram_hashes(ids: np.ndarray, n: int = 4) -> np.ndarray:
    """uint64 rolling hash of each n-gram (wraparound; collisions negligible)."""
    m = len(ids) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    u = ids.astype(np.uint64) + np.uint64(1)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * _P + u[k:k + m]
    return h


def body_offset(html: str) -> int:
    """Token count of the canonical prefix before <body> (diagnostic: head/style boilerplate)."""
    s = canon_html(html)
    i = s.find("<body")
    return len(tokenizer().encode(s[:i] if i >= 0 else "", add_special_tokens=False).ids)


def omega(y_hash: np.ndarray, e_set: np.ndarray, n_tokens_y: int, n: int = 4, start: int = 0) -> float:
    """omega_n(e, y). y_hash = ngram_hashes(kappa(y)); e_set = np.unique(ngram_hashes(kappa(e)))."""
    if n_tokens_y == 0:
        return 0.0
    if len(y_hash) == 0 or len(e_set) == 0:
        return 0.0
    pos = np.searchsorted(e_set, y_hash)
    pos[pos == len(e_set)] = 0
    hit = e_set[pos] == y_hash
    cov = np.zeros(n_tokens_y, dtype=bool)
    m = len(y_hash)
    for k in range(n):
        cov[k:k + m] |= hit
    cov = cov[start:]
    return float(cov.mean()) if len(cov) else 0.0


# ----------------------------------------------------------------------------- phi
LABELS = ["Advertisement", "Background Image", "Bottom Navigation", "Button Bar", "Card", "Checkbox",
          "Date Picker", "Drawer", "Icon", "Image", "Input", "List Item", "Map View", "Modal", "Multi-Tab",
          "Number Stepper", "On/Off Switch", "Pager Indicator", "Radio Button", "Slider", "Text",
          "Text Button", "Toolbar", "Video", "Web View", "OTHER"]
_LI = {l: i for i, l in enumerate(LABELS)}
PARENTS = ["ROOT"] + LABELS
_PI = {l: i for i, l in enumerate(PARENTS)}
NB = 3
D_BAND = len(LABELS) * NB
D_BIG = len(PARENTS) * len(LABELS)
D_PHI = D_BAND + D_BIG
_NUM = re.compile(r"(left|top|width|height)\s*:\s*(-?[\d.]+)px")


def _box(style: str) -> dict:
    return {k: float(v) for k, v in _NUM.findall(style or "")}


def phi(html_abs: str, H: float = 640.0) -> np.ndarray:
    root = lhtml.document_fromstring(html_abs)
    band = np.zeros(D_BAND)
    big = np.zeros(D_BIG)

    def walk(el, parent_lab: str, ox: float, oy: float):
        for c in el:
            if not isinstance(c.tag, str):
                continue
            lab = c.get("data-label")
            if lab is not None:
                li = _LI.get(lab, _LI["OTHER"])
                b = _box(c.get("style"))
                top = oy + b.get("top", 0.0)
                cy = top + b.get("height", 0.0) / 2
                bi = 0 if cy < 0.2 * H else (2 if cy > 0.8 * H else 1)
                band[li * NB + bi] += 1
                big[_PI[parent_lab] * len(LABELS) + li] += 1
                walk(c, LABELS[li], ox + b.get("left", 0.0), top)
            else:
                walk(c, parent_lab, ox, oy)

    body = root.find("body")
    walk(body if body is not None else root, "ROOT", 0.0, 0.0)
    v = np.concatenate([band / max(band.sum(), 1), big / max(big.sum(), 1)])
    nrm = np.linalg.norm(v)
    return v / nrm if nrm else v
