"""Checklist scorer for query_mid (main metric for the mid setting).

A checklist is a list of items, each {"type": <componentLabel>, "keyword"?: str, "min_count"?: int}.
An item is satisfied when the rendered output contains >= min_count (default 1) elements whose
normalized category is accepted for `type` and (if keyword is given) whose text contains the keyword.

- Categories come from metrics.NORMALIZE_JS (same normalizer as element F1). We reuse it by import and
  only patch its push line so each element also returns its text.
- Element text = innerText + placeholder/aria-label/alt/value/title of the element and its descendants.
  Form controls also see their <label> text and a short parent's innerText (<= 60 chars).
- type "Text" accepts any element: the keyword only has to appear in the page text, which is
  metrics.ATTR_TEXT_JS (innerText + placeholder/aria-label/alt/value/title fallback).
- Keyword match: NFKC, lowercase, non-alphanumerics -> space; substring match, then a fallback with
  spaces removed ("log in" ~ "login").
- Lenient score (diagnostic): keyword anywhere in the page, or the type present, ignoring the pairing.
"""
from __future__ import annotations

import re
import unicodedata

from genui.metrics import ATTR_TEXT_JS, NORMALIZE_JS, extract_html

_TXT_FN = r"""
  const __attrs = ['placeholder','aria-label','alt','value','title'];
  const __txt = el => {
    const parts = [el.innerText || el.textContent || ''];
    for (const e of [el, ...el.querySelectorAll('*')]) for (const a of __attrs) { const v = e.getAttribute(a); if (v) parts.push(v); }
    if (['INPUT','SELECT','TEXTAREA'].includes(el.tagName)) {
      if (el.labels) for (const l of el.labels) parts.push(l.innerText || '');
      const p = el.parentElement;
      if (p) { const pt = (p.innerText || '').trim(); if (pt.length <= 60) parts.push(pt); }
    }
    return parts.join(' \n ');
  };"""
_ANCHOR_DECL = "const out = [];"
_ANCHOR_PUSH = "if (c) out.push(c);"
assert NORMALIZE_JS.count(_ANCHOR_DECL) == 1 and NORMALIZE_JS.count(_ANCHOR_PUSH) == 1, "metrics.NORMALIZE_JS changed"
ELEMENTS_JS = (NORMALIZE_JS.replace(_ANCHOR_DECL, _ANCHOR_DECL + _TXT_FN)
               .replace(_ANCHOR_PUSH, "if (c) out.push([c, __txt(el)]);"))

# type -> accepted normalized categories (None = any element / page text)
ACCEPT = {
    "Text": None,
    "Text Button": {"Text Button"},
    "Input": {"Input"},
    "Toolbar": {"Toolbar"},
    "Checkbox": {"Checkbox"},
    "Radio Button": {"Radio Button"},
    "On/Off Switch": {"On/Off Switch", "Checkbox"},  # a plain checkbox styled as a toggle is accepted
    "Bottom Navigation": {"Bottom Navigation"},
    "Multi-Tab": {"Multi-Tab"},
    "List Item": {"List Item"},
    "Image": {"Image"},
    "Modal": {"Modal"},
    "Drawer": {"Drawer"},
    "Card": {"Card"},
    "Slider": {"Slider"},
    "Date Picker": {"Date Picker"},
    "Number Stepper": {"Number Stepper"},
    "Icon": {"Icon"},
}


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zÀ-￿]+", " ", s)).strip()


def kw_in(kw: str, hay_norm: str) -> bool:
    k = norm_text(kw)
    if not k:
        return True
    return k in hay_norm or k.replace(" ", "") in hay_norm.replace(" ", "")


def analyze(page, html: str) -> dict:
    page.set_content(html, wait_until="load", timeout=10000)
    els = page.evaluate(ELEMENTS_JS)
    text = page.evaluate(ATTR_TEXT_JS)
    return {"els": [(c, norm_text(t)) for c, t in els], "page": norm_text(text)}


def item_ok(item: dict, a: dict) -> tuple[bool, bool]:
    """Return (strict, lenient) satisfaction for one item on an analysis dict."""
    typ, kw, need = item["type"], item.get("keyword"), int(item.get("min_count", 1))
    acc = ACCEPT.get(typ, {typ})
    kw_page = kw_in(kw, a["page"]) if kw else True
    if acc is None:  # Text: page text only
        return kw_page, kw_page
    typed = [t for c, t in a["els"] if c in acc]
    n = sum(1 for t in typed if (kw_in(kw, t) if kw else True))
    strict = n >= need
    lenient = strict or (len(typed) >= need and kw_page) or (kw is not None and kw_page)
    return strict, lenient


def score_analysis(checklist: list[dict], a: dict) -> dict:
    res = [item_ok(it, a) for it in checklist]
    k = len(res) or 1
    return {"check_score": sum(s for s, _ in res) / k, "check_all": all(s for s, _ in res),
            "check_lenient": sum(l for _, l in res) / k, "item_ok": [s for s, _ in res]}


class ChecklistScorer:
    """One headless page, network blocked (same setup as metrics.Scorer)."""

    def __init__(self, browser):
        self.ctx = browser.new_context(viewport={"width": 360, "height": 640}, java_script_enabled=True)
        self.ctx.route("**/*", lambda route: route.abort())
        self.page = self.ctx.new_page()

    def analyze(self, html: str) -> dict:
        return analyze(self.page, html)

    def score(self, output_text: str, checklist: list[dict]) -> dict:
        html = extract_html(output_text)
        empty = {"check_score": 0.0, "check_all": False, "check_lenient": 0.0, "item_ok": [False] * len(checklist)}
        if html is None:
            return {**empty, "extracted": False}
        try:
            a = self.analyze(html)
        except Exception as e:  # noqa: BLE001
            return {**empty, "extracted": True, "render_error": str(e)[:200]}
        return {**score_analysis(checklist, a), "extracted": True}
