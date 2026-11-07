"""Automatic accuracy metrics for generated UI HTML (headless Playwright only).

- extract_html(text): pull the HTML document out of a model response.
- Normalizer (NORMALIZE_JS): maps any DOM to RICO componentLabel categories by tag/role/type/size.
  It ignores data-label, so GT and model output are scored by the same rule.
- element_f1: multiset F1 between normalized categories of output and of the GT flow HTML.
- text_recall: share of key_texts found in rendered innerText or in placeholder/aria-label/alt/value/title.
"""
from __future__ import annotations

import collections
import re

FENCE = re.compile(r"```(?:html|HTML)?\s*\n(.*?)(?:```|$)", re.S)

NORMALIZE_JS = r"""() => {
  const out = [];
  const vh = window.innerHeight || 640;
  const textTags = new Set(['P','H1','H2','H3','H4','H5','H6','SPAN','LABEL','DIV','SMALL','STRONG','EM','B','I',
                            'TD','TH','DT','DD','FIGCAPTION','CAPTION','LEGEND','TIME','BLOCKQUOTE','PRE','CODE','SECTION']);
  const ownText = el => Array.from(el.childNodes).some(n => n.nodeType === 3 && n.textContent.trim().length > 0);
  const inside = (el, sel) => el.parentElement && el.parentElement.closest(sel);
  for (const el of document.body ? document.body.querySelectorAll('*') : []) {
    const t = el.tagName, role = (el.getAttribute('role') || '').toLowerCase();
    const cls = (el.getAttribute('class') || '').toLowerCase();
    const r = el.getBoundingClientRect();
    const small = r.width > 0 && r.height > 0 && r.width <= 48 && r.height <= 48;
    let c = null;
    if (t === 'INPUT') {
      const ty = (el.getAttribute('type') || 'text').toLowerCase();
      if (role === 'switch') c = 'On/Off Switch';
      else if (ty === 'checkbox') c = 'Checkbox';
      else if (ty === 'radio') c = 'Radio Button';
      else if (ty === 'range') c = 'Slider';
      else if (ty === 'number') c = 'Number Stepper';
      else if (['date','time','datetime-local','month','week'].includes(ty)) c = 'Date Picker';
      else if (['button','submit','reset'].includes(ty)) c = 'Text Button';
      else if (ty !== 'hidden') c = 'Input';
    } else if (t === 'TEXTAREA' || t === 'SELECT') c = 'Input';
    else if (t === 'BUTTON' || role === 'button' || (t === 'A' && el.textContent.trim())) {
      c = (el.textContent.trim() ? 'Text Button' : 'Icon');
    } else if (t === 'IMG' || t === 'SVG' || t === 'svg' || t === 'PICTURE' || t === 'CANVAS') {
      if (inside(el, 'svg,picture')) c = null; else c = small ? 'Icon' : 'Image';
    } else if (t === 'I' && (cls.includes('icon') || cls.includes('fa') || !ownText(el))) c = 'Icon';
    else if (cls.split(/\s+/).some(k => k === 'icon' || k.startsWith('icon-') || k.endsWith('-icon') || k === 'material-icons')) c = 'Icon';
    else if (t === 'HEADER' || role === 'banner' || role === 'toolbar') c = 'Toolbar';
    else if (role === 'tablist') c = 'Multi-Tab';
    else if (t === 'NAV' || role === 'navigation') c = (r.top >= 0.75 * vh) ? 'Bottom Navigation' : null;
    else if (t === 'LI' || role === 'listitem') c = 'List Item';
    else if (t === 'ARTICLE' || cls.split(/\s+/).includes('card')) c = 'Card';
    else if (t === 'DIALOG' || role === 'dialog' || role === 'alertdialog') c = 'Modal';
    else if (t === 'ASIDE') c = cls.includes('ad') && !cls.includes('header') ? 'Advertisement' : 'Drawer';
    else if (t === 'VIDEO') c = 'Video';
    else if (t === 'IFRAME') c = 'Web View';
    if (c === null && textTags.has(t) && ownText(el) && !inside(el, 'button,a,[role=button],label > input')) c = 'Text';
    if (c) out.push(c);
  }
  return out;
}"""

ATTR_TEXT_JS = r"""() => {
  const parts = [document.body ? document.body.innerText : ''];
  for (const el of document.querySelectorAll('[placeholder],[aria-label],[alt],[value],[title]')) {
    for (const a of ['placeholder','aria-label','alt','value','title']) {
      const v = el.getAttribute(a); if (v) parts.push(v);
    }
  }
  return parts.join('\n');
}"""


def extract_html(text: str) -> str | None:
    if not text:
        return None
    m = FENCE.search(text)
    if m and "<" in m.group(1):
        return m.group(1).strip()
    i = min([k for k in (text.lower().find("<!doctype"), text.lower().find("<html")) if k >= 0], default=-1)
    if i >= 0:
        return text[i:].strip()
    if re.search(r"<(div|body|main|header|section)\b", text, re.I):
        return text.strip()
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def multiset_f1(pred: list[str], gt: list[str]) -> dict:
    cp, cg = collections.Counter(pred), collections.Counter(gt)
    ov = sum((cp & cg).values())
    p = ov / sum(cp.values()) if cp else 0.0
    r = ov / sum(cg.values()) if cg else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    # type-level (set) F1, ignores counts
    sp, sg = set(cp), set(cg)
    tp = len(sp & sg)
    tp_p = tp / len(sp) if sp else 0.0
    tp_r = tp / len(sg) if sg else 0.0
    tf = 2 * tp_p * tp_r / (tp_p + tp_r) if tp_p + tp_r else 0.0
    return {"elem_p": p, "elem_r": r, "elem_f1": f, "type_f1": tf}


class Scorer:
    """Holds one headless page; network fully blocked."""

    def __init__(self, browser):
        self.ctx = browser.new_context(viewport={"width": 360, "height": 640}, java_script_enabled=True)
        self.ctx.route("**/*", lambda route: route.abort())
        self.page = self.ctx.new_page()
        self.errors: list[str] = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))

    def analyze(self, html: str) -> dict:
        self.errors.clear()
        self.page.set_content(html, wait_until="load", timeout=10000)
        info = self.page.evaluate("() => ({w: document.body ? document.body.scrollWidth : 0,"
                                  " h: document.body ? document.body.scrollHeight : 0,"
                                  " n: document.body ? document.body.querySelectorAll('*').length : 0})")
        cats = self.page.evaluate(NORMALIZE_JS)
        text = self.page.evaluate(ATTR_TEXT_JS)
        return {"info": info, "cats": cats, "text": text, "js_errors": list(self.errors)}

    def score(self, output_text: str, gt: dict, gt_analysis: dict | None = None) -> dict:
        html = extract_html(output_text)
        res = {"extracted": html is not None, "render_ok": False, "elem_p": 0.0, "elem_r": 0.0,
               "elem_f1": 0.0, "type_f1": 0.0, "text_recall": 0.0, "n_pred_elems": 0}
        g = gt_analysis or self.analyze(gt["gt_html_flow"])
        if html is None:
            return res
        try:
            a = self.analyze(html)
        except Exception as e:  # noqa: BLE001
            res["render_error"] = str(e)[:200]
            return res
        ok = a["info"]["w"] > 0 and a["info"]["h"] > 0 and a["info"]["n"] > 0
        res["render_ok"] = bool(ok)
        res["js_errors"] = len(a["js_errors"])
        if not ok:
            return res
        res.update(multiset_f1(a["cats"], g["cats"]))
        res["n_pred_elems"] = len(a["cats"])
        hay = _norm(a["text"])
        kt = gt.get("key_texts") or []
        res["text_recall"] = sum(_norm(t) in hay for t in kt) / len(kt) if kt else 1.0
        return res
