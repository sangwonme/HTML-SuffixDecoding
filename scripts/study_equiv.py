"""Equivalence hierarchy checker L0..L4 (render pipeline: parse -> style -> layout -> paint).

L0 token/text identical
L1 DOM equivalent: parsed DOM tree equal (tags, sorted attrs, whitespace-collapsed non-empty text nodes)
L2 computed-style equivalent: same rendered element sequence with equal (tag, own text, full getComputedStyle);
   attributes/class names/selectors are free (alpha-renaming, dead/overridden declarations, defaults)
L3 layout equivalent (delta px): same element sequence, equal (tag, own text, NON-geometric computed style),
   every border box within delta px.  (paint-relevant props must still match, so L2 => L3 => ~L4)
L4 perceptual: D_w = 0.8*(1-SSIM viewport) + 0.2*(1-SSIM below-fold); viewport only if no below-fold content.
"""
from __future__ import annotations

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np

from study_common import Renderer, compare

SIG_JS = r"""() => {
 const GEO = /^(width|height|min-|max-|margin|padding|top$|left$|right$|bottom$|inset|block-size|inline-size|gap|row-gap|column-gap|flex|grid|justify|align|place|display|position|box-sizing|order|float|clear|line-height|vertical-align|overflow|transform-origin|perspective-origin|x$|y$|r$|rx|ry|cx|cy|d$)/;
 function h(s){let x=2166136261>>>0; for(let i=0;i<s.length;i++){x^=s.charCodeAt(i); x=Math.imul(x,16777619)>>>0;} return x.toString(36)+':'+s.length;}
 function dom(n){ if(n.nodeType===3){const t=n.data.replace(/\s+/g,' '); return t.trim()? JSON.stringify(t):'';}
   if(n.nodeType!==1) return '';
   const a=[...n.attributes].map(x=>x.name+'='+x.value.trim().replace(/\s+/g,' ')).sort().join(' ');
   let s='<'+n.tagName+(a?' '+a:'')+'>'; for(const c of n.childNodes) s+=dom(c); return s+'</>'; }
 const out={dom: h(dom(document.documentElement)), els: []};
 const all=[document.documentElement, ...document.documentElement.querySelectorAll('*')]
   .filter(e=>!e.closest('head') && !['HEAD','STYLE','SCRIPT','META','TITLE','LINK'].includes(e.tagName));
 for(const e of all){ const cs=getComputedStyle(e); const st=[], ng=[];
   for(let i=0;i<cs.length;i++){const p=cs[i]; const v=cs.getPropertyValue(p); st.push(p+':'+v); if(!GEO.test(p)) ng.push(p+':'+v);}
   const r=e.getBoundingClientRect(); let own=''; for(const c of e.childNodes) if(c.nodeType===3) own+=c.data;
   own=own.replace(/\s+/g,' ').trim();
   out.els.push([e.tagName, own, h(st.join('|')), h(ng.join('|')), r.x, r.y, r.width, r.height]); }
 return out; }"""

DELTAS = (0, 2, 8)
EPS = (0.01, 0.05)


class Checker:
    def __init__(self, rend: Renderer | None = None):
        self.r = rend or Renderer()

    def measure(self, html):
        self.r.page.set_content(html or "", wait_until="load", timeout=15000)
        sig = self.r.page.evaluate(SIG_JS)
        from PIL import Image
        import io
        png = self.r.page.screenshot(full_page=True, animations="disabled", caret="hide")
        self.r.n += 1
        img = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        return {"html": html, "sig": sig, "img": img}

    @staticmethod
    def d4(a, b):
        ia, ib = a["img"], b["img"]
        vp = compare(ia[:640], ib[:640])
        if ia.shape[0] <= 640 and ib.shape[0] <= 640:
            return vp["D"], vp["pix"]
        if ia.shape != ib.shape:
            return 0.8 * vp["D"] + 0.2, vp["pix"]
        bf = compare(ia[640:], ib[640:])
        return 0.8 * vp["D"] + 0.2 * bf["D"], vp["pix"]

    def levels(self, ref, html, m=None):
        """Return dict of equivalence booleans of html vs measured ref."""
        m = m or self.measure(html)
        a, b = ref["sig"], m["sig"]
        out = {"L0": html == ref["html"], "L1": a["dom"] == b["dom"]}
        same_seq = len(a["els"]) == len(b["els"]) and all(x[0] == y[0] and x[1] == y[1] for x, y in zip(a["els"], b["els"]))
        out["L2"] = same_seq and all(x[2] == y[2] for x, y in zip(a["els"], b["els"]))
        ng = same_seq and all(x[3] == y[3] for x, y in zip(a["els"], b["els"]))
        maxd = max((max(abs(x[k] - y[k]) for k in range(4, 8)) for x, y in zip(a["els"], b["els"])), default=0.0) if same_seq else 1e9
        for d in DELTAS:
            out[f"L3_d{d}"] = bool(ng and maxd <= d)
        D, pix = self.d4(ref, m)
        for e in EPS:
            out[f"L4_e{e}"] = D <= e
        ta = [x[1] for x in a["els"] if x[1]]
        tb = [x[1] for x in b["els"] if x[1]]
        out["text_eq"] = ta == tb  # visible own-text sequence unchanged
        for e in EPS:
            out[f"L4t_e{e}"] = out[f"L4_e{e}"] and out["text_eq"]
        out["D4"] = D
        out["pix"] = pix
        return out
