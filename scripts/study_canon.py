"""kappa: canonical form (study-local implementation; src/genui/canon.py did not exist at study time).

kappa_L1(y): lenient parse (lxml.html) -> drop comments and whitespace-only text nodes -> sort attributes ->
             compact serialization (explicit end tags for non-void elements, attrs as name="value").
kappa(y)   : kappa_L1 + drop data-* attributes + rename classes to c0,c1,... in first-appearance order
             (class attributes and .selectors in <style>) + CSS whitespace minify inside <style>.
"""
from __future__ import annotations

import re
from html import escape

import lxml.html
from lxml import etree

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
RAW = {"style", "script"}


def _min_css(c):
    c = re.sub(r"/\*.*?\*/", "", c, flags=re.S)
    c = re.sub(r"\s+", " ", c)
    c = re.sub(r"\s*([{}:;,>])\s*", r"\1", c)
    return c.replace(";}", "}").strip()


def kappa(html: str, level: int = 2) -> str:
    if not html or not html.strip():
        return ""
    try:
        doc = lxml.html.document_fromstring(html)
    except (etree.ParserError, ValueError):
        return re.sub(r"\s+", " ", html).strip()
    names: dict[str, str] = {}

    def cname(n):
        return names.setdefault(n, f"c{len(names)}")

    if level >= 2:  # first-appearance order over class attributes in document order
        for el in doc.iter():
            if isinstance(el.tag, str) and el.get("class"):
                for n in el.get("class").split():
                    cname(n)
    out = []

    def text(s, parent):
        if s is None:
            return
        if parent in RAW:
            if level >= 2 and parent == "style":
                s = _min_css(s)
                s = re.sub(r"\.(-?[_a-zA-Z][\w-]*)(?=[^{}]*\{)", lambda m: "." + cname(m.group(1)), s)
            out.append(s)
        elif s.strip():
            out.append(escape(re.sub(r"\s+", " ", s), quote=False))

    def ser(el):
        if not isinstance(el.tag, str):  # comment / PI
            text(el.tail, el.getparent().tag if el.getparent() is not None else "")
            return
        tag = el.tag.lower()
        attrs = []
        for k, v in el.attrib.items():
            k = k.lower()
            if level >= 2 and k.startswith("data-"):
                continue
            v = re.sub(r"\s+", " ", v).strip()
            if level >= 2 and k == "class":
                v = " ".join(cname(n) for n in v.split())
            attrs.append((k, v))
        attrs.sort()
        out.append("<" + tag + "".join(f' {k}="{escape(v)}"' for k, v in attrs) + ">")
        text(el.text, tag)
        for ch in el:
            ser(ch)
        if tag not in VOID:
            out.append(f"</{tag}>")
        par = el.getparent()
        text(el.tail, par.tag.lower() if par is not None and isinstance(par.tag, str) else "")

    ser(doc)
    return "<!DOCTYPE html>" + "".join(out)
