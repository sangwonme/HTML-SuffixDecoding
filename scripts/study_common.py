"""Shared helpers for the preliminary study (S1-S4). CPU only.

- resource guard: BelowNormal priority, <=4 cores, 6GB RSS guard (incl. Chromium children)
- tokenizer (Gemma 4 E2B tokenizer.json only, no weights)
- char-level HTML/CSS lexer -> lexical context per char + grammar-forced chars
- renderer: one headless Chromium, 360x640, network blocked; SSIM / pixel diff
"""
from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[k] = "4"

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "ms-playwright"))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs" / "study"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CONDS = ["E2B_nf4_short_zs", "E2B_nf4_rich_zs", "E4B_nf4_rich_zs", "12B_nf4_rich_zs",
         "E2B_nf4_rich_fs", "E4B_nf4_rich_fs", "12B_nf4_rich_fs"]
RSS_LIMIT_GB = 6.0


def guard_setup():
    p = psutil.Process()
    try:
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        p.cpu_affinity(list(range(min(4, psutil.cpu_count()))))
    except Exception:
        pass


def tame_children():
    for c in psutil.Process().children(recursive=True):
        try:
            c.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            c.cpu_affinity(list(range(min(int(os.environ.get("SIM_THREADS", "4")), psutil.cpu_count()))))
        except Exception:
            pass


def check_rss():
    p = psutil.Process()
    tot = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            tot += c.memory_info().rss
        except Exception:
            pass
    gb = tot / 2**30
    if gb > RSS_LIMIT_GB:
        raise SystemExit(f"RSS guard hit: {gb:.2f} GB")
    return gb


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_runs():
    pilot = {r["id"]: r for r in load_jsonl(ROOT / "data" / "processed" / "pilot.jsonl")}
    runs = {}
    for c in CONDS:
        rows = [r for r in load_jsonl(ROOT / "runs" / "gemma" / f"{c}_pilot.jsonl") if "id" in r]
        runs[c] = rows
    return pilot, runs


_TK = None


def tokenizer():
    global _TK
    if _TK is None:
        from tokenizers import Tokenizer
        _TK = Tokenizer.from_file(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized" / "tokenizer.json"))
    return _TK


def encode(text):
    e = tokenizer().encode(text, add_special_tokens=False)
    return e.ids, e.offsets


def ntok(text):
    return len(tokenizer().encode(text, add_special_tokens=False).ids)


_RET = None


def retrieved_examples(row):
    """Same retrieval as scripts/bench_gemma.py build_messages (rich_fs)."""
    global _RET
    if _RET is None:
        from genui.retrieval import Retriever
        _RET = Retriever(load_jsonl(ROOT / "data" / "processed" / "train_pool.jsonl"))
    return _RET.topk(row["query_rich"], 3, max_html_chars=4000)


# ---------------------------------------------------------------- lexer
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
NULL_ATTR = re.compile(r"^(data-[\w-]*|id|aria-[\w-]*|title|role)$", re.I)
TOL_ATTR = {"width", "height"}
NUM_RE = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:px|em|rem|%|vh|vw|pt|s|ms|deg|fr)?")
COLOR_RE = re.compile(r"#[0-9a-fA-F]{3,8}\b|(?:rgba?|hsla?)\([^)]*\)")
NAMED_COLORS = {"white", "black", "red", "green", "blue", "gray", "grey", "transparent", "orange", "yellow",
                "purple", "pink", "navy", "teal", "silver", "maroon", "olive", "lime", "aqua", "fuchsia",
                "lightgray", "lightgrey", "darkgray", "darkgrey", "whitesmoke", "gainsboro", "dimgray"}

# context labels
CTX = ["fence", "doctype", "tag", "attr_name", "attr_val", "null_attr", "tol_attr", "css_selector",
       "css_prop", "css_tol", "css_val", "text", "head_title", "comment", "whitespace"]
S_NULL = {"null_attr", "head_title", "comment", "whitespace"}
S_TOL = {"css_tol", "tol_attr"}
CTX_GROUP = {"fence": "markup", "doctype": "markup", "tag": "markup", "attr_name": "markup",
             "attr_val": "attr_value", "null_attr": "nonrender_attr", "tol_attr": "css_numeric_color",
             "css_selector": "css_other", "css_prop": "css_other", "css_val": "css_other",
             "css_tol": "css_numeric_color", "text": "text", "head_title": "text_nonrender",
             "comment": "comment", "whitespace": "whitespace"}


def _mark_css(s, base, ctx, value_spans):
    """Lex CSS text s (starting at absolute offset base). Fill ctx; collect value spans."""
    state, depth, i, n = "sel", 0, 0, len(s)
    vstart = None
    while i < n:
        ch = s[i]
        if s.startswith("/*", i):
            j = s.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                ctx[base + k] = "comment"
            i = j
            continue
        if state == "sel":
            ctx[base + i] = "css_selector"
            if ch == "{":
                # @media etc. keep selector state at deeper level
                head = s[:i].rsplit("}", 1)[-1].strip()
                state = "sel" if head.startswith("@") and not head.startswith("@font-face") else "prop"
                depth += 1
            elif ch == "}":
                depth = max(0, depth - 1)
        elif state == "prop":
            ctx[base + i] = "css_prop"
            if ch == ":":
                state, vstart = "val", i + 1
            elif ch == "}":
                state, depth = "sel", max(0, depth - 1)
        elif state == "val":
            ctx[base + i] = "css_val"
            if ch in ";}":
                ctx[base + i] = "css_prop"
                value_spans.append((base + vstart, base + i))
                if ch == "}":
                    state, depth = "sel", max(0, depth - 1)
                else:
                    state = "prop"
        i += 1
    if state == "val" and vstart is not None:
        value_spans.append((base + vstart, base + n))


def _attr_css(s, base, ctx, value_spans):
    """style="..." attribute value: declarations only."""
    state, vstart = "prop", 0
    for i, ch in enumerate(s):
        if state == "prop":
            ctx[base + i] = "css_prop"
            if ch == ":":
                state, vstart = "val", i + 1
        else:
            ctx[base + i] = "css_val"
            if ch == ";":
                ctx[base + i] = "css_prop"
                value_spans.append((base + vstart, base + i))
                state = "prop"
    if state == "val":
        value_spans.append((base + vstart, base + len(s)))


def lex(text):
    """Return (ctx list per char, forced bool per char)."""
    n = len(text)
    ctx = ["text"] * n
    forced = [False] * n
    value_spans = []
    stack = []
    # fence region
    h0 = text.find("<")
    h1 = text.rfind(">")
    for k in range(0, max(h0, 0)):
        ctx[k] = "fence"
    if h1 >= 0:
        for k in range(h1 + 1, n):
            ctx[k] = "fence"
        if text[:h1 + 1].lower().endswith("</html>"):  # closing fence is forced after </html>
            for k in range(h1 + 1, n):
                if text[k] in "`":
                    forced[k] = True
    i = max(h0, 0)
    end = h1 + 1 if h1 >= 0 else n
    in_title = False
    while i < end:
        if text.startswith("<!--", i):
            j = text.find("-->", i)
            j = n if j < 0 else j + 3
            for k in range(i, j):
                ctx[k] = "comment"
            i = j
            continue
        if text.startswith("<!", i):
            j = text.find(">", i)
            j = n if j < 0 else j + 1
            for k in range(i, j):
                ctx[k] = "doctype"
            i = j
            continue
        if text.startswith("</", i):
            m = re.match(r"</\s*([A-Za-z][\w-]*)\s*>?", text[i:])
            if m:
                name = m.group(1).lower()
                j = i + m.end()
                for k in range(i, j):
                    ctx[k] = "tag"
                # grammar-forced: closing name + '>' when it matches the innermost open element
                if stack and stack[-1] == name:
                    for k in range(i + m.start(1), j):
                        if not text[k].isspace() and text[k] != "/":
                            forced[k] = True
                if name in stack:
                    while stack and stack.pop() != name:
                        pass
                if name == "title":
                    in_title = False
                i = j
                continue
        if text[i] == "<" and i + 1 < n and text[i + 1].isalpha():
            m = re.match(r"<([A-Za-z][\w-]*)", text[i:])
            name = m.group(1).lower()
            j = i + m.end()
            for k in range(i, j):
                ctx[k] = "tag"
            # attributes
            while j < n and text[j] not in ">":
                if text[j].isspace():
                    ctx[j] = "whitespace"
                    j += 1
                    continue
                if text[j] == "/":
                    ctx[j] = "tag"
                    j += 1
                    continue
                am = re.match(r"([^\s=>/]+)(\s*=\s*)?", text[j:])
                if not am or am.end() == 0:
                    ctx[j] = "tag"
                    j += 1
                    continue
                aname = am.group(1).lower()
                null = bool(NULL_ATTR.match(aname))
                actx = "null_attr" if null else "attr_name"
                a0 = j
                for k in range(j, j + am.end()):
                    ctx[k] = actx
                j += am.end()
                if am.group(2) and j < n:
                    q = text[j]
                    if q in "\"'":
                        forced[j] = True  # opening quote after '=' (convention, see POC_study.md)
                        ctx[j] = actx if null else "attr_val"
                        e = text.find(q, j + 1)
                        e = n if e < 0 else e
                        vs, ve = j + 1, e
                        j = min(n, e + 1)
                        if e < n:
                            ctx[e] = actx if null else "attr_val"
                    else:
                        e = j
                        while e < n and not text[e].isspace() and text[e] != ">":
                            e += 1
                        vs, ve = j, e
                        j = e
                    vctx = "null_attr" if null else ("tol_attr" if aname in TOL_ATTR else "attr_val")
                    for k in range(vs, ve):
                        ctx[k] = vctx
                    if aname == "style" and ve > vs:
                        _attr_css(text[vs:ve], vs, ctx, value_spans)
            if j < n:
                ctx[j] = "tag"
            j = min(n, j + 1)
            selfclose = text[max(i, j - 2):j].startswith("/")
            if name not in VOID and not selfclose:
                stack.append(name)
            i = j
            if name == "title":
                in_title = True
            if name in ("style", "script"):
                e = text.lower().find(f"</{name}", i)
                e = end if e < 0 else e
                if name == "style":
                    _mark_css(text[i:e], i, ctx, value_spans)
                i = e
            continue
        ctx[i] = "head_title" if in_title else "text"
        i += 1
    # css numeric/color values
    for vs, ve in value_spans:
        seg = text[vs:ve]
        for m in list(NUM_RE.finditer(seg)) + list(COLOR_RE.finditer(seg)):
            # skip digits that are part of identifiers (e.g. h1)
            if m.start() > 0 and (seg[m.start() - 1].isalpha() or seg[m.start() - 1] == "-" and m.group().startswith("-") is False):
                continue
            for k in range(vs + m.start(), vs + m.end()):
                ctx[k] = "css_tol"
        for m in re.finditer(r"[A-Za-z]+", seg):
            if m.group().lower() in NAMED_COLORS:
                for k in range(vs + m.start(), vs + m.end()):
                    ctx[k] = "css_tol"
    # whitespace chars outside text content count as whitespace context
    for k in range(n):
        if text[k].isspace() and ctx[k] not in ("text", "head_title", "comment"):
            ctx[k] = "whitespace"
    return ctx, forced


def token_context(text, offsets, ctx):
    out = []
    for a, b in offsets:
        seg = text[a:b]
        if b <= a:
            out.append("whitespace")
            continue
        if seg.strip() == "":
            out.append("whitespace")
            continue
        cnt = {}
        for k in range(a, b):
            if not text[k].isspace():
                cnt[ctx[k]] = cnt.get(ctx[k], 0) + 1
        out.append(max(cnt, key=cnt.get))
    return out


# ---------------------------------------------------------------- render
def ssim(a, b):
    from scipy.ndimage import uniform_filter
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    w = 7
    mu_a, mu_b = uniform_filter(a, w), uniform_filter(b, w)
    saa = uniform_filter(a * a, w) - mu_a ** 2
    sbb = uniform_filter(b * b, w) - mu_b ** 2
    sab = uniform_filter(a * b, w) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return float(m.mean())


def gray(img):
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2])


def compare(img_a, img_b):
    """D = 1-SSIM (grayscale) + pixel-diff ratio (any channel |diff|>0). Different sizes -> D=1."""
    if img_a.shape != img_b.shape:
        return {"D": 1.0, "pix": 1.0}
    s = ssim(gray(img_a), gray(img_b))
    pix = float((np.abs(img_a.astype(int) - img_b.astype(int)).max(-1) > 0).mean())
    return {"D": max(0.0, 1.0 - s), "pix": pix}


class Renderer:
    def __init__(self):
        from playwright.sync_api import sync_playwright
        self.pw = sync_playwright().start()
        self.br = self.pw.chromium.launch(headless=True, args=["--disable-gpu", "--renderer-process-limit=1"])
        self.ctx = self.br.new_context(viewport={"width": 360, "height": 640}, java_script_enabled=False)
        self.ctx.route("**/*", lambda route: route.abort())
        self.page = self.ctx.new_page()
        self.n = 0
        tame_children()

    def shot(self, html, full_page=False):
        from PIL import Image
        self.page.set_content(html or "", wait_until="load", timeout=15000)
        png = self.page.screenshot(full_page=full_page, animations="disabled", caret="hide")
        self.n += 1
        if self.n % 50 == 0:
            check_rss()
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))

    def close(self):
        try:
            self.br.close()
        finally:
            self.pw.stop()
