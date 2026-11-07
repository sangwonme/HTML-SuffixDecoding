"""S2': level-wise canonicalization + perceptual redundancy eta_L (L1, L2, L3(delta=2), L4(eps=0.01,0.05)).

Nested greedy (ddmin) deletion/canonicalization, each step verified by re-render against R(y):
  L1  comments, whitespace-only text between tags, collapse multi-space, optional end tags / html-head-body start tags,
      attribute unquoting, default attrs (type=text on input, type=text/css on style)          check: DOM-equal
  L2  CSS minify, delete whole rules, delete declarations, delete attributes, class alpha-rename (short names)
                                                                                                 check: computed-style equal
  L3  same candidates                                                                            check: layout equal (2px) + paint props equal
  L4  same candidates                                                                            check: D_w <= eps
eta_L = 1 - ntok(y_L)/ntok(y_html).  y_html = extracted HTML (no ``` fence).

  .venv/Scripts/python.exe -I scripts/study_s2_eta.py [cond ...]
"""
from __future__ import annotations

import sys, pathlib, json, re, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from study_common import OUT, guard_setup, load_runs, ntok, check_rss
from study_equiv import Checker
from genui.metrics import extract_html

OPT_END = r"</(?:li|p|td|th|tr|option|html|head|body|thead|tbody|dt|dd)\s*>"


def apply(base, edits):
    out, pos = [], 0
    for a, b, rep in sorted(edits):
        if a < pos:
            continue
        out.append(base[pos:a]); out.append(rep); pos = b
    out.append(base[pos:])
    return "".join(out)


def ddmin(base, edits, ok, budget):
    acc, queue = [], [edits] if edits else []
    while queue and budget[0] > 0:
        g = queue.pop(0)
        budget[0] -= 1
        if ok(apply(base, acc + g)):
            acc += g
        elif len(g) > 1:
            h = len(g) // 2
            queue[:0] = [g[:h], g[h:]]
    return apply(base, acc)


def start_tags(t):
    return list(re.finditer(r"<[a-zA-Z][^<>]*>", t))


def style_blocks(t):
    return [(m.start(1), m.end(1)) for m in re.finditer(r"<style[^>]*>(.*?)</style\s*>", t, re.S | re.I)]


# ---------------- candidate generators (each returns non-overlapping edits on text t)
def e_l1(t):
    E = []
    E += [(m.start(), m.end(), "") for m in re.finditer(r"<!--.*?-->", t, re.S)]
    sb = style_blocks(t)
    inside = lambda i: any(a <= i < b for a, b in sb)
    E += [(m.start(1), m.end(1), "") for m in re.finditer(r">(\s+)<", t) if not inside(m.start())]
    E += [(m.start(), m.end(), "") for m in re.finditer(OPT_END, t, re.I)]
    E += [(m.start(), m.end(), "") for m in re.finditer(r"<(?:html|head|body)\s*>", t, re.I)]
    for m in start_tags(t):
        for a in re.finditer(r"""=\s*"([^\s"'=<>`]+)"|=\s*'([^\s"'=<>`]+)'""", m.group()):
            v = a.group(1) or a.group(2)
            E.append((m.start() + a.start(), m.start() + a.end(), "=" + v))
        for a in re.finditer(r"""\s+type\s*=\s*["']?(?:text|text/css)["']?""", m.group(), re.I):
            if re.match(r"<(input|style)\b", m.group(), re.I):
                E.append((m.start() + a.start(), m.start() + a.end(), ""))
    # collapse multi-whitespace inside text nodes
    E += [(m.start(), m.end(), " ") for m in re.finditer(r"(?<=[^>\s])[ \t\r\n]{2,}(?=[^<\s])", t) if not inside(m.start())]
    E.sort()
    out, pos = [], -1
    for e in E:  # drop overlaps
        if e[0] >= pos:
            out.append(e); pos = e[1]
    return out


def minify_css(c):
    c = re.sub(r"/\*.*?\*/", "", c, flags=re.S)
    c = re.sub(r"\s+", " ", c)
    c = re.sub(r"\s*([{}:;,>])\s*", r"\1", c)
    return c.replace(";}", "}").strip()


def e_minify(t):
    return [(a, b, minify_css(t[a:b])) for a, b in style_blocks(t) if minify_css(t[a:b]) != t[a:b]]


def e_rules(t):
    E = []
    for a, b in style_blocks(t):
        E += [(a + m.start(), a + m.end(), "") for m in re.finditer(r"[^{}]+\{[^{}]*\}", t[a:b])]
    return E


def e_decls(t):
    E = []
    for a, b in style_blocks(t):
        for m in re.finditer(r"\{([^{}]*)\}", t[a:b]):
            base = a + m.start(1)
            for d in re.finditer(r"[^;]+;?", m.group(1)):
                if d.group().strip():
                    E.append((base + d.start(), base + d.end(), ""))
    for m in start_tags(t):
        sm = re.search(r"""style\s*=\s*(["'])(.*?)\1""", m.group(), re.S)
        if sm:
            base = m.start() + sm.start(2)
            for d in re.finditer(r"[^;]+;?", sm.group(2)):
                if d.group().strip():
                    E.append((base + d.start(), base + d.end(), ""))
    return E


def e_attrs(t):
    E = []
    for m in start_tags(t):
        if re.match(r"<(meta|html)\b", m.group(), re.I):
            continue
        for a in re.finditer(r"""\s+([\w:-]+)(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?""", m.group()):
            if a.start() == 0:
                continue
            E.append((m.start() + a.start(), m.start() + a.end(), ""))
    return E


def alpha_rename(t):
    names = []
    for m in start_tags(t):
        for c in re.finditer(r"""class\s*=\s*["']?([^"'>]*)""", m.group()):
            for n in c.group(1).split():
                if n not in names:
                    names.append(n)
    if not names:
        return t
    letters = "abcdefghijklmnopqrstuvwxyz"
    short = {n: (letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]) for i, n in enumerate(names)}
    def ren_class(m):
        return m.group(1) + " ".join(short.get(n, n) for n in m.group(2).split())
    t2 = re.sub(r"""(class\s*=\s*["']?)([^"'>]*)""", ren_class, t)
    def ren_css(block):
        return re.sub(r"\.(-?[_a-zA-Z][\w-]*)", lambda m: "." + short.get(m.group(1), m.group(1)), block)
    out, pos = [], 0
    for a, b in style_blocks(t2):
        # only selector parts (outside {...})
        seg = t2[a:b]
        seg = re.sub(r"([^{}]+)(\{[^{}]*\})", lambda m: ren_css(m.group(1)) + m.group(2), seg)
        out += [t2[pos:a], seg]; pos = b
    out.append(t2[pos:])
    return "".join(out)


def run_sample(chk, html, budget_per_pass=160):
    ref = chk.measure(html)
    n0 = ntok(html)
    res = {"n_tok": n0}

    def ok_fn(key):
        def ok(txt):
            try:
                return bool(chk.levels(ref, txt)[key])
            except Exception:
                return False
        return ok

    passes_style = [("minify", e_minify), ("rules", e_rules), ("decls", e_decls), ("attrs", e_attrs)]
    cur = ddmin(html, e_l1(html), ok_fn("L1"), [400])
    res["L1"] = {"tok": ntok(cur)}
    # L2
    ok2 = ok_fn("L2")
    for name, gen in passes_style:
        cur = ddmin(cur, gen(cur), ok2, [budget_per_pass])
    ren = alpha_rename(cur)
    if ren != cur and ok2(ren):
        res["alpha_rename_tok_saved"] = ntok(cur) - ntok(ren)
        cur = ren
    res["L2"] = {"tok": ntok(cur)}
    for key in ("L3_d2", "L4_e0.01", "L4_e0.05"):
        ok = ok_fn(key)
        for name, gen in passes_style[1:]:
            cur = ddmin(cur, gen(cur), ok, [budget_per_pass])
        res[key] = {"tok": ntok(cur)}
        if key == "L4_e0.05":
            res["final_html"] = cur
    for k in ("L1", "L2", "L3_d2", "L4_e0.01", "L4_e0.05"):
        res[k]["eta"] = 1 - res[k]["tok"] / n0
    lv = chk.levels(ref, cur)
    res["final_D4"] = lv["D4"]
    return res


def main():
    guard_setup()
    conds = sys.argv[1:] or ["E2B_nf4_rich_fs", "12B_nf4_rich_zs", "E2B_nf4_rich_zs", "12B_nf4_rich_fs",
                             "E4B_nf4_rich_fs", "E4B_nf4_rich_zs", "E2B_nf4_short_zs"]
    pilot, runs = load_runs()
    outp = OUT / "s2_eta.json"
    allres = json.loads(outp.read_text(encoding="utf-8")) if outp.exists() else {}
    chk = Checker()
    try:
        for c in conds:
            allres.setdefault(c, {})
            for r in runs[c]:
                if r["id"] in allres[c]:
                    continue
                html = extract_html(r["response"])
                t0 = time.time()
                res = run_sample(chk, html)
                res["sec"] = time.time() - t0
                allres[c][r["id"]] = res
                print(c, r["id"], {k: round(res[k]["eta"], 3) for k in ("L1", "L2", "L3_d2", "L4_e0.01", "L4_e0.05")},
                      f"{res['sec']:.0f}s rss={check_rss():.2f}GB", flush=True)
                outp.write_text(json.dumps(allres, indent=1), encoding="utf-8")
    finally:
        chk.r.close()


if __name__ == "__main__":
    main()
