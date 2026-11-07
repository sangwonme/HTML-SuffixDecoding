"""S2' part 2: offline PLD simulation (draft = n-gram lookup in the 3 retrieved example HTMLs, target = actual
output y) and level-wise classification of every draft/target mismatch.  + S4' sibling isomorphism (text-only).

For each mismatch: take the DOM unit enclosing the mismatch on both sides (innermost element, or the CSS rule if
the position is inside <style>), substitute the draft unit for y's unit, and test y' against y at L0..L4.
Also tests textual class alpha-equivalence of the two units.

  .venv/Scripts/python.exe -I scripts/study_s2_pld.py
Output: runs/study/s2_pld.json, runs/study/s4_siblings.json
"""
from __future__ import annotations

import sys, pathlib, json, re, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from collections import Counter

from study_common import CONDS, OUT, VOID, check_rss, encode, guard_setup, load_runs, retrieved_examples
from study_equiv import Checker
from genui.metrics import extract_html
from study_canon import kappa

NMAX, NMIN, K = 3, 2, 8
MAX_EVENTS = 200
LEVELS = ["L0", "L1", "L2", "L3_d0", "L3_d2", "L3_d8", "L4_e0.01", "L4_e0.05"]


def element_spans(t):
    """[(start, end, name, start_tag_end)] for elements outside style/script/comments."""
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
                    end = m.end() if s[2] == name else m.start()
                    out.append((s[0], end, s[2], s[1]))
                    if s[2] == name:
                        break
            continue
        if name in VOID or m.group().endswith("/>"):
            out.append((m.start(), m.end(), name, m.end()))
        else:
            stack.append((m.start(), m.end(), name))
    for s in stack:
        out.append((s[0], len(t), s[2], s[1]))
    # also style elements as units
    for m in re.finditer(r"<style[^>]*>.*?</style\s*>", t, re.S | re.I):
        out.append((m.start(), m.end(), "style", m.start() + m.group().find(">") + 1))
    return out


def unit_at(t, pos, spans):
    for m in re.finditer(r"<style[^>]*>(.*?)</style\s*>", t, re.S | re.I):
        if m.start(1) <= pos < m.end(1):
            for r in re.finditer(r"[^{}]+\{[^{}]*\}", m.group(1)):
                if m.start(1) + r.start() <= pos < m.start(1) + r.end():
                    a = m.start(1) + r.start()
                    # strip leading whitespace from rule span
                    while a < pos and t[a].isspace():
                        a += 1
                    return ("css_rule", a, m.start(1) + r.end())
            return ("style", m.start(), m.end())
    best = None
    for a, b, name, _ in spans:
        if a <= pos < b and (best is None or b - a < best[2] - best[1]):
            best = ("elem:" + name, a, b)
    return best or ("doc", 0, len(t))


def alpha_canon(s):
    names = {}
    def nm(n):
        return names.setdefault(n, f"c{len(names)}")
    s = re.sub(r"""(class\s*=\s*["']?)([^"'>]*)""", lambda m: m.group(1) + " ".join(nm(x) for x in m.group(2).split()), s)
    s = re.sub(r"\.(-?[_a-zA-Z][\w-]*)(?=[^{}]*\{)", lambda m: "." + nm(m.group(1)), s)
    return re.sub(r"\s+", " ", s).strip()


def pld(ids, srcs):
    """Return steps [(t, a, k, src_idx, j)] ; j = example token index of draft start (first occurrence)."""
    index = {}
    for si, sids in enumerate(srcs):
        for n in range(NMIN, NMAX + 1):
            for j in range(len(sids) - n):
                index.setdefault(tuple(sids[j:j + n]), (si, j + n))
    steps, t, T = [], 1, len(ids)
    while t < T:
        found = None
        for n in range(NMAX, NMIN - 1, -1):
            if t >= n and tuple(ids[t - n:t]) in index:
                found = index[tuple(ids[t - n:t])]
                break
        if not found:
            t += 1
            continue
        si, j = found
        draft = srcs[si][j:j + K]
        a = 0
        while a < len(draft) and t + a < T and draft[a] == ids[t + a]:
            a += 1
        steps.append((t, a, len(draft), si, j))
        t += a + 1
    return steps


def siblings(t):
    spans = sorted(element_spans(t))
    # children map
    kids = {}
    for i, (a, b, name, se) in enumerate(spans):
        par = None
        for a2, b2, n2, _ in spans:
            if a2 < a and b <= b2 and (par is None or b2 - a2 < par[1] - par[0]):
                par = (a2, b2)
        kids.setdefault(par, []).append((a, b))
    def sig(a, b, with_class):
        seg = t[a:b]
        if with_class:
            return " ".join(m.group(0) for m in re.finditer(r"</?[a-zA-Z][\w-]*|class\s*=\s*[\"']?[^\"'>\s]*", seg))
        return " ".join(re.findall(r"</?[a-zA-Z][\w-]*", seg))
    res = Counter()
    cov = {"tag": [], "cls": []}
    for par, ch in kids.items():
        if par is None or len(ch) < 2:
            continue
        for wc in (False, True):
            seen = set()
            for a, b in ch:
                s = sig(a, b, wc)
                key = "cls" if wc else "tag"
                res[f"n_{key}"] += 1
                res[f"chars_{key}"] += b - a
                if s in seen:
                    res[f"iso_{key}"] += 1
                    cov[key].append((a, b))
                seen.add(s)
    for key, iv in cov.items():  # union of isomorphic-sibling spans (no double counting of nesting)
        covered, end = 0, -1
        for a, b in sorted(iv):
            if b > end:
                covered += b - max(a, end); end = b
        res[f"iso_chars_{key}"] = covered
    res["doc_chars"] = len(t)
    return res


def main():
    guard_setup()
    pilot, runs = load_runs()
    chk = Checker()
    allres, sib = {}, {}
    try:
        for c in CONDS:
            ev_all, steps_all, sibc = [], [], Counter()
            t0 = time.time()
            for r in runs[c]:
                y = extract_html(r["response"])
                sibc.update(siblings(y))
                exs = [e["gt_html_flow"] for e in retrieved_examples(pilot[r["id"]])]
                ids, offs = encode(y)
                src = [encode(e) for e in exs]
                steps = pld(ids, [s[0] for s in src])
                ref = chk.measure(y)
                ky1, ky2 = kappa(y, 1), kappa(y, 2)
                yspans = element_spans(y)
                espans = [element_spans(e) for e in exs]
                mism = [s for s in steps if s[1] < s[2] and s[0] + s[1] < len(ids)]
                cache = {}
                for (t, a, k, si, j) in mism[:MAX_EVENTS]:
                    cy = offs[t + a][0]
                    cd = src[si][1][j + a][0]
                    uy = unit_at(y, cy, yspans)
                    ud = unit_at(exs[si], cd, espans[si])
                    dtxt = exs[si][ud[1]:ud[2]]
                    ytxt = y[uy[1]:uy[2]]
                    y2 = y[:uy[1]] + dtxt + y[uy[2]:]
                    if y2 not in cache:
                        try:
                            cache[y2] = chk.levels(ref, y2)
                        except Exception:
                            cache[y2] = {L: False for L in LEVELS}
                    lv = cache[y2]
                    ev = {L: bool(lv[L]) for L in LEVELS}
                    ev["L4t_e0.01"] = bool(lv.get("L4t_e0.01", False)); ev["L4t_e0.05"] = bool(lv.get("L4t_e0.05", False))
                    ev["text_eq"] = bool(lv.get("text_eq", False)); ev["D4"] = float(lv.get("D4", 1.0))
                    ev["kappa1_eq"] = kappa(y2, 1) == ky1
                    ev["kappa2_eq"] = kappa(y2, 2) == ky2
                    ev["kind_y"], ev["kind_d"] = uy[0], ud[0]
                    ev["same_kind"] = uy[0] == ud[0]
                    ev["alpha_eq"] = alpha_canon(ytxt) == alpha_canon(dtxt)
                    ev["ws_eq"] = re.sub(r"\s+", " ", ytxt).strip() == re.sub(r"\s+", " ", dtxt).strip()
                    ev["unit_chars"] = len(ytxt)
                    ev["a"], ev["k"] = a, k
                    ev["id"] = r["id"]
                    ev_all.append(ev)
                steps_all += [(s[1], s[2]) for s in steps]
                check_rss()
            n = len(ev_all)
            cum = {}
            acc = False
            for L in LEVELS:
                cum[L] = sum(any(e[x] for x in LEVELS[:LEVELS.index(L) + 1]) for e in ev_all) / max(1, n)
            # alpha_L: drafted tokens accepted; a mismatch equivalent at <=L counts as full acceptance of the step
            drafted = sum(k for _, k in steps_all)
            base_acc = sum(a for a, _ in steps_all)
            alpha = {"L0_exact": base_acc / max(1, drafted)}
            for L in LEVELS[1:]:
                extra = sum(e["k"] - e["a"] for e in ev_all if any(e[x] for x in LEVELS[:LEVELS.index(L) + 1]))
                alpha[L] = (base_acc + extra) / max(1, drafted)
            kap = {"kappa1_eq": sum(e["kappa1_eq"] for e in ev_all) / max(1, n),
                   "kappa2_eq": sum(e["kappa1_eq"] or e["kappa2_eq"] for e in ev_all) / max(1, n)}
            for e5 in ("0.01", "0.05"):
                cum[f"L4t_e{e5}"] = sum(any(e[x] for x in LEVELS[:6]) or e[f"L4t_e{e5}"] for e in ev_all) / max(1, n)
                extra = sum(e["k"] - e["a"] for e in ev_all if any(e[x] for x in LEVELS[:6]) or e[f"L4t_e{e5}"])
                alpha[f"L4t_e{e5}"] = (base_acc + extra) / max(1, drafted)
            kinds = Counter(e["kind_y"].split(":")[0] if not e["kind_y"].startswith("elem") else "element" for e in ev_all)
            allres[c] = {"n_steps": len(steps_all), "n_mismatch_checked": n,
                         "mean_accept_len_exact": base_acc / max(1, len(steps_all)),
                         "mismatch_equiv_cum": cum, "mismatch_kappa_eq": kap, "alpha_L": alpha,
                         "alpha_rename_eq_share": sum(e["alpha_eq"] for e in ev_all) / max(1, n),
                         "alpha_rename_only_share": sum(e["alpha_eq"] and not e["ws_eq"] for e in ev_all) / max(1, n),
                         "same_kind_share": sum(e["same_kind"] for e in ev_all) / max(1, n),
                         "mismatch_unit_kinds": {k: v / max(1, n) for k, v in kinds.items()},
                         "equiv_by_kind": {kk: {L: sum(any(e[x] for x in LEVELS[:LEVELS.index(L) + 1]) for e in ev_all
                                                     if (e["kind_y"].startswith("elem") and kk == "element") or e["kind_y"] == kk)
                                                / max(1, kinds[kk]) for L in LEVELS} for kk in kinds},
                         "events": ev_all}
            sib[c] = {"tag_iso_share": sibc["iso_tag"] / max(1, sibc["n_tag"]),
                      "class_iso_share": sibc["iso_cls"] / max(1, sibc["n_cls"]),
                      "tag_iso_char_share_of_doc": sibc["iso_chars_tag"] / max(1, sibc["doc_chars"]),
                      "class_iso_char_share_of_doc": sibc["iso_chars_cls"] / max(1, sibc["doc_chars"]),
                      "raw": dict(sibc)}
            print(c, f"steps={len(steps_all)} mism={n} alpha={ {k: round(v, 3) for k, v in alpha.items()} }",
                  f"cum={ {k: round(v, 3) for k, v in cum.items()} } kap={ {k: round(v, 3) for k, v in kap.items()} } alphaRen={allres[c]['alpha_rename_only_share']:.3f}",
                  f"sib={sib[c]['tag_iso_share']:.3f} {time.time() - t0:.0f}s", flush=True)
            (OUT / "s2_pld.json").write_text(json.dumps(allres, indent=1), encoding="utf-8")
            (OUT / "s4_siblings.json").write_text(json.dumps(sib, indent=1), encoding="utf-8")
    finally:
        chk.r.close()


if __name__ == "__main__":
    main()
