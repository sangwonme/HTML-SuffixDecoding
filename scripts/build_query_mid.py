"""Build query_mid + machine-checkable checklist for every sample (CPU only, template-based).

Usage:
    .venv/Scripts/python.exe -I scripts/build_query_mid.py [--splits dev,test,train_pool] [--limit N]
Outputs:
    data/processed/mid/{pilot,dev,test,train_pool}.jsonl   (id, query_mid, checklist, selection metadata)
    data/processed/mid/build_report.json
Reads (never writes) data/processed/*.jsonl and the raw parquet (play_store_name only; untrusted strings).

Pipeline per sample:
  1. parse GT flow HTML (data-label) -> candidate checklist items (keyword items + type-only items)
  2. render GT once in headless Chromium, keep only candidates the GT itself satisfies under
     metrics.NORMALIZE_JS (checklist.item_ok) -> GT self-score is 1.0 by construction; rejects are counted
  3. pick 2-4 salient items (seeded per id), write purpose + element sentence from varied templates
  4. leak checks: digits / count words, verbatim >=5-word n-gram of any GT text, every item mentioned.
     On failure try the other human captions as purpose; if all fail, keep the row with leak_ok=false.
Resource limits: process priority BELOW_NORMAL, CPU affinity = 2 logical cores (Chromium inherits both),
one headless browser, RSS guard 6 GB (python + chromium).
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import glob
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(ROOT / ".cache" / "ms-playwright"))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import lxml.html  # noqa: E402
import psutil  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from genui.checklist import item_ok, kw_in, norm_text  # noqa: E402
from genui import checklist as CL  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = PROC / "mid"
RAW = ROOT / "data" / "raw" / "RICO-Screen2Words" / "data"
SEED = 20261107
RSS_LIMIT_GB = 6.0
AFFINITY_MASK = 0xC00  # logical cores 10,11

STOP = set("""a an the of in on for and or to is are be with by at from your you my me our we us it its this that
these those as if not no yes but so do does did can will just all any each more most other some such only own same
than too very up down out off over under again then once here there when where why how what which who whom
please now new get got go see use here""".split())
NUM_WORDS = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                       r"fifteen|twenty|thirty|dozen|dozens|hundred|thousand)\b", re.I)
SCREEN_NOUNS = {
    "page", "screen", "display", "window", "dialog", "popup", "pop-up", "pop", "app", "application",
    "list", "menu", "notification", "widget", "interface", "form", "message", "box", "tab", "view",
    "option", "options", "settings", "setting", "profile", "home", "homepage", "login", "log", "sign",
    "search", "map", "calendar", "gallery", "alert", "prompt", "panel", "section", "bar", "keyboard",
}
# words in the query that count as "mentioning" each type
TYPE_WORDS = {
    "Text": None, "Text Button": r"button", "Input": r"field|input|text box|box", "Toolbar": r"bar|header",
    "Checkbox": r"checkbox|check box", "Radio Button": r"radio", "On/Off Switch": r"switch|toggle",
    "Bottom Navigation": r"bottom navigation|bottom nav|navigation bar|bottom bar|tab bar", "Multi-Tab": r"tab",
    "List Item": r"list", "Image": r"image|picture|photo", "Modal": r"dialog|pop-up|popup",
    "Drawer": r"drawer|side menu|side panel", "Card": r"card", "Slider": r"slider",
    "Date Picker": r"date picker|date", "Number Stepper": r"stepper|number", "Icon": r"icon",
}


# ------------------------------------------------------------------ resources
def limit_resources():
    if os.name == "nt":
        k = ctypes.windll.kernel32
        h = k.GetCurrentProcess()
        k.SetPriorityClass(h, 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS (inherited by chromium)
        k.SetProcessAffinityMask(h, AFFINITY_MASK)  # inherited by child processes


def rss_gb() -> float:
    p = psutil.Process()
    tot = p.memory_info().rss
    for c in p.children(recursive=True):
        try:
            tot += c.memory_info().rss
        except psutil.Error:
            pass
    return tot / 2**30


# ------------------------------------------------------------------ helpers
def words(s):
    return norm_text(s).split()


def ascii_ok(s):
    return all(ord(ch) < 128 for ch in s)


def clean_label(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip().strip(":").strip()
    if t.isupper() and len(t) > 1:
        t = t.title() if len(t.split()) > 1 or len(t) > 3 else t
    return t


def short_ok(t: str, maxw=4, maxc=30) -> bool:
    w = words(t)
    return (1 <= len(w) <= maxw and len(t) <= maxc and ascii_ok(t) and not re.search(r"\d", t)
            and any(len(x) >= 2 and x.isalpha() for x in w) and '"' not in t)


def app_name(raw):
    if not raw or not isinstance(raw, str):
        return None
    a = re.split(r"\s[-:|]\s|[\[\(:]", raw)[0].strip()
    a = re.sub(r"\b(free|lite|pro|hd)\b\s*$", "", a, flags=re.I).strip()
    if not a or not ascii_ok(a) or re.search(r"\d", a) or len(a.split()) > 3 or len(a) > 24:
        return None
    if not re.fullmatch(r"[A-Za-z][A-Za-z '&.!]*", a):
        return None
    return a


def ngrams(toks, n=5):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


# ------------------------------------------------------------------ GT parsing -> candidates
def parse_gt(html: str):
    root = lxml.html.fromstring(html)
    els = []
    for el in root.iter():
        lab = el.get("data-label") if isinstance(el.tag, str) else None
        if not lab:
            continue
        own = (el.text or "").strip() if el.tag in ("p", "button", "div", "li", "article") else ""
        if el.tag == "input":
            own = (el.get("placeholder") or el.get("aria-label") or "").strip()
        elif el.tag == "img":
            own = (el.get("alt") or "").strip()
        els.append((lab, re.sub(r"\s+", " ", own), el))
    return els


def best_token(text, df):
    cands = [w for w in words(text) if w.isascii() and w.isalpha() and len(w) >= 3 and w not in STOP]
    if not cands:
        return None
    return min(cands, key=lambda w: (df.get(w, 0), cands.index(w)))


def candidates(els, df):
    """Return list of (priority, order, item, mention_info). Keyword items first, type-only after."""
    out = []
    labels = collections.Counter(l for l, _, _ in els)
    for i, (lab, own, el) in enumerate(els):
        if lab == "Input" and own:
            if short_ok(own, 3):
                out.append((1, i, {"type": "Input", "keyword": norm_text(own)}, {"label": clean_label(own)}))
            elif (t := best_token(own, df)):
                out.append((1.5, i, {"type": "Input", "keyword": t}, {"token": t}))
        elif lab == "Text Button" and own:
            if short_ok(own):
                out.append((2, i, {"type": "Text Button", "keyword": norm_text(own)}, {"label": clean_label(own)}))
            elif (t := best_token(own, df)):
                out.append((4, i, {"type": "Text Button", "keyword": t}, {"token": t}))
        elif lab == "Toolbar":
            title = next((x.text.strip() for x in el.iter() if isinstance(x.tag, str) and x.get("data-label") == "Text"
                          and x.text and x.text.strip()), "")
            if title and short_ok(title):
                out.append((2.5, i, {"type": "Toolbar", "keyword": norm_text(title)}, {"label": clean_label(title)}))
            elif title and (t := best_token(title, df)):
                out.append((2.6, i, {"type": "Toolbar", "keyword": t}, {"token": t}))
        elif lab in ("Checkbox", "On/Off Switch", "Radio Button"):
            txt = own
            if not txt and el.getparent() is not None:  # label from a Text sibling in the same row
                txt = next(((s.text or "").strip() for s in el.getparent() if isinstance(s.tag, str)
                            and s.get("data-label") == "Text" and (s.text or "").strip()), "")
            if txt and short_ok(txt):
                out.append((3, i, {"type": lab, "keyword": norm_text(txt)}, {"label": clean_label(txt)}))
            elif txt and (t := best_token(txt, df)):
                out.append((3.2, i, {"type": lab, "keyword": t}, {"token": t}))
        elif lab in ("Bottom Navigation", "Multi-Tab"):
            tabs = [x.text.strip() for x in el.iter() if isinstance(x.tag, str) and x.get("data-label")
                    in ("Text", "Text Button") and x.text and short_ok(x.text.strip(), 2, 16)]
            if tabs:
                out.append((3 if lab == "Bottom Navigation" else 3.5, i,
                            {"type": lab, "keyword": norm_text(tabs[0])}, {"label": clean_label(tabs[0])}))
        elif lab == "Text" and own:
            in_bar = any(a.get("data-label") in ("Toolbar", "Bottom Navigation", "Multi-Tab")
                         for a in el.iterancestors() if isinstance(a.tag, str))
            if in_bar:
                continue
            w = words(own)
            if short_ok(own) and len(w) >= 2 and w[-1] not in STOP and any(len(x) >= 3 and x not in STOP for x in w):
                out.append((5, i, {"type": "Text", "keyword": norm_text(own)}, {"label": clean_label(own)}))
            elif len(w) >= 5 and (t := best_token(own, df)) and len(t) >= 4 and df.get(t, 0) <= 200:
                out.append((6, i, {"type": "Text", "keyword": t}, {"token": t}))
            elif len(w) == 1 and short_ok(own) and w[0].isalpha() and len(w[0]) >= 3 and w[0] not in STOP:
                out.append((10, i, {"type": "Text", "keyword": w[0]}, {"label": clean_label(own)}))  # late fallback
    # type-only fallbacks (one per type)
    big = len(els) + 1
    if labels["Input"]:
        out.append((7, big, {"type": "Input", "min_count": 2 if labels["Input"] >= 2 else 1}, {}))
    if labels["List Item"] >= 3:
        out.append((7.2, big, {"type": "List Item", "min_count": 2}, {}))
    for lab, p in [("Modal", 7.1), ("Drawer", 7.1), ("Bottom Navigation", 7.3), ("Multi-Tab", 7.4),
                   ("Slider", 6.5), ("Date Picker", 6.5), ("Number Stepper", 6.5), ("On/Off Switch", 6.6),
                   ("Checkbox", 6.6), ("Radio Button", 6.6), ("Toolbar", 8), ("Card", 8.2), ("Image", 8.5),
                   ("Text Button", 9)]:
        if labels[lab]:
            mc = 2 if lab in ("Card", "Radio Button") and labels[lab] >= 2 else 1
            out.append((p, big, {"type": lab, "min_count": mc}, {}))
    if labels["Icon"]:
        out.append((10.5, big, {"type": "Icon", "min_count": 1}, {}))  # late fallback
    out.sort(key=lambda c: (c[0], c[1]))
    return out


def select(valid, rng):
    k = rng.choice([2, 3, 3, 4])
    kw_items = [c for c in valid if "keyword" in c[2] and c[0] < 10]
    to_items = [c for c in valid if "keyword" not in c[2] and c[0] < 10]
    late = [c for c in valid if c[0] >= 10]  # single-word texts, icons: only when fewer than 2 items so far
    chosen, per_type, kws = [], collections.Counter(), []
    for c in kw_items:
        if len(chosen) >= k:
            break
        it = c[2]
        kn = it["keyword"]
        if per_type[it["type"]] >= (1 if it["type"] == "Text" else 2) or any(kn in o or o in kn for o in kws):
            continue
        chosen.append(c)
        per_type[it["type"]] += 1
        kws.append(kn)
    for c in to_items:
        if len(chosen) >= k:
            break
        if per_type[c[2]["type"]]:
            continue
        chosen.append(c)
        per_type[c[2]["type"]] += 1
    for c in late:
        if len(chosen) >= 2:
            break
        kn = c[2].get("keyword")
        if per_type[c[2]["type"]] or (kn and any(kn in o or o in kn for o in kws)):
            continue
        chosen.append(c)
        per_type[c[2]["type"]] += 1
    chosen.sort(key=lambda c: c[1])  # describe top-to-bottom
    return chosen


# ------------------------------------------------------------------ wording
def q(s):
    return f'"{s}"'


def phrase(item, info, rng, plural_inputs=False):
    t, lab, tok = item["type"], info.get("label"), info.get("token")
    C = rng.choice
    if t == "Text Button":
        if lab:
            return C([f"a {q(lab)} button", f"a button labeled {q(lab)}", f"a {q(lab)} button",
                      f"a button that says {q(lab)}"])
        if tok:
            return C([f"a button about {tok}", f"a button mentioning {tok}", f"a {tok} button"])
        return C(["a button", "an action button"])
    if t == "Input":
        if lab:
            return C([f"an input field for {lab.lower()}", f"a {lab.lower()} field",
                      f"a text box for {lab.lower()}"])
        if tok:
            return C([f"an input field about {tok}", f"a text field for {tok}"])
        return C(["some input fields", "text input fields"]) if item.get("min_count", 1) >= 2 else \
            C(["a text input field", "an input box"])
    if t == "Toolbar":
        if lab:
            return C([f"a top bar titled {q(lab)}", f"a header bar that says {q(lab)}",
                      f"an app bar with the title {q(lab)}"])
        if tok:
            return C([f"a top bar with a title about {tok}", f"a header bar mentioning {tok}"])
        return C(["a top toolbar", "a header bar"])
    if t in ("Checkbox", "On/Off Switch", "Radio Button"):
        noun = {"Checkbox": C(["a checkbox", "a check box"]), "On/Off Switch": C(["a toggle switch", "an on/off switch"]),
                "Radio Button": "radio buttons"}[t]
        what = q(lab) if lab else tok
        if what:
            return C([f"{noun} for {what}", f"{noun} labeled {what}"]) if t != "Radio Button" else \
                f"radio buttons including {what}"
        return noun if t != "Radio Button" else C(["a set of radio buttons", "radio buttons to choose from"])
    if t == "Bottom Navigation":
        if lab:
            return C([f"a bottom navigation bar with a {q(lab)} tab", f"a bottom nav bar that includes {q(lab)}"])
        return C(["a bottom navigation bar", "a bottom nav bar"])
    if t == "Multi-Tab":
        if lab:
            return C([f"tabs including {q(lab)}", f"a row of tabs with a {q(lab)} tab"])
        return C(["a row of tabs", "tabs at the top"])
    if t == "Text":
        if lab:
            return C([f"a line of text saying {q(lab)}", f"a heading {q(lab)}", f"the text {q(lab)}"])
        return C([f"some text about {tok}", f"a short description mentioning {tok}", f"a note about {tok}"])
    simple = {
        "List Item": ["a list of items", "a scrollable list", "a list of entries"],
        "Image": ["an image", "a picture"],
        "Modal": ["a pop-up dialog", "a dialog box"],
        "Drawer": ["a side drawer menu", "a navigation drawer"],
        "Card": ["a few cards", "content cards"],
        "Slider": ["a slider", "a slider control"],
        "Date Picker": ["a date picker", "a date field"],
        "Number Stepper": ["a number stepper", "a quantity stepper"],
        "Icon": ["a few icons", "some icon buttons"],
    }
    return C(simple[t])


def noun_phrase(summary, rng):
    s = summary.strip().rstrip(".").strip()
    s = s[0].lower() + s[1:] if s else s
    ws = re.findall(r"[a-z\-]+", s.lower())
    if ws and ws[0] in {"a", "an", "the"}:
        return s
    verb = {"shows": "showing", "displays": "displaying", "displayed": "displaying", "show": "showing",
            "display": "displaying"}
    if ws and (ws[0] in verb or (ws[0].endswith("ing") and ws[0] not in SCREEN_NOUNS)):
        return "a screen " + " ".join([verb.get(ws[0], s.split()[0])] + s.split()[1:])
    if any(w in SCREEN_NOUNS for w in ws[:3]):
        return ("an " if s[:1] in "aeiou" else "a ") + s
    return rng.choice(["a screen showing ", "a mobile screen showing ", "an app screen showing "]) + s


PURPOSE = ["Create {np}.", "Design {np}.", "I need {np}.", "Build {np}.", "Make {np}.", "Can you create {np}?"]
PURPOSE_APP = ["Create {np} for the {app}.", "For my {app}, I need {np}.", "Design {np} in the {app}.",
               "Build {np} for the {app}."]
ELEM = ["It should have {L}.", "Include {L}.", "The screen needs {L}.", "Make sure it has {L}.", "Add {L}.",
        "Please put {L} on it.", "It needs {L}."]


def join(parts, rng):
    if len(parts) == 1:
        return parts[0]
    sep = rng.choice([", and ", " and "]) if len(parts) > 2 else " and "
    return ", ".join(parts[:-1]) + sep + parts[-1]


def compose(summary, app, chosen, rng):
    np_ = noun_phrase(summary, rng)
    if app and re.search(r"(?<![a-z])app(lication)?s?(?![a-z])", summary, re.I):  # summary already names the app
        app = None
    if app:
        appp = app if app.lower().endswith(" app") else f"{app} app"
        purpose = rng.choice(PURPOSE_APP).format(np=np_, app=appp)
    else:
        purpose = rng.choice(PURPOSE).format(np=np_)
    purpose = purpose[0].upper() + purpose[1:]
    if not chosen:
        return purpose
    L = join([phrase(c[2], c[3], rng) for c in chosen], rng)
    return purpose + " " + rng.choice(ELEM).format(L=L)


def leak_check(query, checklist, gt_grams):
    nq = norm_text(query)
    unquoted = re.sub(r'"[^"]*"', " ", query)  # count words inside quoted GT labels (e.g. "Big Ten") are not counts
    res = {"digits": bool(re.search(r"\d", query)), "count_words": bool(NUM_WORDS.search(unquoted)),
           "verbatim": sorted(" ".join(g) for g in ngrams(nq.split()) & gt_grams)[:3], "unmentioned": []}
    for it in checklist:
        tw = TYPE_WORDS.get(it["type"])
        ok_kw = kw_in(it["keyword"], nq) if it.get("keyword") else True
        ok_ty = True if tw is None else bool(re.search(tw, query, re.I))
        if not (ok_kw and ok_ty):
            res["unmentioned"].append(it["type"] + ":" + str(it.get("keyword")))
    res["ok"] = not (res["digits"] or res["count_words"] or res["verbatim"] or res["unmentioned"])
    return res


# ------------------------------------------------------------------ main
def load(name):
    with open(PROC / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            yield {k: r[k] for k in ("id", "screen_id", "source_summary", "source_summary_index", "all_summaries",
                                     "gt_html_flow", "screenshot_path", "category")}


def load_app_names():
    names = {}
    for f in sorted(glob.glob(str(RAW / "*.parquet"))):
        for r in pq.read_table(f, columns=["screenId", "play_store_name"]).to_pylist():
            names[r["screenId"]] = r["play_store_name"]
    return names


def build_df(split_names):
    df = collections.Counter()
    for s in split_names:
        for r in load(s):
            for lab, own, _ in parse_gt(r["gt_html_flow"]):
                df.update(set(w for w in words(own) if w.isalpha()))
    return df


def process_row(r, a, df, apps, stats):
    rng = random.Random(f"{SEED}:{r['id']}")
    els = parse_gt(r["gt_html_flow"])
    cands = candidates(els, df)
    valid = []
    for c in cands:
        if item_ok(c[2], a)[0]:
            valid.append(c)
        else:
            stats["rejected_by_normalizer"][c[2]["type"] + ("+kw" if "keyword" in c[2] else "")] += 1
    stats["cand_total"] += len(cands)
    stats["cand_valid"] += len(valid)
    chosen = select(valid, rng)
    checklist = [c[2] for c in chosen]
    gt_texts = [own for _, own, _ in els if own]
    gt_grams = set().union(*[ngrams(norm_text(t).split()) for t in gt_texts]) if gt_texts else set()
    app = app_name(apps.get(r["screen_id"])) if rng.random() < 0.5 else None
    # purpose candidates: chosen consensus summary first, then the other human captions
    sums = [(r["source_summary_index"], r["source_summary"])] + [
        (i, s) for i, s in enumerate(r["all_summaries"]) if i != r["source_summary_index"] and len(s.split()) >= 4]
    first = None
    for tries, (si, s) in enumerate([x for x in sums for _ in range(3)]):  # 3 template draws per caption
        sub = random.Random(f"{SEED}:{r['id']}:{tries}")
        query = compose(s, app, chosen, sub)
        lk = leak_check(query, checklist, gt_grams)
        if first is None:
            first = (si, query, lk)
        if lk["ok"]:
            break
    else:
        si, query, lk = first
    n_kw = sum("keyword" in it for it in checklist)
    mode = "keyword" if n_kw == len(checklist) else ("type_only" if n_kw == 0 else "mixed")
    return {
        "id": r["id"], "query_mid": query, "checklist": checklist,
        "checklist_mode": mode, "n_items": len(checklist), "n_keyword_items": n_kw,
        "few_salient": n_kw < 2,
        "purpose_summary_index": si, "app_name": app,
        "leak_ok": lk["ok"], "leak": {k: v for k, v in lk.items() if k != "ok"},
        "n_candidates": len(cands), "n_candidates_gt_valid": len(valid),
        "category": r["category"], "screenshot_path": r["screenshot_path"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="dev,test,train_pool")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    limit_resources()
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = build_df(["dev", "test", "train_pool"])
    apps = load_app_names()
    print(f"df/app names ready {time.time() - t0:.0f}s, rss {rss_gb():.2f}GB", file=sys.stderr, flush=True)
    report_path = OUT / "build_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.update({"seed": SEED, "note": "checklist items are pre-validated on GT under metrics.NORMALIZE_JS"})
    with sync_playwright() as pw:
        br = pw.chromium.launch(headless=True, args=["--disable-gpu", "--renderer-process-limit=1"])
        sc = CL.ChecklistScorer(br)
        results = {}
        for split in args.splits.split(","):
            stats = {"rejected_by_normalizer": collections.Counter(), "cand_total": 0, "cand_valid": 0}
            rows, ts = [], time.time()
            for i, r in enumerate(load(split)):
                if args.limit and i >= args.limit:
                    break
                a = sc.analyze(r["gt_html_flow"])
                row = process_row(r, a, df, apps, stats)
                row["gt_self"] = CL.score_analysis(row["checklist"], a)["check_score"] if row["checklist"] else None
                row["usable"] = bool(row["checklist"]) and row["leak_ok"]  # evaluation should skip unusable rows
                rows.append(row)
                if i % 500 == 499:
                    g = rss_gb()
                    print(f"{split} {i + 1} {time.time() - ts:.0f}s rss {g:.2f}GB", file=sys.stderr, flush=True)
                    if g > RSS_LIMIT_GB:
                        sys.exit(f"RSS guard: {g:.2f}GB > {RSS_LIMIT_GB}GB")
            results[split] = rows
            with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n = len(rows)
            report[split] = {
                "rows": n, "sec": round(time.time() - ts, 1),
                "leak_ok_rate": round(sum(r["leak_ok"] for r in rows) / n, 4),
                "leak_fail_counts": {k: sum(bool(r["leak"][k]) for r in rows)
                                     for k in ("digits", "count_words", "verbatim", "unmentioned")},
                "usable": sum(r["usable"] for r in rows), "empty_checklist": sum(not r["checklist"] for r in rows),
                "gt_self_mean": round(sum(r["gt_self"] for r in rows if r["gt_self"] is not None)
                                      / max(1, sum(r["gt_self"] is not None for r in rows)), 4),
                "gt_self_lt1": sum(r["gt_self"] is not None and r["gt_self"] < 1 for r in rows),
                "mode_counts": dict(collections.Counter(r["checklist_mode"] for r in rows)),
                "n_items_counts": dict(sorted(collections.Counter(r["n_items"] for r in rows).items())),
                "few_salient": sum(r["few_salient"] for r in rows),
                "item_type_counts": dict(collections.Counter(it["type"] + ("+kw" if "keyword" in it else "")
                                                            for r in rows for it in r["checklist"]).most_common()),
                "candidates_total": stats["cand_total"], "candidates_gt_valid": stats["cand_valid"],
                "rejected_by_normalizer": dict(stats["rejected_by_normalizer"].most_common()),
                "query_words_mean": round(sum(len(r["query_mid"].split()) for r in rows) / n, 1),
            }
            print(split, json.dumps(report[split]), file=sys.stderr, flush=True)
        br.close()
    if "test" in results:  # pilot is a subset of test (same ids, same rows)
        pilot_ids = [json.loads(l)["id"] for l in open(PROC / "pilot.jsonl", encoding="utf-8")]
        by = {r["id"]: r for r in results["test"]}
        if all(i in by for i in pilot_ids):
            with open(OUT / "pilot.jsonl", "w", encoding="utf-8") as f:
                for i in pilot_ids:
                    f.write(json.dumps(by[i], ensure_ascii=False) + "\n")
            report["pilot"] = {"rows": len(pilot_ids), "source": "subset of test",
                               "leak_ok_rate": round(sum(by[i]["leak_ok"] for i in pilot_ids) / len(pilot_ids), 4)}
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"done {time.time() - t0:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
