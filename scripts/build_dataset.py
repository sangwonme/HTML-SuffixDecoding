"""Build the (user query -> UI HTML) dataset from Screen2Words (RICO).

Deterministic, re-runnable. Usage:
    .venv/Scripts/python.exe -I scripts/build_dataset.py
Outputs (data/processed/):
    pilot.jsonl dev.jsonl test.jsonl train_pool.jsonl  (one JSON object per screen)
    screenshots/<id>.jpg                               (eval samples only)
    build_report.json                                  (filter counts, split sizes)
Raw parquet in data/raw/ is untrusted data: it is only parsed as JSON/strings, never executed.
"""
from __future__ import annotations

import collections
import glob
import hashlib
import html
import json
import random
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "RICO-Screen2Words" / "data"
OUT = ROOT / "data" / "processed"
SEED = 20261107
N_PILOT, N_DEV, N_TEST = 50, 100, 500

# RICO screens are 1440x2560 px; we target a 360x640 CSS-px mobile viewport.
SCREEN_W, SCREEN_H = 1440, 2560
SCALE = 360 / SCREEN_W

MIN_ELEMS, MAX_ELEMS = 3, 150
MIN_SUMMARY_WORDS = 4
MIN_CONTENT_WORDS = 2  # summary needs >=2 words outside GENERIC_WORDS
GENERIC_WORDS = {
    "a", "an", "the", "of", "in", "on", "for", "and", "to", "is", "are", "with", "which", "that", "this",
    "page", "screen", "display", "displaying", "displayed", "displays", "show", "shows", "showing", "shown",
    "app", "application", "option", "options", "list", "various", "different", "multiple", "some", "many",
    "information", "info", "details", "detail", "button", "buttons", "icon", "icons", "text", "pop", "up",
    "popup", "menu", "view", "window", "box", "image", "images", "an", "it", "its", "has", "having",
}
LOW_CONTENT_LABELS = {"Web View", "Map View", "Video", "Advertisement"}  # Background Image excluded: often sits behind real forms
LOW_CONTENT_AREA = 0.5  # one such element covering >=50% of the screen -> drop
TEXT_LABELS = {"Text", "Text Button", "Input", "Toolbar", "Checkbox", "Radio Button", "On/Off Switch"}
MAX_KEY_TEXTS = 15
MAX_TEXT_CHARS = 40

# ---------------------------------------------------------------------------
# componentLabel -> (HTML tag, extra attrs). Documented in DATASET.md.
TAG_MAP = {
    "Text": "p",
    "Text Button": "button",
    "Input": "input",
    "Image": "img",
    "Icon": "span",
    "List Item": "li",
    "Toolbar": "header",
    "Card": "article",
    "Checkbox": "input:checkbox",
    "Radio Button": "input:radio",
    "On/Off Switch": "input:switch",
    "Slider": "input:range",
    "Number Stepper": "input:number",
    "Date Picker": "input:date",
    "Modal": "div:dialog",
    "Drawer": "aside",
    "Multi-Tab": "nav:tablist",
    "Bottom Navigation": "nav",
    "Button Bar": "div:group",
    "Pager Indicator": "div:pager",
    "Background Image": "div:bgimage",
    "Advertisement": "aside:ad",
    "Web View": "div:webview",
    "Map View": "div:map",
    "Video": "video",
}
VOID = {"input", "img"}

BASE_CSS_FLOW = (
    "*{box-sizing:border-box}body{margin:0;width:360px;font:14px/1.3 sans-serif}"
    ".row{display:flex;flex-direction:row;align-items:center;gap:4px}"
    ".col{display:flex;flex-direction:column;gap:2px}"
    "p{margin:2px 4px}ul{list-style:none;margin:0;padding:0}"
    "img,.ph{background:#ccc;display:inline-block}"
    ".icon{display:inline-block;width:24px;height:24px;background:#888;border-radius:4px}"
    "header{background:#eee;padding:4px}"
)
BASE_CSS_ABS = (
    "*{box-sizing:border-box}body{margin:0;width:360px;height:640px;position:relative;overflow:hidden;"
    "font:11px/1.2 sans-serif}"
    ".a{position:absolute;margin:0;padding:0;overflow:hidden}"
    "img.a,.ph{background:#ccc}.icon{background:#888;border-radius:3px}"
    "header.a{background:#eee}button.a{font-size:11px}"
)


def clean_text(t) -> str:
    if not isinstance(t, str):
        return ""
    return re.sub(r"\s+", " ", t).strip()


def area(b) -> int:
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def valid_bounds(b) -> bool:
    """Node-level 'invisible' rule: zero/negative area or fully off-screen."""
    if not (isinstance(b, list) and len(b) == 4):
        return False
    x1, y1, x2, y2 = b
    if x2 <= x1 or y2 <= y1:
        return False
    if x2 <= 0 or y2 <= 0 or x1 >= SCREEN_W or y1 >= SCREEN_H:
        return False
    return True


def prune(node, stats) -> dict | None:
    """Return a cleaned tree: drop invisible nodes; keep only fields we need."""
    b = node.get("bounds")
    is_root = stats is not None and stats.get("_root", False)
    if not is_root and not valid_bounds(b):
        stats["invisible_nodes"] += 1
        return None
    if stats is not None:
        stats["_root"] = False
    kids = []
    for c in node.get("children") or []:
        pc = prune(c, stats)
        if pc is not None:
            kids.append(pc)
    kids.sort(key=lambda n: (n["bounds"][1], n["bounds"][0]))
    return {
        "label": node.get("componentLabel"),
        "text": clean_text(node.get("text")),
        "icon": clean_text(node.get("iconClass")),
        "bounds": b if isinstance(b, list) and len(b) == 4 else [0, 0, SCREEN_W, SCREEN_H],
        "children": kids,
    }


def iter_labeled(n):
    if n["label"]:
        yield n
    for c in n["children"]:
        yield from iter_labeled(c)


# ---------------------------------------------------------------- summaries
def _tok(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def choose_summary(captions):
    """Consensus rule: rank captions by mean token-Jaccard to the other captions
    (most representative first); tie-break by original index. Return the first one
    with >= MIN_SUMMARY_WORDS words and >= MIN_CONTENT_WORDS non-generic words, or None."""
    caps = [clean_text(c) for c in captions]
    toks = [_tok(c) for c in caps]
    scores = []
    for i, ti in enumerate(toks):
        sims = [len(ti & tj) / len(ti | tj) if (ti | tj) else 0.0
                for j, tj in enumerate(toks) if j != i]
        scores.append((-(sum(sims) / max(1, len(sims))), i))
    for _, i in sorted(scores):
        content = [w for w in re.findall(r"[a-z0-9]+", caps[i].lower()) if w not in GENERIC_WORDS]
        if len(caps[i].split()) >= MIN_SUMMARY_WORDS and len(content) >= MIN_CONTENT_WORDS:
            return caps[i], i
    return None, None


SCREEN_NOUNS = {
    "page", "screen", "display", "window", "dialog", "popup", "pop-up", "pop", "app", "application",
    "list", "menu", "notification", "widget", "interface", "form", "message", "box", "tab", "view",
    "option", "options", "settings", "setting", "profile", "home", "homepage", "login", "log", "sign",
    "search", "map", "calendar", "gallery", "alert", "prompt", "panel", "section", "bar", "keyboard",
}


def make_query_short(summary: str) -> str:
    """Deterministic template:
    - starts with an article       -> "Create {s}."
    - one of first 3 words is a screen noun -> "Create a {s}."
    - otherwise                    -> "Create a mobile app screen showing {s}."
    """
    s = summary.strip().rstrip(".").strip()
    s = s[0].lower() + s[1:] if s else s
    words = re.findall(r"[a-z\-]+", s.lower())
    if words and words[0] in {"a", "an", "the"}:
        q = f"Create {s}."
    elif any(w in SCREEN_NOUNS for w in words[:3]):
        art = "an" if s[:1] in "aeiou" else "a"
        q = f"Create {art} {s}."
    else:
        q = f"Create a mobile app screen showing {s}."
    return q


def plural(label, n):
    name = label.lower()
    if n == 1:
        return f"1 {name}"
    if name.endswith("x"):
        return f"{n} {name}es"
    return f"{n} {name}s"


def make_query_rich(q_short: str, tree) -> tuple[str, list[str], dict]:
    labeled = list(iter_labeled(tree))
    counts = collections.Counter(n["label"] for n in labeled)
    order = sorted(counts, key=lambda k: (-counts[k], k))
    texts, seen = [], set()
    for n in labeled:  # pre-order with children sorted top->bottom, left->right
        if n["label"] in TEXT_LABELS and n["text"]:
            t = n["text"][:MAX_TEXT_CHARS]
            if t.lower() not in seen:
                seen.add(t.lower())
                texts.append(t)
        if len(texts) >= MAX_KEY_TEXTS:
            break
    lines = [q_short, "", "Elements: " + ", ".join(plural(k, counts[k]) for k in order) + "."]
    if texts:
        lines.append("Key texts (top to bottom): " + "; ".join(f'"{t}"' for t in texts) + ".")
    return "\n".join(lines), texts, dict(counts)


# ---------------------------------------------------------------- HTML
def esc(s):
    return html.escape(s, quote=True)


def open_tag(label, text, icon, extra_cls="", style=""):
    spec = TAG_MAP.get(label, "div")
    tag, _, kind = spec.partition(":")
    attrs = [f'data-label="{esc(label)}"']
    cls = [c for c in [extra_cls] if c]
    if tag == "input":
        itype = {"checkbox": "checkbox", "radio": "radio", "switch": "checkbox", "range": "range",
                 "number": "number", "date": "date"}.get(kind, "text")
        attrs.append(f'type="{itype}"')
        if kind == "switch":
            attrs.append('role="switch"')
        if itype == "text" and text:
            attrs.append(f'placeholder="{esc(text)}"')
        elif text:
            attrs.append(f'aria-label="{esc(text)}"')
    elif tag == "img":
        attrs.append(f'alt="{esc(text)}"')
    elif kind == "dialog":
        attrs.append('role="dialog"')
    elif kind == "tablist":
        attrs.append('role="tablist"')
    elif kind == "group":
        attrs.append('role="group"')
    elif kind:
        cls.append(kind)
    if label == "Icon":
        cls.append("icon")
        if icon:
            attrs.append(f'aria-label="{esc(icon)}"')
    if label in {"Image", "Background Image", "Map View", "Web View", "Video"}:
        cls.append("ph")
    if cls:
        attrs.append(f'class="{" ".join(cls)}"')
    if style:
        attrs.append(f'style="{style}"')
    return tag, f"<{tag} {' '.join(attrs)}>"


def is_row(children):
    if len(children) < 2:
        return False
    top = max(c["bounds"][1] for c in children)
    bot = min(c["bounds"][3] for c in children)
    return top < bot  # all children share a horizontal band


def flow_nodes(n, out):
    """Collapse unlabeled wrappers: return list of html strings for this subtree."""
    kids_html = []
    kids = n["children"]
    i = 0
    while i < len(kids):
        c = kids[i]
        if c["label"] == "List Item":  # group consecutive list items into <ul>
            j = i
            items = []
            while j < len(kids) and kids[j]["label"] == "List Item":
                items.extend(flow_nodes(kids[j], out))
                j += 1
            ul_cls = "row" if is_row(kids[i:j]) else "col"
            kids_html.append(f'<ul class="{ul_cls}">' + "".join(items) + "</ul>")
            i = j
            continue
        kids_html.extend(flow_nodes(c, out))
        i += 1
    layout = "row" if is_row(kids) else "col"
    if not n["label"]:
        if len(kids_html) > 1:
            return [f'<div class="{layout}">' + "".join(kids_html) + "</div>"]
        return kids_html
    w = round((n["bounds"][2] - n["bounds"][0]) * SCALE)
    h = round((n["bounds"][3] - n["bounds"][1]) * SCALE)
    style = f"width:{w}px;height:{h}px" if n["label"] in {"Image", "Icon"} else ""
    tag, ot = open_tag(n["label"], n["text"], n["icon"], extra_cls=layout if kids_html else "", style=style)
    if tag in VOID:  # void element cannot hold children: emit children as following siblings
        return [ot] + kids_html
    if tag in {"p", "button"} and kids_html:  # avoid parser auto-closing (block/button nesting)
        ot, tag = "<div" + ot[len(tag) + 1:], "div"
    inner = esc(n["text"]) if n["text"] else ""
    return [ot + inner + "".join(kids_html) + f"</{tag}>"]


def abs_nodes(n, px, py):
    b = n["bounds"]
    parts = []
    if n["label"]:
        x, y = (b[0] - px) * SCALE, (b[1] - py) * SCALE
        w, h = (b[2] - b[0]) * SCALE, (b[3] - b[1]) * SCALE
        style = f"left:{x:.0f}px;top:{y:.0f}px;width:{w:.0f}px;height:{h:.0f}px"
        tag, ot = open_tag(n["label"], n["text"], n["icon"], extra_cls="a", style=style)
        if tag in VOID:  # children become siblings positioned in the parent's frame
            out = [ot]
            for c in n["children"]:
                out.extend(abs_nodes(c, px, py))
            return out
        if tag in {"p", "button"} and n["children"]:
            ot, tag = "<div" + ot[len(tag) + 1:], "div"
        inner = esc(n["text"]) if n["text"] else ""
        kids = []
        for c in n["children"]:
            kids.extend(abs_nodes(c, b[0], b[1]))
        return [ot + inner + "".join(kids) + f"</{tag}>"]
    for c in n["children"]:
        parts.extend(abs_nodes(c, px, py))
    return parts


def doc(body, css, title):
    return ("<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=360\">"
            f"<title>{esc(title)}</title><style>{css}</style></head><body>{body}</body></html>")


def to_html_flow(tree, title):
    return doc("".join(flow_nodes(tree, None)), BASE_CSS_FLOW, title)


def to_html_abs(tree, title):
    return doc("".join(abs_nodes(tree, 0, 0)), BASE_CSS_ABS, title)


def struct_signature(tree):
    items = [(n["label"], n["text"]) for n in iter_labeled(tree)]
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()


# ---------------------------------------------------------------- main
def load_split(split):
    cols = ["screenId", "captions", "app_package_name", "category", "semantic_annotations"]
    rows = []
    for f in sorted(glob.glob(str(RAW / f"{split}-*.parquet"))):
        rows.extend(pq.read_table(f, columns=cols).to_pylist())
    return rows


def process(rows, filt):
    kept = []
    seen_sig = set()
    for r in sorted(rows, key=lambda r: r["screenId"]):
        filt["input"] += 1
        try:
            raw = json.loads(r["semantic_annotations"])
        except Exception:
            filt["F0_parse_error"] += 1
            continue
        nstat = {"invisible_nodes": 0, "_root": True}
        tree = prune(raw, nstat)
        filt["_invisible_nodes_removed"] += nstat["invisible_nodes"]
        rb = tree["bounds"]
        if (rb[2] - rb[0]) > (rb[3] - rb[1]):
            filt["F1_landscape"] += 1
            continue
        labeled = list(iter_labeled(tree))
        if len(labeled) < MIN_ELEMS:
            filt["F2_too_few_elements(<3)"] += 1
            continue
        if len(labeled) > MAX_ELEMS:
            filt["F3_too_many_elements(>150)"] += 1
            continue
        if not any(n["text"] for n in labeled if n["label"] in TEXT_LABELS):
            filt["F4_no_visible_text"] += 1
            continue
        screen_area = SCREEN_W * SCREEN_H
        if any(n["label"] in LOW_CONTENT_LABELS and area(n["bounds"]) >= LOW_CONTENT_AREA * screen_area
               for n in labeled):
            filt["F5_low_content(webview/map/video/ad>=50%)"] += 1
            continue
        summary, cap_idx = choose_summary(r["captions"])
        if summary is None:
            filt["F6_low_quality_summary(<4 words or <2 content words)"] += 1
            continue
        sig = (r["app_package_name"], struct_signature(tree))
        if sig in seen_sig:
            filt["F7_duplicate_in_app"] += 1
            continue
        seen_sig.add(sig)
        filt["kept"] += 1
        q_short = make_query_short(summary)
        q_rich, key_texts, counts = make_query_rich(q_short, tree)
        sid = f"s2w_{r['screenId']}"
        kept.append({
            "id": sid,
            "screen_id": r["screenId"],
            "app": r["app_package_name"],
            "category": r["category"],
            "source_summary": summary,
            "source_summary_index": cap_idx,
            "all_summaries": [clean_text(c) for c in r["captions"]],
            "query_short": q_short,
            "query_rich": q_rich,
            "key_texts": key_texts,
            "element_counts": counts,
            "gt_html_flow": to_html_flow(tree, summary),
            "gt_html_abs": to_html_abs(tree, summary),
            "screenshot_path": None,
        })
    return kept


def stratified(rows, n, rng):
    """Proportional allocation by category (largest remainder), seeded sample within category."""
    by = collections.defaultdict(list)
    for r in rows:
        by[r["category"]].append(r)
    total = len(rows)
    quota = {c: n * len(v) / total for c, v in by.items()}
    alloc = {c: int(q) for c, q in quota.items()}
    rest = n - sum(alloc.values())
    for c in sorted(quota, key=lambda c: (-(quota[c] - alloc[c]), c))[:rest]:
        alloc[c] += 1
    out = []
    for c in sorted(by):
        pool = sorted(by[c], key=lambda r: r["screen_id"])
        out.extend(rng.sample(pool, min(alloc[c], len(pool))))
    rng.shuffle(out)
    return out


def save_screenshots(split_files, wanted):
    """Extract original screenshots (JPEG bytes) for the selected eval ids only."""
    (OUT / "screenshots").mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in split_files:
        for f in sorted(glob.glob(str(RAW / f"{split}-*.parquet"))):
            pf = pq.ParquetFile(f)
            for g in range(pf.num_row_groups):
                t = pf.read_row_group(g, columns=["screenId"]).column(0).to_pylist()
                if not any(f"s2w_{s}" in wanted for s in t):
                    continue
                imgs = pf.read_row_group(g, columns=["screenId", "image"]).to_pylist()
                for r in imgs:
                    sid = f"s2w_{r['screenId']}"
                    if sid in wanted:
                        p = OUT / "screenshots" / f"{sid}.jpg"
                        p.write_bytes(r["image"]["bytes"])
                        paths[sid] = p.relative_to(ROOT).as_posix()
    return paths


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    report = {"seed": SEED, "filters": {}, "splits": {}}
    data = {}
    for split in ["train", "val", "test"]:
        filt = collections.Counter()
        data[split] = process(load_split(split), filt)
        report["filters"][split] = dict(filt)
        print(split, dict(filt), file=sys.stderr)

    apps = {s: {r["app"] for r in data[s]} for s in data}
    report["app_overlap"] = {
        "train&val": len(apps["train"] & apps["val"]),
        "train&test": len(apps["train"] & apps["test"]),
        "val&test": len(apps["val"] & apps["test"]),
    }
    test = stratified(data["test"], N_TEST, rng)
    pilot = stratified(test, N_PILOT, random.Random(SEED + 1))  # pilot is a subset of test
    dev = stratified(data["val"], N_DEV, random.Random(SEED + 2))
    train_pool = data["train"]

    wanted = {r["id"] for r in test + dev}
    paths = save_screenshots(["val", "test"], wanted)
    for r in test + dev:
        r["screenshot_path"] = paths.get(r["id"])
    pilot_ids = {r["id"] for r in pilot}
    pilot = [r for r in test if r["id"] in pilot_ids]

    for name, rows in [("pilot", pilot), ("dev", dev), ("test", test), ("train_pool", train_pool)]:
        write_jsonl(OUT / f"{name}.jsonl", rows)
        report["splits"][name] = len(rows)
    report["test_category_counts"] = dict(collections.Counter(r["category"] for r in test))
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["splits"]), file=sys.stderr)


if __name__ == "__main__":
    main()
