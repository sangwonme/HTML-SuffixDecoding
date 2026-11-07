"""PoC: RNSD (Render-Null relaxed Speculative Decoding) vs greedy vs PLD (lossless prompt-lookup decoding).

All three conditions use the SAME hand-written decode loop on the same model/prompt:
  greedy : draft = [] (one token per target forward)
  pld    : n-gram draft from context pool C (prompt incl. 3 few-shot HTML examples + own output),
           accept the longest prefix with d_i == g_i (lossless)
  rnsd   : same draft; a mismatch d_i != g_i is also accepted when
           (a) lexer state s_i in S_null (inter-tag whitespace, comment, data-*/id/aria-*/title attr value)
           (b) s_i in S_tol (CSS numeric / color value) and d_i in target top-5 and p(d_i) >= tau * p(g_i)
KV cache rollback: DynamicCache.activate_past_recording() + crop(-k) (same mechanism HF assisted decoding uses).

Usage: .venv/Scripts/python.exe -I scripts/poc_rnsd.py --model E2B --n 10 [--conds greedy,pld,rnsd]
Outputs runs/poc/<model>_<cond>.jsonl (one line per sample, plus _meta). Holds ../.gpu_busy while measuring.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoTokenizer, BitsAndBytesConfig, DynamicCache  # noqa: E402

torch.set_num_threads(4)
GPU_BUSY = ROOT.parent / ".gpu_busy"
RSS_LIMIT_GB = 14.0
MODELS = {"E2B": "gemma-4-E2B-it-qat-q4_0-unquantized", "E4B": "gemma-4-E4B-it-qat-q4_0-unquantized"}
SYSTEM = ("You generate mobile app UI screens as HTML. Given a user's request, output ONE complete, "
          "self-contained HTML document for a 360x640 px mobile viewport: inline <style> CSS only, "
          "no JavaScript, no external resources (no CDN, fonts, or remote images). Use semantic elements "
          "(header, button, input, img, ul/li, nav, etc.) and include the texts the user mentions. "
          "Reply with only the HTML inside a single ```html code block, no explanation.")
GAMMA = 16
NGRAM_MAX = 3
TOPK = 5
TAU = 0.1


def check_rss():
    rss = psutil.Process().memory_info().rss / 2**30
    if rss > RSS_LIMIT_GB:
        raise SystemExit(f"RSS guard: {rss:.1f} GB > {RSS_LIMIT_GB} GB")


# ----------------------------------------------------------------------------- lexer
NULL_ATTR = re.compile(r"^(data-[\w-]*|id|aria-[\w-]*|title)$", re.I)
TOL_PROPS = re.compile(
    r"^(padding|margin|gap|row-gap|column-gap|font-size|line-height|width|height|min-width|min-height|max-width|"
    r"max-height|top|left|right|bottom|border-radius|border-width|letter-spacing|flex-basis|"
    r"(padding|margin)-(top|right|bottom|left)|color|background|background-color|border-color|border|"
    r"border-(top|right|bottom|left)|outline-color|fill|stroke|box-shadow|opacity)$", re.I)
VALUE_TOKEN = re.compile(r"^\s*[#]?[0-9a-fA-F.]*(px|em|rem|%|vh|vw|pt)?\s*$")
ATTR_RE = re.compile(r"""([^\s=/>"']+)\s*=\s*("([^"]*)"?|'([^']*)'?)""")


def lexer_state(t: str) -> dict:
    """Coarse incremental-HTML lexer state at the END of prefix t (the model output so far)."""
    lc = t.rfind("<!--")
    if lc > t.rfind("-->"):
        return {"s": "comment"}
    lt, gt = t.rfind("<"), t.rfind(">")
    low = t.lower()
    ls, le = low.rfind("<style"), low.rfind("</style")
    if ls > le:
        open_end = t.find(">", ls)
        if open_end != -1 and lt <= open_end:  # no '<' after the <style ...> opener: inside the CSS block
            return _css_state(t[open_end + 1:], "style_block")
    if lt > gt:  # inside a tag
        tag = t[lt:]
        last = None
        for m in ATTR_RE.finditer(tag):
            last = m
        if last is not None and last.end() == len(tag):
            val = last.group(2)
            q = val[0] if val[:1] in "\"'" else None
            closed = q is not None and len(val) >= 2 and val.endswith(q)
            if q and not closed:
                name = last.group(1)
                if NULL_ATTR.match(name):
                    return {"s": "attr_null", "quote": q, "attr": name.lower()}
                if name.lower() == "class":
                    return {"s": "class_val", "quote": q}
                if name.lower() == "style":
                    st = _css_state(val[1:], "style_attr")
                    st["quote"] = q
                    return st
        return {"s": "tag"}
    seg = t[gt + 1:] if gt >= 0 else t
    if gt >= 0 and seg.strip() == "":
        return {"s": "ws_text"}
    return {"s": "text"}


def _css_state(css: str, where: str) -> dict:
    cut = max(css.rfind(";"), css.rfind("{"), css.rfind("}"))
    decl = css[cut + 1:]
    if ":" in decl and "{" not in decl and where == "style_attr" or (":" in decl and css.rfind("{") > css.rfind("}")):
        prop = decl.split(":", 1)[0].strip()
        return {"s": "css_value", "prop": prop, "where": where}
    if where == "style_block" and css.rfind("{") <= css.rfind("}") and re.search(r"\.[\w-]*$", decl):
        return {"s": "css_class_sel", "where": where}
    return {"s": "css_other", "where": where}


def null_ok(st: dict, d_text: str) -> bool:
    """Rule (a): state in S_null and the draft token stays inside that render-null region."""
    if st["s"] == "comment":
        return "-->" not in d_text and "-" not in d_text[-2:]
    if st["s"] == "attr_null":
        return st["quote"] not in d_text and ">" not in d_text and "<" not in d_text
    if st["s"] == "ws_text":
        return d_text.strip() == ""
    return False


def null_ok_v2(st: dict, d_text: str) -> bool:
    """Narrowed S_null (used by L4): whitespace, comments, data-* attribute values only
    (aria-*/title carry accessibility meaning / tooltips; id may be referenced by CSS or anchors)."""
    if st["s"] == "attr_null" and not st.get("attr", "").startswith("data-"):
        return False
    return null_ok(st, d_text)


def tol_state(st: dict) -> bool:
    return st["s"] == "css_value" and bool(TOL_PROPS.match(st.get("prop", "")))


IDENT = re.compile(r"^\s?[A-Za-z_][\w-]*$|^[\w-]+$")
OPT_END = {"li", "p", "td", "tr", "option"}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
TAG_RE = re.compile(r"<(/?)([a-zA-Z][\w-]*)[^>]*?(/?)>")


def innermost_open(t: str):
    stack = []
    for m in TAG_RE.finditer(t):
        name = m.group(2).lower()
        if name in VOID_TAGS or m.group(3):
            continue
        if m.group(1):
            if name in stack:
                while stack and stack.pop() != name:
                    pass
        else:
            stack.append(name)
    return stack[-1] if stack else None


CLASS_ATTR_RE = re.compile(r"""class\s*=\s*["']([^"']*)["']""")
SEL_RE = re.compile(r"\.([A-Za-z_][\w-]*)")


def bound_names(prefix: str) -> set:
    """Class names already bound in the output: complete selectors inside <style> and complete class="..." values."""
    names = set()
    low = prefix.lower()
    for m in re.finditer(r"<style[^>]*>(.*?)(</style>|$)", prefix, re.S | re.I):
        css = m.group(1)
        if m.group(2) == "":  # still inside the open style block: drop the identifier being typed
            css = re.sub(r"[\w-]*$", "", css)
        names.update(SEL_RE.findall(re.sub(r"\{[^}]*\}", "{}", css)))
    for m in CLASS_ATTR_RE.finditer(prefix):
        names.update(m.group(1).split())
    return names


LEVEL_RULES = {  # cumulative rule sets per condition
    "pld": set(),
    "rnsd": {"a", "b"},
    "L1": {"L1_ws", "L1_quote", "L1_optend"},
    "L2": {"L1_ws", "L1_quote", "L1_optend", "L2_rename"},
    "L4": {"L1_ws", "L1_quote", "L1_optend", "L2_rename", "a2", "b"},
}


def relaxed_accept(cond, rules, st, prefix, d, g, d_text, g_text, rest_text, logits_j, sigma, sigma_inv):
    """Return the rule name that accepts draft token d != target token g, or None."""
    if "L1_ws" in rules and st["s"] == "ws_text" and d_text.strip() == "" and g_text.strip() == "":
        return "L1_ws"
    if "L1_quote" in rules and {d_text, g_text} <= {'="', "='"} and st["s"] == "tag":
        return "L1_quote"
    if "L1_optend" in rules and st["s"] in ("ws_text", "text"):
        top = innermost_open(prefix)
        if top in OPT_END:
            # target closes </top>, draft starts the next sibling <top ...> (implicit close) -- or the reverse
            if g_text.startswith("</") and d_text.startswith("<") and not d_text.startswith("</")                     and re.match(rf"<{top}", rest_text):
                return "L1_optend"
            if d_text.startswith("</") and re.match(rf"</{top}\s*>", rest_text) and g_text.startswith("<")                     and not g_text.startswith("</"):
                return "L1_optend"
    if "L2_rename" in rules and st["s"] in ("class_val", "css_class_sel") and IDENT.match(d_text)             and IDENT.match(g_text) and d_text.strip() and g_text.strip():
        # binding-aware alpha-renaming: only FRESH names may be renamed. A name already bound earlier in the
        # output (a .selector in <style> or a class="..." use) refers to existing CSS rules, so substituting a
        # different name at a use site changes the computed style (v2 fix; v1 "naive" accepted any identifier).
        tail = re.search(r"[\w-]*$", prefix).group(0)
        g_full, d_full = tail + g_text.strip(), tail + d_text.strip()
        bound = bound_names(prefix)
        if any(b.startswith(g_full) or b.startswith(d_full) for b in bound):
            return None
        if sigma.get(g, d) == d and sigma_inv.get(d, g) == g:  # consistent bijection
            sigma[g] = d
            sigma_inv[d] = g
            return "L2_rename"
        return None
    if "a" in rules and null_ok(st, d_text):
        return "a"
    if "a2" in rules and null_ok_v2(st, d_text):
        return "a2"
    if "b" in rules and tol_state(st) and VALUE_TOKEN.match(d_text) and VALUE_TOKEN.match(g_text):
        p = torch.softmax(logits_j, -1)
        top = torch.topk(p, TOPK).indices.tolist()
        if d in top and float(p[d]) >= TAU * float(p[g]):
            return "b"
    return None


# ----------------------------------------------------------------------------- data
def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def _ex_ids(v):
    if isinstance(v, dict):
        v = v.get("ids") or v.get("top3") or v.get("exemplars") or list(v.values())[0]
    return [x if isinstance(x, str) else (x.get("id") or x.get("train_id")) for x in v][:3]


def build_prompts(n, fewshot="tfidf", exemplars=None, exemplar_html=None, draft_source_html=None, rows_file=None):
    from genui.retrieval import Retriever
    mid = ROOT / "data" / "processed" / "mid"
    pilot = load_jsonl(Path(rows_file) if rows_file else mid / "pilot.jsonl")[:n]
    app_of = {}
    if rows_file:  # requests from another split: exclude the request itself and same-app exemplars from retrieval
        for fn in ("train_pool.jsonl", "test.jsonl", "dev.jsonl", "pilot.jsonl"):
            with open(ROOT / "data" / "processed" / fn, encoding="utf-8") as f:
                for l in f:
                    r_ = json.loads(l)
                    app_of[r_["id"]] = r_["app"]
    pool_mid = load_jsonl(mid / "train_pool.jsonl")
    html_by_id = {}
    with open(ROOT / "data" / "processed" / "train_pool.jsonl", encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            html_by_id[r["id"]] = r["gt_html_flow"]
    orig_html = dict(html_by_id)  # exemplar SELECTION (incl. the 4000-char filter) always uses the original HTML
    if exemplar_html:  # e.g. interleaved / perturbed versions of the exemplar HTML: {train_id: html}
        alt = json.load(open(exemplar_html, encoding="utf-8"))
        html_by_id.update({k: v for k, v in alt.items() if k in html_by_id})
        print(f"exemplar html overridden for {len(alt)} ids", flush=True)
    draft_html = None
    if draft_source_html:  # P1-B: draft source only (prompt keeps original / --exemplar-html)
        alt2 = json.load(open(draft_source_html, encoding="utf-8"))
        draft_html = dict(orig_html)
        draft_html.update({k: v for k, v in alt2.items() if k in draft_html})
        print(f"draft-source html overridden for {len(alt2)} ids", flush=True)
    pool = [dict(r, gt_html_flow=orig_html[r["id"]]) for r in pool_mid if r["id"] in html_by_id]
    by_id = {r["id"]: r for r in pool}
    ex_map = None
    if exemplars:
        raw = json.load(open(exemplars, encoding="utf-8"))
        if isinstance(raw, list):
            raw = {x.get("id") or x.get("pilot_id"): x for x in raw}
        ex_map = {k: _ex_ids(v) for k, v in raw.items()}
    ret = Retriever(pool, field="query_mid") if fewshot == "tfidf" and not ex_map else None
    out = []
    for r in pilot:
        if ex_map is not None:
            if r["id"] not in ex_map:
                print("no exemplars for", r["id"], "- skipped", flush=True)
                continue
            exs = [by_id[i] for i in ex_map[r["id"]] if i in by_id]
        elif fewshot == "none":
            exs = []
        else:
            if rows_file:
                me = app_of.get(r["id"])
                exs = [x for x in ret.topk(r["query_mid"], 80, max_html_chars=4000)
                       if x["id"] != r["id"] and app_of.get(x["id"]) != me][:3]
            else:
                exs = ret.topk(r["query_mid"], 3, max_html_chars=4000)
        def mk(hmap):
            m_ = [{"role": "system", "content": SYSTEM}]
            for ex in exs:
                m_.append({"role": "user", "content": ex["query_mid"]})
                m_.append({"role": "assistant", "content": "```html\n" + hmap[ex["id"]] + "\n```"})
            m_.append({"role": "user", "content": r["query_mid"]})
            return m_
        out.append({"row": r, "msgs": mk(html_by_id), "draft_msgs": mk(draft_html) if draft_html else None,
                    "example_ids": [e["id"] for e in exs]})
    return out


# ----------------------------------------------------------------------------- draft
def ngram_draft(ctx: np.ndarray, gamma: int) -> list[int]:
    """Prompt-lookup draft: last n tokens (n=3..1) matched at their most recent earlier occurrence in ctx."""
    L = len(ctx)
    for n in range(NGRAM_MAX, 0, -1):
        if L <= n:
            continue
        pat = ctx[L - n:]
        win = np.lib.stride_tricks.sliding_window_view(ctx[: L - 1], n)  # windows ending before the last token
        hits = np.nonzero((win == pat).all(axis=1))[0]
        if len(hits):
            s = int(hits[-1]) + n
            d = ctx[s: s + gamma].tolist()
            if d:
                return d
    return []


# ----------------------------------------------------------------------------- DOM-aware drafter (gate 2)
_SIM = None


def _sim():
    """study_draft_sim (read-only import): Lexer, sim_score, Index -- the exact DOM rule of the simulator."""
    global _SIM
    if _SIM is None:
        import importlib.util
        prev = os.environ.get("SIM_THREADS")
        os.environ["SIM_THREADS"] = "4"
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location("study_draft_sim", ROOT / "scripts" / "study_draft_sim.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _SIM = m
        if prev is None:
            os.environ.pop("SIM_THREADS", None)
        for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[_v] = "4"
    return _SIM


class DomDrafter:
    """Incremental version of study_draft_sim's 'dom' policy.
    Annotation state (Lexer over html code-fence regions) and the n-gram index are updated only with NEW tokens
    (no re-parse per step). Selection rule identical to run_policy(policy='dom')."""

    def __init__(self, tok):
        self.S = _sim()
        self.tok, self.tcache = tok, {}
        self.ann, self.text, self.inhtml, self.lx, self.k, self.pend = [], "", False, None, -1, ""
        self.idx = self.S.Index(3)

    def _s(self, t):
        s = self.tcache.get(t)
        if s is None:
            s = self.tcache[t] = self.tok.decode([t])
        return s

    def state(self):
        return self.lx.state() if self.inhtml and self.lx else None

    def push(self, t):
        FENCE = "`" * 3 + "html"
        self.ann.append(self.state())
        for ch in self._s(t):
            self.text += ch
            if not self.inhtml:
                if self.text.endswith(FENCE + "\n") or self.text.endswith(FENCE):
                    self.pend = "open"
                if self.pend == "open" and ch == "\n" and self.text.rstrip("\n").endswith(FENCE):
                    self.inhtml, self.lx, self.k, self.pend = True, self.S.Lexer(), self.k + 1, ""
            else:
                if self.text.endswith("`" * 3):
                    self.inhtml = False
                    continue
                self.lx.feed(ch)
        if len(self.text) > 64:
            self.text = self.text[-64:]

    def draft(self, ctx, g):
        L = len(ctx)
        while len(self.ann) < L:
            self.push(int(ctx[len(self.ann)]))
        self.idx.extend(ctx, L)
        cur = self.state()
        hits = []
        for n in range(3, 0, -1):
            if L <= n:
                continue
            h = self.idx.hits(ctx, L, n)
            if h:
                hits = h
                break
        src, best = None, None
        if hits:
            for c in hits[-300:][::-1]:
                sc = self.S.sim_score(self.ann[c], cur)
                if best is None or sc > best:
                    best, src = sc, c
        elif cur is not None:
            for c in range(L - 1, 0, -1):
                if self.ann[c] is None:
                    continue
                sc = self.S.sim_score(self.ann[c], cur)
                if best is None or sc > best:
                    best, src = sc, c
        if src is None or g <= 0:
            return []
        return ctx[src: src + min(g, L - src)].tolist()


_RANKER = None


def _ranker_mod():
    """study_ranker (read-only import): RankerDrafter, load_weights."""
    global _RANKER
    if _RANKER is None:
        _sim()
        import importlib.util
        spec = importlib.util.spec_from_file_location("study_ranker", ROOT / "scripts" / "study_ranker.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _RANKER = m
        for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[_v] = "4"
    return _RANKER


# ----------------------------------------------------------------------------- decode
@torch.inference_mode()
def decode(model, tok, enc, cond, max_new, eos_ids, decoded_cache, draft_mode="ngram", ranker_w=None,
           draft_prompt_ids=None):
    ids = enc["input_ids"]
    n_prompt = ids.shape[1]
    cache = DynamicCache(config=model.config)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(**enc, past_key_values=cache, use_cache=True, logits_to_keep=1)
    first = int(out.logits[0, -1].argmax())
    torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    cache = out.past_key_values
    plain = cond.startswith("greedy")
    if not plain:
        cache.activate_past_recording()
    gen = [first]
    prompt_np = ids[0].cpu().numpy().astype(np.int64)
    if draft_prompt_ids is not None:  # P1-B: draft context = draft-source prompt + own output
        prompt_np = np.asarray(draft_prompt_ids, dtype=np.int64)
    ctx = np.concatenate([prompt_np, np.array([first], dtype=np.int64)])
    n_fwd = 1
    steps = []  # per verify step: k (draft len), accepted, relaxed positions
    fire_a = fire_b = 0
    fires = {}
    sigma, sigma_inv = {}, {}
    rules = LEVEL_RULES.get(cond, set())
    pos_reach = np.zeros(GAMMA + 1)
    pos_exact = np.zeros(GAMMA + 1)
    pos_acc = np.zeros(GAMMA + 1)
    t_fwd = 0.0
    t_draft = 0.0
    n_draft = 0
    drafter = None
    t_init = 0.0
    ranker = None
    if draft_mode == "dom" and not plain:
        ti = time.perf_counter()
        drafter = DomDrafter(tok)
        drafter.draft(ctx, 0)  # annotate + index the prompt (counted as draft cost)
        t_init = time.perf_counter() - ti
    elif draft_mode == "ranker" and not plain:
        ti = time.perf_counter()
        tcache = {}

        def _dec1(t):
            s_ = tcache.get(t)
            if s_ is None:
                s_ = tcache[t] = tok.decode([t])
            return s_
        ranker = _ranker_mod().RankerDrafter(prompt_np, _dec1, ranker_w, max_new=max_new)
        t_init = time.perf_counter() - ti
    done = first in eos_ids
    while not done and len(gen) < max_new:
        check_rss()
        if plain:
            draft = []
        else:
            td = time.perf_counter()
            gmax = min(GAMMA, max_new - len(gen) - 1)
            if ranker is not None:
                draft = ranker.draft(gen, GAMMA)
            elif drafter is not None:
                draft = drafter.draft(ctx, gmax)
            else:
                draft = ngram_draft(ctx, gmax)
            t_draft += time.perf_counter() - td
            n_draft += 1
        k = len(draft)
        inp = torch.tensor([[gen[-1]] + draft], device=0)
        tf = time.perf_counter()
        o = model(input_ids=inp, past_key_values=cache, use_cache=True)
        logits = o.logits[0].float()
        g = logits.argmax(-1).tolist()  # g[j] = target greedy token after inp[:j+1]
        n_fwd += 1
        accepted = []
        relaxed = []
        j = 0
        while j < k:
            d = draft[j]
            pos_reach[j] += 1
            gj = g[j]
            if gj in sigma and sigma[gj] == d and cond in ("L2", "L4"):
                gj = d  # binding-aware match: target name maps to the draft name under sigma
                fires["L2_sigma_match"] = fires.get("L2_sigma_match", 0) + 1
            if d == gj:
                pos_exact[j] += 1
                pos_acc[j] += 1
                accepted.append(d)
                j += 1
                continue
            ok = None
            if rules:
                prefix = decoded_cache(gen + accepted)
                st = lexer_state(prefix)
                rest = tok.decode(draft[j:j + 6]) if "L1_optend" in rules else ""
                ok = relaxed_accept(cond, rules, st, prefix, d, g[j], tok.decode([d]), tok.decode([g[j]]), rest,
                                    logits[j], sigma, sigma_inv)
            if ok is None:
                break
            pos_acc[j] += 1
            fire_a += ok == "a"
            fire_b += ok == "b"
            fires[ok] = fires.get(ok, 0) + 1
            relaxed.append((len(gen) + len(accepted), ok, tok.decode([d]), tok.decode([g[j]])))
            accepted.append(d)
            j += 1
        bonus = g[j]
        new = accepted + [bonus]
        rej = k - j
        if not plain:
            cache.crop(-rej if rej > 0 else 0)
        torch.cuda.synchronize()
        t_fwd += time.perf_counter() - tf
        for t in new:
            gen.append(t)
            if t in eos_ids or len(gen) >= max_new:
                done = True
                break
        ctx = np.concatenate([prompt_np, np.array(gen, dtype=np.int64)])
        steps.append({"k": k, "acc": j, "relaxed": relaxed})
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    return {
        "gen": gen, "n_prompt": n_prompt, "ttft_s": ttft, "total_s": total, "n_fwd": n_fwd,
        "decode_fwd_s_mean": t_fwd / max(1, n_fwd - 1), "verify_s": t_fwd, "draft_s": t_draft + t_init,
        "draft_init_s": t_init, "draft_steps": n_draft, "steps": steps, "fire_a": fire_a, "fire_b": fire_b,
        "fires": fires, "sigma": {str(k): v for k, v in sigma.items()},
        "pos_reach": pos_reach.tolist(), "pos_exact": pos_exact.tolist(), "pos_acc": pos_acc.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="E2B", choices=list(MODELS))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--conds", default="greedy,pld,rnsd")
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fewshot", default="tfidf", choices=["tfidf", "none"])
    ap.add_argument("--exemplar-html", default=None, help="json {train_id: html} replacing exemplar outputs")
    ap.add_argument("--draft", default="ngram", choices=["ngram", "dom"], help="draft source (cond 'dom' always dom)")
    ap.add_argument("--ranker-dir", default=str(ROOT / "runs" / "study"), help="ranker_mlp_fold*.npz + ranker_fold_map.json")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--prompts-file", default=None,
                    help="jsonl {id, msgs, example_ids}: ready chat prompts (e.g. external data); draft source = prompt + output")
    ap.add_argument("--rows", default=None, help="jsonl of query_mid rows to run instead of mid/pilot.jsonl")
    ap.add_argument("--ranker-weights", default=None, help="single npz for all requests (overrides fold map)")
    ap.add_argument("--draft-source-html", default=None, help="P1-B: json {train_id: html} for the draft source only")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--shuffle", action="store_true", help="random condition order per sample/rep (seeded)")
    ap.add_argument("--exemplars", default=None, help="json: pilot id -> 3 train_pool ids (overrides --fewshot)")
    args = ap.parse_args()
    vm = psutil.virtual_memory()
    if vm.available / 2**30 < 8:
        raise SystemExit("free RAM < 8 GB")
    if args.prompts_file:
        prompts = [{"row": {"id": x["id"]}, "msgs": x["msgs"], "draft_msgs": None, "example_ids": x.get("example_ids", [])}
                   for x in load_jsonl(Path(args.prompts_file))][: args.n]
    else:
        prompts = build_prompts(args.n, args.fewshot, args.exemplars, args.exemplar_html, args.draft_source_html, args.rows)
    check_rss()
    mpath = ROOT / "models" / MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(mpath)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        mpath, quantization_config=qcfg, dtype=torch.bfloat16, device_map={"": 0},
        attn_implementation={"text_config": "sdpa", "vision_config": "sdpa", "audio_config": "eager"})
    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])

    dec_memo = {}

    def decoded_cache(ids):
        # decode the output prefix (memoized on length; the prefix only grows within a sample)
        key = (len(ids), ids[-1] if ids else None)
        if key not in dec_memo:
            dec_memo[key] = tok.decode(ids, skip_special_tokens=True)
        return dec_memo[key]

    encs = [tok.apply_chat_template(p["msgs"], add_generation_prompt=True, return_tensors="pt", return_dict=True)
            for p in prompts]
    dprompt = [list(tok.apply_chat_template(p["draft_msgs"], add_generation_prompt=True, tokenize=True,
                                            return_dict=True)["input_ids"]) if p.get("draft_msgs") else None
               for p in prompts]
    dprompt_by_id = {p["row"]["id"]: d for p, d in zip(prompts, dprompt)}
    rw_by_id = {}
    if "ranker" in args.conds.split(","):
        RM = _ranker_mod()
        rdir = Path(args.ranker_dir)
        if args.ranker_weights:
            w1 = RM.load_weights(Path(args.ranker_weights))
            for p in prompts:
                rw_by_id[p["row"]["id"]] = (Path(args.ranker_weights).stem, w1)
        else:
            fmap = json.load(open(rdir / "ranker_fold_map.json", encoding="utf-8"))
            wf = {}
            for p in prompts:
                fo = fmap[p["row"]["id"]]  # fold that did NOT train on this request's app (no leakage)
                if fo not in wf:
                    wf[fo] = RM.load_weights(rdir / f"ranker_mlp_fold{fo}.npz")
                rw_by_id[p["row"]["id"]] = (fo, wf[fo])

    def dmode(c):
        return "ranker" if c == "ranker" else ("dom" if (c == "dom" or args.draft == "dom") else "ngram")
    out_dir = Path(args.outdir) if args.outdir else ROOT / "runs" / "poc"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "model_dir": MODELS[args.model], "quant": "nf4/bf16 compute", "attn": "sdpa(text)",
            "max_new": args.max_new, "gamma": GAMMA, "ngram_max": NGRAM_MAX, "topk": TOPK, "tau": TAU,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "free_ram_gb_start": round(vm.available / 2**30, 1), "started_utc": datetime.now(timezone.utc).isoformat(),
            "fewshot": args.prompts_file or args.exemplars or args.fewshot, "ranker_weights": args.ranker_weights, "exemplar_html": args.exemplar_html, "kv_rollback": "DynamicCache.activate_past_recording + crop(-k) (real wall-clock, no estimation)"}
    # warm-up on dev-like prompt variants (first prompts, short generations), every condition
    for c in args.conds.split(","):
        for p_, e in list(zip(prompts, encs))[: args.warmup]:
            dec_memo.clear()
            decode(model, tok, {k: v.to(0) for k, v in e.items()}, c, 32, eos_ids, decoded_cache,
                   draft_mode=dmode(c), ranker_w=rw_by_id.get(p_["row"]["id"], (None, None))[1],
                   draft_prompt_ids=dprompt_by_id.get(p_["row"]["id"]))
    print("warm-up done", flush=True)

    GPU_BUSY.write_text(f"GenerativeUI poc_rnsd pid={os.getpid()} since {datetime.now().isoformat()}\n")
    try:
        files = {c: open(out_dir / f"{args.model}_{c}{args.tag}.jsonl", "w", encoding="utf-8") for c in args.conds.split(",")}
        for c, f in files.items():
            f.write(json.dumps({"_meta": dict(meta, cond=c)}) + "\n")
        import random as _random
        jobs = []
        for rep_i in range(args.reps):
            for i, (p, e) in enumerate(zip(prompts, encs)):
                order = list(files)
                if args.shuffle:
                    _random.Random(f"{p['row']['id']}-{rep_i}-20261107").shuffle(order)
                jobs += [(rep_i, i, p, e, c) for c in order]
        for rep_i, i, p, e, c in jobs:
            if True:  # conditions interleaved per sample (same thermal / load conditions)
                f = files[c]
                dec_memo.clear()
                torch.cuda.reset_peak_memory_stats()
                r = decode(model, tok, {k: v.to(0) for k, v in e.items()}, c, args.max_new, eos_ids, decoded_cache,
                           draft_mode=dmode(c), ranker_w=rw_by_id.get(p["row"]["id"], (None, None))[1],
                           draft_prompt_ids=dprompt_by_id.get(p["row"]["id"]))
                text = tok.decode(r["gen"], skip_special_tokens=True)
                n_out = len(r["gen"])
                rec = {"id": p["row"]["id"], "cond": c, "example_ids": p["example_ids"], "prompt_tokens": r["n_prompt"],
                       "output_tokens": n_out, "ttft_s": r["ttft_s"], "total_s": r["total_s"], "n_fwd": r["n_fwd"],
                       "tokens_per_fwd": n_out / r["n_fwd"], "decode_fwd_s_mean": r["decode_fwd_s_mean"],
                       "hit_max_new": n_out >= args.max_new, "fire_a": r["fire_a"], "fire_b": r["fire_b"],
                       "fires": r["fires"], "sigma": r["sigma"],
                       "pos_reach": r["pos_reach"], "pos_exact": r["pos_exact"], "pos_acc": r["pos_acc"],
                       "relaxed": [x for s in r["steps"] for x in s["relaxed"]],
                       "step_acc": [(s["k"], s["acc"]) for s in r["steps"]],
                       "gen_ids": r["gen"], "response": text,
                       "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20),
                       "rep": rep_i, "draft_source_html": args.draft_source_html,
                       "draft_prompt_tokens": len(dprompt_by_id[p["row"]["id"]]) if dprompt_by_id.get(p["row"]["id"]) else None, "ranker_fold": rw_by_id.get(p["row"]["id"], (None, None))[0] if c == "ranker" else None, "verify_s": r["verify_s"], "draft_s": r["draft_s"], "draft_init_s": r["draft_init_s"],
                       "draft_steps": r["draft_steps"],
                       "peak_rss_mib": round(getattr(psutil.Process().memory_info(), "peak_wset", 0) / 2**20),
                       "rss_mib": round(psutil.Process().memory_info().rss / 2**20)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[r{rep_i} {i+1}/{len(prompts)}] {c:6s} {rec['id']} in={r['n_prompt']} out={n_out} fwd={r['n_fwd']} "
                      f"tok/fwd={rec['tokens_per_fwd']:.2f} ttft={r['ttft_s']:.2f}s total={r['total_s']:.1f}s "
                      f"draft={r['draft_s']:.2f}s verify={r['verify_s']:.1f}s", flush=True)
        for f in files.values():
            f.close()
    finally:
        try:
            GPU_BUSY.unlink()
        except FileNotFoundError:
            pass
    print("done", flush=True)


if __name__ == "__main__":
    main()
