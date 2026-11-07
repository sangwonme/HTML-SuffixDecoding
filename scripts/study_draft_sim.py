"""Greedy-replay simulation of draft-position policies for prompt-lookup decoding (CPU only).

For a greedy target y (token ids) and the exact prompt context of its run, each policy proposes a draft of
gamma=16 tokens from ctx = prompt + y[:pos]; accepted = longest common prefix with y[pos:], plus one bonus
token (exactly what greedy verification yields for a greedy target).  n_fwd = 1 (prefill) + #verify steps.

Policies
  std     poc_rnsd.ngram_draft: n=3..1, most recent earlier occurrence
  n8      same rule, n=8..1
  k4      longest-n hits, best of the 4 most recent (tree-verification upper bound: K*gamma draft tokens/fwd)
  dom     longest-n hits (n=3..1), choose the hit whose structural annotation at the continuation point is most
          similar to the current one (lexical state, attr/CSS-property name, tag path suffix, depth, sibling
          index); no hit -> structure-only proposal (best-matching annotated position)
  oracle  any start position in ctx maximising accepted length (upper bound of position choice)

  .venv/Scripts/python.exe -I scripts/study_draft_sim.py
Output: runs/study/draft_sim.json, runs/study/figures/draft_sim.png
"""
from __future__ import annotations

import os, sys, pathlib, json, time
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[k] = os.environ.get("SIM_THREADS", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import psutil

from study_common import FIG, OUT, ROOT, check_rss, load_jsonl

GAMMA = 16
SYSTEM = ("You generate mobile app UI screens as HTML. Given a user's request, output ONE complete, "
          "self-contained HTML document for a 360x640 px mobile viewport: inline <style> CSS only, "
          "no JavaScript, no external resources (no CDN, fonts, or remote images). Use semantic elements "
          "(header, button, input, img, ul/li, nav, etc.) and include the texts the user mentions. "
          "Reply with only the HTML inside a single ```html code block, no explanation.")
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}


def guard(threads):
    p = psutil.Process()
    try:
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        p.cpu_affinity(list(range(threads)))
    except Exception:
        pass


# ------------------------------------------------------------------ causal HTML/CSS annotator
class Lexer:
    """Character-level causal lexer. state() = annotation of the NEXT character position."""

    def __init__(self):
        self.mode = "text"   # text | lt | tagname | endtag | attrs | attrname | attrval | comment | css | decl
        self.stack, self.sib = [], [0]
        self.buf, self.name, self.attr, self.q = "", "", "", ""
        self.css = "sel"     # sel | prop | val (inside <style>)
        self.cssprop = ""
        self.last_event = "none"   # start | end | text

    def state(self):
        m = self.mode
        if m in ("lt", "endtag"):
            lex = "boundary"
        elif m == "tagname":
            lex = "tagname"
        elif m in ("attrs", "attrname"):
            lex = "attrname"
        elif m == "attrval":
            lex = "cssval" if self.attr == "style" else "attrval"
        elif m == "comment":
            lex = "comment"
        elif m == "css":
            lex = {"sel": "csssel", "prop": "cssprop", "val": "cssval"}[self.css]
        else:
            lex = "text"
        key = self.cssprop if lex in ("cssval", "cssprop") else (self.attr if lex == "attrval" else "")
        return (lex, key, tuple(self.stack[-3:]), len(self.stack), self.sib[-1], self.last_event)

    def feed(self, ch):
        m = self.mode
        if m == "css":
            if self.buf.endswith("</style") or (ch == "<" and True):
                pass
            self.buf += ch
            if self.buf.lower().endswith("</style>"):
                self.mode = "text"; self.buf = ""; self._close("style"); return
            if self.css == "sel" and ch == "{":
                self.css = "prop"; self.cssprop = ""
            elif self.css == "prop":
                if ch == ":":
                    self.css = "val"
                elif ch == "}":
                    self.css = "sel"
                elif not ch.isspace() and ch not in ";":
                    self.cssprop += ch
            elif self.css == "val" and ch in ";}":
                self.css = "prop" if ch == ";" else "sel"; self.cssprop = ""
            if len(self.buf) > 16:
                self.buf = self.buf[-16:]
            return
        if m == "text":
            if ch == "<":
                self.mode = "lt"
            elif not ch.isspace():
                self.last_event = "text"
            return
        if m == "lt":
            if ch == "/":
                self.mode = "endtag"; self.name = ""
            elif ch == "!":
                self.mode = "comment"; self.buf = ""
            else:
                self.mode = "tagname"; self.name = ch.lower()
            return
        if m == "comment":
            self.buf += ch
            if self.buf.endswith("-->") or (self.buf.startswith("doctype") is False and ch == ">" and not self.buf.startswith("--")):
                self.mode = "text"
            return
        if m == "endtag":
            if ch == ">":
                self._close(self.name.strip().lower()); self.mode = "text"; self.last_event = "end"
            else:
                self.name += ch
            return
        if m == "tagname":
            if ch.isspace():
                self.mode = "attrs"
            elif ch in "/>":
                self._open(ch)
            else:
                self.name += ch.lower()
            return
        if m in ("attrs", "attrname"):
            if ch == ">" or ch == "/":
                if ch == ">":
                    self._open(">")
                return
            if ch == "=":
                self.mode = "attrval"; self.q = ""; self.buf = ""
                return
            if ch.isspace():
                self.mode = "attrs"
                return
            if m == "attrs":
                self.attr = ""
            self.mode = "attrname"; self.attr += ch.lower()
            return
        if m == "attrval":
            if not self.q and not self.buf and ch in "\"'":
                self.q = ch; return
            if (self.q and ch == self.q) or (not self.q and (ch.isspace() or ch == ">")):
                self.mode = "attrs"
                if ch == ">":
                    self._open(">")
                return
            self.buf += ch
            if self.attr == "style":
                if ch == ":":
                    pass
            return

    def _open(self, ch):
        name = self.name
        self.sib[-1] += 1
        self.last_event = "start"
        if name in VOID or ch == "/":
            self.mode = "text" if ch == ">" else "attrs"
            if ch == "/":
                self.mode = "attrs"
            return
        self.stack.append(name)
        self.sib.append(0)
        self.mode = "css" if name == "style" else "text"
        if name == "style":
            self.css, self.buf = "sel", ""

    def _close(self, name):
        if name in self.stack:
            while self.stack:
                n = self.stack.pop(); self.sib.pop()
                if n == name:
                    break


def annotate(tok_strs):
    """Annotation (state before token) for every token; None outside ```html ... ``` regions.
    Also returns region index per token (k-th html block, -1 outside)."""
    ann, reg = [], []
    text = ""
    inhtml, lx, k = False, None, -1
    pend = ""
    for s in tok_strs:
        ann.append(lx.state() if inhtml and lx else None)
        reg.append(k if inhtml else -1)
        for ch in s:
            text += ch
            if not inhtml:
                if text.endswith("```html\n") or text.endswith("```html"):
                    pend = "open"
                if pend == "open" and ch == "\n" and text.rstrip("\n").endswith("```html"):
                    inhtml, lx, k, pend = True, Lexer(), k + 1, ""
            else:
                if text.endswith("```"):
                    inhtml = False
                    continue
                lx.feed(ch)
        if len(text) > 64:
            text = text[-64:]
    return ann, reg


def sim_score(a, b):
    if a is None or b is None:
        return -1.0
    s = 4.0 * (a[0] == b[0]) + 2.0 * (a[1] == b[1] and a[1] != "")
    pa, pb = a[2], b[2]
    c = 0
    while c < min(len(pa), len(pb)) and pa[len(pa) - 1 - c] == pb[len(pb) - 1 - c]:
        c += 1
    s += c + 1.0 * (a[3] == b[3]) + 1.0 * (a[4] == b[4]) + 0.5 * (a[5] == b[5])
    return s


# ------------------------------------------------------------------ policies
class Index:
    """n-gram -> sorted list of continuation positions (s+n) for ctx (grows with ctx)."""

    def __init__(self, nmax):
        self.nmax = nmax
        self.d = {}
        self.L = 0

    def extend(self, ctx, L):
        # add n-grams whose continuation index c = j+n satisfies c <= L-1 (window ends before last token)
        for c in range(max(1, self.L), L):
            for n in range(1, self.nmax + 1):
                j = c - n
                if j < 0:
                    break
                self.d.setdefault(tuple(ctx[j:c]), []).append(c)
        self.L = L

    def hits(self, ctx, L, n):
        key = tuple(ctx[L - n:L])
        h = self.d.get(key, [])
        return [c for c in h if c <= L - 1]  # continuation strictly before end (match ends at <= L-2)


def accept_len(ctx, s, L, y, pos, gamma):
    a = 0
    while a < gamma and s + a < L and pos + a < len(y) and ctx[s + a] == y[pos + a]:
        a += 1
    return a


def oracle_best(ctx, L, y, pos, gamma):
    cand = np.nonzero(ctx[:L] == y[pos])[0]
    best, bs = 0, None
    if len(cand) == 0:
        return 0, None
    a = 1
    alive = cand
    while a < gamma and pos + a < len(y):
        nxt = alive[(alive + a < L)]
        nxt = nxt[ctx[nxt + a] == y[pos + a]] if len(nxt) else nxt
        if len(nxt) == 0:
            break
        alive = nxt
        a += 1
    return a, int(alive[-1])


def run_policy(policy, prompt, y, ann_ctx, reg_ctx, max_new=2048, gamma=GAMMA, nmax=3, K=4, record=False):
    T = len(y)
    ctx = np.concatenate([prompt, y]).astype(np.int64)
    P = len(prompt)
    idx = Index(8 if policy == "n8" else 3) if policy != "oracle" else None
    pos = 1  # first token comes from prefill
    n_fwd = 1
    steps = []
    while pos < T:
        L = P + pos
        g = min(gamma, max_new - pos - 1)
        src, a = None, 0
        if g <= 0:
            d_ok = False
        if policy == "oracle":
            a, src = oracle_best(ctx, L, y, pos, g)
            if a == 0:
                src = None
        else:
            idx.extend(ctx, L)
            nm = 8 if policy == "n8" else 3
            hit_n, hits = 0, []
            for n in range(nm, 0, -1):
                if L <= n:
                    continue
                h = idx.hits(ctx, L, n)
                if h:
                    hit_n, hits = n, h
                    break
            if policy in ("std", "n8"):
                if hits:
                    src = hits[-1]
            elif policy == "k4oracle":
                best = -1
                for c in hits[-K:][::-1]:
                    al = accept_len(ctx, c, L, y, pos, g)
                    if al > best:
                        best, src = al, c
            elif policy == "k4vote":
                cands = hits[-K:][::-1]   # most recent first; ties -> most recent
                if cands:
                    draft, alive = [], list(cands)
                    for j in range(g):
                        nxt = [(c, int(ctx[c + j])) for c in alive if c + j < L]
                        if not nxt:
                            break
                        cnt = {}
                        for _, tkn in nxt:
                            cnt[tkn] = cnt.get(tkn, 0) + 1
                        top = max(cnt.values())
                        tkn = next(t_ for _, t_ in nxt if cnt[t_] == top)
                        draft.append(tkn)
                        alive = [c for c, t_ in nxt if t_ == tkn]
                    a = 0
                    while a < len(draft) and pos + a < len(y) and draft[a] == y[pos + a]:
                        a += 1
                    src = cands[0]
                    n_fwd += 1
                    if record:
                        steps.append({"pos": pos, "k": len(draft), "a": int(a), "src": int(src), "src_reg": int(reg_ctx[src])})
                    pos += a + 1
                    continue
            elif policy == "dom":
                cur = ann_ctx[L] if L < len(ann_ctx) else None
                if hits:
                    best = None
                    for c in hits[-300:][::-1]:
                        sc = sim_score(ann_ctx[c], cur)
                        if best is None or sc > best:
                            best, src = sc, c
                elif cur is not None:
                    best = None
                    for c in range(L - 1, 0, -1):
                        if ann_ctx[c] is None:
                            continue
                        sc = sim_score(ann_ctx[c], cur)
                        if best is None or sc > best:
                            best, src = sc, c
            if src is not None and policy != "k4vote":
                a = accept_len(ctx, src, L, y, pos, g)
        k = 0 if src is None else min(g, L - src)
        n_fwd += 1
        if record:
            steps.append({"pos": pos, "k": int(k), "a": int(a), "src": None if src is None else int(src),
                          "src_reg": None if src is None else int(reg_ctx[src])})
        pos += a + 1
    return n_fwd, steps


# ------------------------------------------------------------------ data
def load_targets(tok):
    pool_html = {r["id"]: r["gt_html_flow"] for r in load_jsonl(ROOT / "data" / "processed" / "train_pool.jsonl")}
    mid = ROOT / "data" / "processed" / "mid"
    mid_pool = {r["id"]: r for r in load_jsonl(mid / "train_pool.jsonl")}
    mid_pilot = {r["id"]: r for r in load_jsonl(mid / "pilot.jsonl")}
    inter = json.load(open(ROOT / "runs" / "retrieval" / "pilot_exemplar_html_interleaved.json", encoding="utf-8"))
    out = []
    for tag, fn, ovr in [("C1", "E2B_greedy.jsonl", None), ("C0", "E2B_greedy_C0.jsonl", None),
                         ("C4g", "E2B_greedy_C4.jsonl", None), ("C7g", "E2B_greedy_C7.jsonl", inter),
                         ("C2pld", "E2B_pld.jsonl", None)]:
        p = ROOT / "runs" / "poc" / fn
        if not p.exists():
            continue
        for r in load_jsonl(p):
            if "id" not in r or "gen_ids" not in r:
                continue
            msgs = [{"role": "system", "content": SYSTEM}]
            for e in r["example_ids"]:
                h = (ovr or {}).get(e, pool_html[e])
                msgs.append({"role": "user", "content": mid_pool[e]["query_mid"]})
                msgs.append({"role": "assistant", "content": "```html\n" + h + "\n```"})
            msgs.append({"role": "user", "content": mid_pilot[r["id"]]["query_mid"]})
            ids = list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])
            out.append({"set": tag, "id": r["id"], "prompt": list(ids), "y": list(r["gen_ids"]),
                        "prompt_ok": len(ids) == r["prompt_tokens"], "log_fwd": r.get("n_fwd"),
                        "max_new": 2048})
    # runs/gemma rich_fs greedy (bench prompts; target = re-tokenized response, approximate)
    from genui.retrieval import Retriever
    proc_pool = load_jsonl(ROOT / "data" / "processed" / "train_pool.jsonl")
    ret = Retriever(proc_pool)
    pilot = {r["id"]: r for r in load_jsonl(ROOT / "data" / "processed" / "pilot.jsonl")}
    for m in ("E2B", "E4B", "12B"):
        for r in load_jsonl(ROOT / "runs" / "gemma" / f"{m}_nf4_rich_fs_pilot.jsonl"):
            if "id" not in r:
                continue
            row = pilot[r["id"]]
            msgs = [{"role": "system", "content": SYSTEM}]
            for ex in ret.topk(row["query_rich"], 3, max_html_chars=4000):
                msgs.append({"role": "user", "content": ex["query_rich"]})
                msgs.append({"role": "assistant", "content": "```html\n" + ex["gt_html_flow"] + "\n```"})
            msgs.append({"role": "user", "content": row["query_rich"]})
            ids = list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])
            y = tok(r["response"], add_special_tokens=False)["input_ids"]
            out.append({"set": f"gemma_{m}_fs", "id": r["id"], "prompt": list(ids), "y": list(y),
                        "prompt_ok": len(ids) == r["prompt_tokens"], "log_fwd": None, "max_new": 2048})
    return out


def lex_of_token(ann):
    if ann is None:
        return "other"
    return ann[0]


def boot_ci(vals, B=2000, seed=0):
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    guard(int(os.environ.get("SIM_THREADS", "2")))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "gemma-4-E2B-it-qat-q4_0-unquantized"))
    targets = load_targets(tok)
    print("targets:", len(targets), "prompt_ok:", sum(t["prompt_ok"] for t in targets), flush=True)
    policies = ["std", "n8", "k4vote", "dom", "k4oracle", "oracle"]
    rows = []
    t0 = time.time()
    for ti, t in enumerate(targets):
        prompt = np.array(t["prompt"], np.int64)
        y = np.array(t["y"], np.int64)
        strs = [tok.decode([i]) for i in list(prompt) + list(y)]
        ann, reg = annotate(strs)
        P = len(prompt)
        reg = [(-2 if i >= P else r) for i, r in enumerate(reg)]  # -2 = own output
        ann.append(None)
        row = {"set": t["set"], "id": t["id"], "T": len(y), "prompt_ok": t["prompt_ok"], "log_fwd": t["log_fwd"]}
        rec = {}
        for pol in policies:
            nf, st = run_policy(pol, prompt, y, ann, reg, max_new=t["max_new"], record=pol in ("std", "oracle", "dom"))
            row[f"fwd_{pol}"] = nf
            if st:
                rec[pol] = st
        # ---- diagnostics on std trajectory
        ctx = np.concatenate([prompt, y])
        std = rec["std"]
        fail = [s for s in std if s["a"] <= 1]
        rec_or = []
        ctx_cat = []
        for s in fail:
            L = P + s["pos"]
            g = min(GAMMA, t["max_new"] - s["pos"] - 1)
            ao, _ = oracle_best(ctx, L, y, s["pos"], g)
            rec_or.append(ao >= 2)
            if s["a"] == 0:
                a_ = ann[P + s["pos"]]
                cat = lex_of_token(a_)
                tokstr = strs[P + s["pos"]]
                if cat in ("boundary",) or "<" in tokstr or ">" in tokstr:
                    cat = "boundary"
                ctx_cat.append((cat, ao >= 2))
        def cat_of(i):
            a_ = ann[P + i]; ts = strs[P + i]
            c = lex_of_token(a_)
            if "<" in ts or ">" in ts or c == "boundary":
                return "boundary"
            return {"attrname": "attr", "attrval": "attr", "cssprop": "css_prop", "csssel": "css_prop",
                    "cssval": "css_val", "tagname": "tagname", "text": "text"}.get(c, "other")
        acc_ctx = {}
        for pol in ("std", "dom"):
            for s_ in rec[pol]:
                for i in range(s_["pos"], min(len(y), s_["pos"] + s_["a"])):
                    c = cat_of(i)
                    acc_ctx.setdefault(c, {"std": 0, "dom": 0})[pol] += 1
        row["acc_by_ctx"] = acc_ctx
        row["fail_steps"] = len(fail)
        row["fail_oracle_ge2"] = int(sum(rec_or))
        row["single_tok_ctx"] = ctx_cat
        # resync: after a rejected step (k>0, a<k), forwards/tokens until a step with a>=2
        rs_f, rs_t, same = [], [], []
        i = 0
        while i < len(std):
            s = std[i]
            if s["k"] > 0 and s["a"] < s["k"]:
                j, nf, nt = i + 1, 0, 0
                while j < len(std) and std[j]["a"] < 2:
                    nf += 1; nt += std[j]["a"] + 1; j += 1
                if j < len(std):
                    rs_f.append(nf + 1); rs_t.append(nt)
                    prev_end = (s["src"] or 0) + s["a"] + 1
                    q = std[j]
                    same.append(q["src_reg"] == s["src_reg"] and q["src"] is not None and 0 <= q["src"] - prev_end <= 64)
                i = j
            else:
                i += 1
        row["resync_fwd"] = rs_f
        row["resync_tok"] = rs_t
        row["resync_same_block"] = same
        rows.append(row)
        if ti % 10 == 0:
            print(ti, t["set"], t["id"], {p: row[f"fwd_{p}"] for p in policies}, "log", t["log_fwd"],
                  f"{time.time() - t0:.0f}s rss={check_rss():.2f}", flush=True)
    summarize(rows, policies)


def summarize(rows, policies):
    res = {"gamma": GAMMA, "policies": policies, "rows": rows, "sets": {}}
    # validation: std on PLD run must reproduce its logged forward count
    val = [r for r in rows if r["set"] == "C2pld"]
    res["validation_note"] = "ideal greedy-target acceptance analysis; std policy replayed on the PLD run's own output"
    res["validation"] = {"n": len(val), "exact_match": sum(r["fwd_std"] == r["log_fwd"] for r in val),
                         "pairs": [(r["id"], r["fwd_std"], r["log_fwd"]) for r in val],
                         "prompt_len_match": sum(r["prompt_ok"] for r in rows), "n_rows": len(rows)}
    groups = {"poc_greedy(C1,C0,C4g,C7g)": [r for r in rows if r["set"] in ("C1", "C0", "C4g", "C7g")],
              "gemma_rich_fs(approx)": [r for r in rows if r["set"].startswith("gemma")]}
    for s in sorted(set(r["set"] for r in rows)):
        groups[s] = [r for r in rows if r["set"] == s]
    for g, rs in groups.items():
        if not rs:
            continue
        d = {"n": len(rs)}
        for p in policies:
            d[f"fwd_{p}"] = float(np.mean([r[f"fwd_{p}"] for r in rs]))
            d[f"tokfwd_{p}"] = float(np.sum([r["T"] for r in rs]) / np.sum([r[f"fwd_{p}"] for r in rs]))
        for p in ("n8", "k4vote", "dom", "k4oracle", "oracle"):
            red = [1 - r[f"fwd_{p}"] / r["fwd_std"] for r in rs]
            d[f"reduction_{p}"] = float(np.mean(red))
            d[f"reduction_{p}_ci"] = boot_ci(red)
        for p in ("n8", "k4vote", "dom", "k4oracle"):
            gap = [(r["fwd_std"] - r[f"fwd_{p}"]) / max(1, r["fwd_std"] - r["fwd_oracle"]) for r in rs]
            d[f"gapclosed_{p}"] = float(np.mean(gap))
            d[f"gapclosed_{p}_ci"] = boot_ci(gap)
        for p in ("std", "n8", "k4vote"):
            dif = [r[f"fwd_{p}"] - r["fwd_dom"] for r in rs]   # >0: DOM uses fewer forwards
            d[f"paired_{p}_minus_dom"] = {"mean": float(np.mean(dif)), "median": float(np.median(dif)), "ci95": boot_ci(dif)}
        best_exec = min(("n8", "k4vote"), key=lambda p: np.mean([r[f"fwd_{p}"] for r in rs]))
        d["best_exec_general"] = best_exec
        dif = [r[f"fwd_{best_exec}"] - r["fwd_dom"] for r in rs]
        d["Q2_bestexec_minus_dom"] = {"mean": float(np.mean(dif)), "median": float(np.median(dif)), "ci95": boot_ci(dif)}
        dif = [r["fwd_dom"] - r["fwd_oracle"] for r in rs]
        d["Q3_dom_minus_oracle"] = {"mean": float(np.mean(dif)), "median": float(np.median(dif)), "ci95": boot_ci(dif)}
        rem = [(r["fwd_dom"] - r["fwd_oracle"]) / max(1, r["fwd_std"] - r["fwd_oracle"]) for r in rs]
        d["Q3_gap_remaining_share"] = {"mean": float(np.mean(rem)), "median": float(np.median(rem)), "ci95": boot_ci(rem)}
        agg = {}
        for r in rs:
            for c, v in r["acc_by_ctx"].items():
                agg.setdefault(c, {"std": 0, "dom": 0}); agg[c]["std"] += v["std"]; agg[c]["dom"] += v["dom"]
        d["Q1_extra_accepted_by_ctx"] = {c: {"std": v["std"], "dom": v["dom"], "extra": v["dom"] - v["std"]} for c, v in agg.items()}
        d["verdict_structure_advantage"] = bool(d["Q2_bestexec_minus_dom"]["ci95"][0] > 0)
        d["verdict_learnable_gap"] = bool(d["Q3_gap_remaining_share"]["median"] >= 0.30)
        d["oracle_speedup_vs_std"] = float(np.mean([r["fwd_std"] / r["fwd_oracle"] for r in rs]))
        fs = sum(r["fail_steps"] for r in rs)
        d["fail_steps_total"] = fs
        d["fail_oracle_recoverable_share"] = sum(r["fail_oracle_ge2"] for r in rs) / max(1, fs)
        allf = [x for r in rs for x in r["resync_fwd"]]
        allt = [x for r in rs for x in r["resync_tok"]]
        alls = [x for r in rs for x in r["resync_same_block"]]
        d["resync_events"] = len(allf)
        d["resync_fwd_mean"] = float(np.mean(allf)) if allf else None
        d["resync_tok_mean"] = float(np.mean(allt)) if allt else None
        d["resync_same_block_share"] = float(np.mean(alls)) if alls else None
        cats = {}
        for r in rs:
            for c, ok in r["single_tok_ctx"]:
                cats.setdefault(c, [0, 0]); cats[c][0] += 1; cats[c][1] += ok
        tot = sum(v[0] for v in cats.values())
        d["single_tok_ctx"] = {c: {"share": v[0] / max(1, tot), "oracle_recoverable": v[1] / max(1, v[0])}
                               for c, v in sorted(cats.items(), key=lambda x: -x[1][0])}
        res["sets"][g] = d
        print(g, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()
                  if k.startswith(("fwd_", "gapclosed_", "Q2", "Q3", "verdict", "best_exec")) and not k.endswith("_ci")}, flush=True)
    print("validation:", res["validation"]["exact_match"], "/", res["validation"]["n"], res["validation"]["pairs"][:5])
    for r in rows:
        r.pop("single_tok_ctx", None)
    (OUT / "draft_sim.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    fig(res)


def fig(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    PAL = {"std": "#2a78d6", "n8": "#eb6834", "k4vote": "#eda100", "dom": "#1baf7a", "k4oracle": "#e87ba4", "oracle": "#b9b8b2"}
    NAME = {"std": "std n=3..1 (exec)", "n8": "n=8..1 (exec)", "k4vote": "K=4 vote (exec)", "dom": "DOM-aware (exec)",
            "k4oracle": "K=4 best-of (oracle)", "oracle": "any position (oracle)"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    sets = [s for s in ("C1", "C4g", "C7g", "C0", "gemma_E2B_fs", "gemma_E4B_fs", "gemma_12B_fs") if s in res["sets"]]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.13
    for j, p in enumerate(res["policies"]):
        vals = [res["sets"][s][f"fwd_{p}"] for s in sets]
        ax.bar(np.arange(len(sets)) + (j - 2.5) * w, vals, w * 0.9, color=PAL[p], label=NAME[p],
               hatch="//" if "oracle" in p else None, edgecolor="#fcfcfb")
    ax.set_xticks(range(len(sets)), sets)
    ax.set_ylabel("forwards per sample (mean)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.set_title("draft-position policies, greedy replay (gamma=16)", loc="left", fontsize=10, pad=24)
    fig.tight_layout(); fig.savefig(FIG / "draft_sim.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
