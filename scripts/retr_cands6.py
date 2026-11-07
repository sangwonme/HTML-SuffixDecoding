"""Latency-oracle-gap prep: 6 exemplar candidates per pilot request + deployable features. CPU only.

Candidates (pilot 50, mid/pilot.jsonl order), in this role order:
  tfidf1, tfidf2, tfidf3 (poc_rnsd ranking), struct1 (beta=0.9, alpha=10), oracle1 (argmax omega_4 in
  TF-IDF top-300), short (highest TF-IDF-ranked candidate within TF-IDF top-20 whose GT flow token count is
  in the pool's lower quartile; fallback: shortest in top-20). Any duplicate is replaced by the next
  unused TF-IDF rank.
Writes runs/retrieval/pilot_cands6.json (id -> 6 train_pool ids),
       runs/retrieval/pilot_cands6_feats.jsonl (deployable features only: model inputs allowed),
       runs/retrieval/pilot_cands6_analysis.jsonl (omega_4 vs GT, GT length: analysis only, NOT model input).
Requires runs/retrieval/cache_kappa_phi.pkl (from scripts/retr_struct.py).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "4"

import json  # noqa: E402
import pickle  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from retr_struct import (CACHE, MAX_HTML, OUT, TOPC, limit, load_split, rss_check)  # noqa: E402
from genui import canon  # noqa: E402
from genui.retrieval import Retriever  # noqa: E402

ALPHA, BETA, SHORT_TOPN = 10.0, 0.9, 20
_STYLE = re.compile(r"<style[^>]*>(.*?)</style>", re.S | re.I)


def html_feats(html: str, tok) -> dict:
    ids = tok.encode(html, add_special_tokens=False).ids
    n = len(ids)
    m = _STYLE.search(html)
    css_tok = len(tok.encode(m.group(1), add_special_tokens=False).ids) if m else 0
    i = html.lower().find("<body")
    pre = len(tok.encode(html[:i], add_special_tokens=False).ids) if i >= 0 else 0
    from lxml import html as lhtml
    root = lhtml.document_fromstring(html)
    body = root.find("body")
    els = [e for e in body.iter() if isinstance(e.tag, str)][1:] if body is not None else []
    classes = {c for e in root.iter() if isinstance(e.tag, str) for c in (e.get("class") or "").split()}
    top = [c for c in body if isinstance(c.tag, str)] if body is not None else []
    if len(top) == 1:  # flow GT wraps everything in one div.col: count its children
        top = [c for c in top[0] if isinstance(c.tag, str)]
    return {"n_tokens": n, "css_frac": css_tok / max(n, 1), "pre_body_frac": pre / max(n, 1),
            "n_elems": len(els), "n_classes": len(classes), "n_top_blocks": len(top)}


def main():
    limit()
    from sklearn.linear_model import Ridge
    pool = load_split("train_pool")
    pilot = load_split("pilot")
    C = pickle.load(open(CACHE, "rb"))
    assert C["pool"]["id"] == [r["id"] for r in pool] and C["pilot"]["id"] == [r["id"] for r in pilot]
    tok = canon.tokenizer()
    ret = Retriever(pool, field="query_mid")
    idx_of = {r["id"]: i for i, r in enumerate(pool)}
    usable = np.array([bool(r["usable"]) for r in pool])
    cand = usable & np.array([len(r["gt_html_flow"]) <= MAX_HTML for r in pool])

    # ridge phi_hat (same settings as the chosen retr_struct model)
    tr = np.where(usable)[0]
    Y = C["pool"]["phi"][tr]
    keep = Y.std(0) > 0
    model = Ridge(alpha=ALPHA, solver="sparse_cg", tol=1e-4, max_iter=300).fit(ret.X[tr], Y[:, keep])
    Xp = ret.vec.transform([r["query_mid"] for r in pilot])
    P = np.zeros((len(pilot), Y.shape[1]), dtype=np.float32)
    P[:, keep] = model.predict(Xp)
    P /= np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    from sklearn.metrics.pairwise import linear_kernel
    S_t = linear_kernel(Xp, ret.X)
    S_phi = P @ C["pool"]["phi"].T
    S_struct = BETA * S_phi + (1 - BETA) * S_t

    # GT flow token counts of the pool (raw HTML, Gemma tokenizer) -> lower quartile threshold
    pool_ntok = np.array([len(e.ids) for e in tok.encode_batch([r["gt_html_flow"] for r in pool],
                                                               add_special_tokens=False)])
    q25 = float(np.percentile(pool_ntok[cand], 25))
    pool_sets = {}

    def pset(j):
        if j not in pool_sets:
            pool_sets[j] = np.unique(C["pool"]["hash"][j])
        return pool_sets[j]

    def om(i, j):
        return canon.omega(C["pilot"]["hash"][i], pset(j), int(C["pilot"]["ntok"][i]))

    def masked_top(s, k):
        return list(np.argsort(-np.where(cand, s, -np.inf), kind="stable")[:k])

    cands, feats, ana = {}, [], []
    n_short_fallback = 0
    for i, r in enumerate(pilot):
        tf_rank = [idx_of[e["id"]] for e in ret.topk(r["query_mid"], TOPC, max_html_chars=MAX_HTML)]
        struct1 = masked_top(S_struct[i], 1)[0]
        top300 = masked_top(S_t[i], TOPC)
        oms = [om(i, j) for j in top300]
        oracle1 = top300[int(np.argmax(oms))]
        short_pool = [j for j in tf_rank[:SHORT_TOPN] if pool_ntok[j] <= q25]
        if not short_pool:
            n_short_fallback += 1
            short_pool = sorted(tf_rank[:SHORT_TOPN], key=lambda j: pool_ntok[j])
        roles = [("tfidf1", tf_rank[0]), ("tfidf2", tf_rank[1]), ("tfidf3", tf_rank[2]),
                 ("struct1", struct1), ("oracle1", oracle1), ("short", None)]
        chosen, out_roles = [], []
        nxt = iter(tf_rank[3:])
        for role, j in roles:
            if role == "short":
                j = next((k for k in short_pool if k not in chosen), None)
                if j is None:
                    j = next(k for k in sorted(tf_rank[:SHORT_TOPN], key=lambda k: pool_ntok[k]) if k not in chosen)
            elif j in chosen:
                role, j = role + "->tfidf_next", next(k for k in nxt if k not in chosen)
            chosen.append(j)
            out_roles.append(role)
        assert len(set(chosen)) == 6
        cands[r["id"]] = [pool[j]["id"] for j in chosen]
        for rank, (role, j) in enumerate(zip(out_roles, chosen)):
            e = pool[j]
            f = {"pilot_id": r["id"], "cand_id": e["id"], "slot": rank, "role": role,
                 "tfidf_rank": tf_rank.index(j) + 1 if j in tf_rank else None,
                 "tfidf_sim": float(S_t[i, j]), "struct_sim": float(S_phi[i, j]),
                 "struct_score": float(S_struct[i, j]),
                 **html_feats(e["gt_html_flow"], tok),
                 "category_match": int(e.get("category") == r.get("category"))}
            feats.append(f)
            ana.append({"pilot_id": r["id"], "cand_id": e["id"], "slot": rank, "role": role,
                        "omega4_vs_gt": om(i, j), "gt_n_tokens": len(tok.encode(r["gt_html_flow"],
                                                                                add_special_tokens=False).ids),
                        "gt_kappa_tokens": int(C["pilot"]["ntok"][i])})
    (OUT / "pilot_cands6.json").write_text(json.dumps(cands, indent=1), encoding="utf-8")
    for name, rows in (("pilot_cands6_feats.jsonl", feats), ("pilot_cands6_analysis.jsonl", ana)):
        with open(OUT / name, "w", encoding="utf-8") as fh:
            for d in rows:
                fh.write(json.dumps(d) + "\n")
    roles = [f["role"] for f in feats]
    print(f"pilot {len(cands)}; q25 GT flow tokens = {q25:.0f}; short fallback {n_short_fallback}; "
          f"replaced dups {sum('->' in x for x in roles)}; rss {rss_check():.2f}GB")
    for k in ("n_tokens", "tfidf_sim", "struct_sim"):
        by = {}
        for f in feats:
            by.setdefault(f["role"].split("->")[0], []).append(f[k])
        print(k, {a: round(float(np.mean(b)), 3) for a, b in by.items()})


if __name__ == "__main__":
    main()
