"""Render-space (structure) retrieval of few-shot exemplars vs TF-IDF vs oracle. CPU only.

Idea: choose exemplar e to maximize E[omega_4(e, y) | x] (copy coverage in the render quotient, see
src/genui/canon.py) instead of query text similarity.
  s(x, e) = beta * cos(phi_hat(x), phi(e)) + (1 - beta) * cos_tfidf(x, e)
  phi_hat = ridge(TF-IDF(query_mid) -> phi), trained on train_pool (usable rows).
  beta tuned on dev 100 for mean omega_4(top-1, GT).
  oracle = argmax omega_4 over the top-300 TF-IDF candidates (uses GT: upper bound).

Pools: TF-IDF baseline = exactly scripts/poc_rnsd.py build_prompts (Retriever over all mid train_pool
rows, query_mid, max_html_chars=4000). struct/oracle candidates = usable rows with flow <= 4000 chars.

Usage: .venv/Scripts/python.exe -I scripts/retr_struct.py [--no-render]
Writes runs/retrieval/{pilot_top3_*.json, dev_top3_*.json, per_dev.jsonl, per_pilot.jsonl, summary.json}
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from genui import canon  # noqa: E402
from genui.retrieval import Retriever  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "runs" / "retrieval"
CACHE = OUT / "cache_kappa_phi.pkl"
RSS_LIMIT_GB = 6.0
N_PILOT = None  # all 50 pilot rows (coordinator request); poc_rnsd uses pilot[:n] of the same order
MAX_HTML = 4000
TOPC = 300
BETAS = [round(b, 2) for b in np.arange(0, 1.0001, 0.05)]
ALPHAS = [0.3, 1.0, 3.0, 10.0]
SEED = 20261107
METHODS = ("tfidf", "struct", "oracle", "mix", "struct_b06")


def limit():
    p = psutil.Process()
    try:
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001
        pass
    try:
        p.cpu_affinity(list(range(4)))
    except Exception:  # noqa: BLE001
        pass


def rss_check():
    g = psutil.Process().memory_info().rss / 2**30
    if g > RSS_LIMIT_GB:
        raise SystemExit(f"RSS guard: {g:.2f} GB > {RSS_LIMIT_GB}")
    return g


def jl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_split(name):
    main = {r["id"]: r for r in jl(PROC / f"{name}.jsonl")}
    out = []
    for m in jl(PROC / "mid" / f"{name}.jsonl"):
        if m["id"] in main:
            r = main[m["id"]]
            out.append(dict(m, gt_html_flow=r["gt_html_flow"], gt_html_abs=r["gt_html_abs"],
                            key_texts=r.get("key_texts")))
    return out


def features(rows, tag):
    t = time.time()
    ids = canon.kappa_ids([r["gt_html_flow"] for r in rows])
    ph = np.stack([canon.phi(r["gt_html_abs"]) for r in rows]).astype(np.float32)
    print(f"[{tag}] kappa/phi for {len(rows)} rows in {time.time() - t:.0f}s, rss {rss_check():.2f}GB", flush=True)
    return ids, ph


def boot_ci(x, B=2000, seed=SEED):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    limit()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------------------------------------------------------- data (same pool as poc_rnsd)
    pool = load_split("train_pool")
    dev = [r for r in load_split("dev") if r["usable"]]
    pilot = load_split("pilot")[:N_PILOT]
    print(f"pool {len(pool)} dev {len(dev)} pilot {len(pilot)}", flush=True)

    if CACHE.exists():
        C = pickle.load(open(CACHE, "rb"))
    else:
        C = {}
        for name, rows in (("dev", dev), ("pilot", pilot), ("pool", pool)):
            ids, ph = features(rows, name)
            C[name] = {"id": [r["id"] for r in rows], "ntok": np.array([len(i) for i in ids]),
                       "hash": [canon.ngram_hashes(i) for i in ids], "phi": ph}
        pickle.dump(C, open(CACHE, "wb"))
    assert C["pool"]["id"] == [r["id"] for r in pool] and C["dev"]["id"] == [r["id"] for r in dev]
    for name, rows in (("dev", dev), ("pilot", pilot)):  # diagnostic: body-only omega
        if "boff" not in C[name]:
            C[name]["boff"] = np.array([canon.body_offset(r["gt_html_flow"]) for r in rows])
    pool_sets = [np.unique(h) for h in C["pool"]["hash"]]
    print(f"features ready {time.time() - t0:.0f}s rss {rss_check():.2f}GB "
          f"mean kappa tokens pool {C['pool']['ntok'].mean():.0f}", flush=True)

    # ---------------------------------------------------------------- TF-IDF (poc_rnsd exact)
    ret = Retriever(pool, field="query_mid")
    idx_of = {r["id"]: i for i, r in enumerate(pool)}
    short = np.array([len(r["gt_html_flow"]) <= MAX_HTML for r in pool])
    usable = np.array([bool(r["usable"]) for r in pool])
    cand = short & usable
    print(f"candidates (usable & flow<={MAX_HTML}): {cand.sum()} / {len(pool)}", flush=True)

    def tfidf_sims(rows):
        from sklearn.metrics.pairwise import linear_kernel
        return linear_kernel(ret.vec.transform([r["query_mid"] for r in rows]), ret.X).astype(np.float32)

    # ---------------------------------------------------------------- phi_hat: ridge TF-IDF -> phi
    from sklearn.linear_model import Ridge
    tr = np.where(usable)[0]
    Y = C["pool"]["phi"][tr]
    keep = Y.std(0) > 0
    S_dev_t, S_pil_t = tfidf_sims(dev), tfidf_sims(pilot)
    Xd, Xp = ret.vec.transform([r["query_mid"] for r in dev]), ret.vec.transform([r["query_mid"] for r in pilot])

    def fit(alpha):
        t = time.time()
        m = Ridge(alpha=alpha, solver="sparse_cg", tol=1e-4, max_iter=300).fit(ret.X[tr], Y[:, keep])
        print(f"  ridge alpha={alpha} fit {time.time() - t:.0f}s", flush=True)
        return m

    def predict(m, X):
        P = np.zeros((X.shape[0], Y.shape[1]), dtype=np.float32)
        P[:, keep] = m.predict(X)
        return P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)

    phiP = C["pool"]["phi"]  # unit L2 rows

    # ---------------------------------------------------------------- omega helpers
    om_cache = {}

    def om(split, i, j):
        k = (split, i, j)
        if k not in om_cache:
            om_cache[k] = canon.omega(C[split]["hash"][i], pool_sets[j], int(C[split]["ntok"][i]))
        return om_cache[k]

    def om_union(split, i, js):
        u = np.unique(np.concatenate([pool_sets[j] for j in js]))
        return canon.omega(C[split]["hash"][i], u, int(C[split]["ntok"][i]))

    def topk_masked(scores, k=3):
        s = np.where(cand, scores, -np.inf)
        return list(np.argsort(-s, kind="stable")[:k])

    # alpha: pick by dev mean omega_4 of top-1 at the best beta (cheap; all on dev)
    best = None
    res_by_alpha = {}
    for alpha in ALPHAS:
        m = fit(alpha)
        Pd = predict(m, Xd)
        cos_phi_true = float(np.mean(np.sum(Pd * C["dev"]["phi"], 1)))
        Sd_phi = Pd @ phiP.T
        curve = {}
        for b in BETAS:
            S = b * Sd_phi + (1 - b) * S_dev_t
            curve[b] = float(np.mean([om("dev", i, topk_masked(S[i], 1)[0]) for i in range(len(dev))]))
        bb = max(curve, key=lambda b: (curve[b], -b))
        res_by_alpha[alpha] = {"cos_phihat_phi_dev": cos_phi_true, "beta_curve": curve, "best_beta": bb,
                               "best_omega": curve[bb]}
        print(f"alpha={alpha}: cos(phi_hat,phi)={cos_phi_true:.3f} best beta={bb} omega={curve[bb]:.4f} "
              f"(beta0 {curve[0.0]:.4f}, beta1 {curve[1.0]:.4f}) rss {rss_check():.2f}GB", flush=True)
        if best is None or curve[bb] > best[2]:
            best = (alpha, m, curve[bb], bb)
    alpha, model, _, beta = best
    print(f"chosen alpha={alpha} beta={beta}", flush=True)

    # ---------------------------------------------------------------- picks
    def picks(split, rows, S_t, X):
        P = predict(model, X)
        S_struct = beta * (P @ phiP.T) + (1 - beta) * S_t
        S_b06 = 0.6 * (P @ phiP.T) + 0.4 * S_t  # variant: lower beta (flat region of the beta curve)
        out = {"tfidf": [], "tfidf_cand": [], "struct": [], "oracle": [], "mix": [], "struct_b06": []}
        for i, r in enumerate(rows):
            out["tfidf"].append([idx_of[e["id"]] for e in ret.topk(r["query_mid"], 3, max_html_chars=MAX_HTML)])
            out["tfidf_cand"].append(topk_masked(S_t[i], 3))
            out["struct"].append(topk_masked(S_struct[i], 3))
            out["struct_b06"].append(topk_masked(S_b06[i], 3))
            # mix: struct top-1 (structure) + TF-IDF top-2 (content), dedup, refill from next TF-IDF ranks
            tf_rank = [idx_of[e["id"]] for e in ret.topk(r["query_mid"], 6, max_html_chars=MAX_HTML)]
            mx = [out["struct"][-1][0]]
            for j in tf_rank:
                if len(mx) == 3:
                    break
                if j not in mx:
                    mx.append(j)
            out["mix"].append(mx)
            top = topk_masked(S_t[i], TOPC)
            oms = np.array([om(split, i, j) for j in top])
            out["oracle"].append([top[k] for k in np.argsort(-oms, kind="stable")[:3]])
        return out

    PK = {"dev": picks("dev", dev, S_dev_t, Xd), "pilot": picks("pilot", pilot, S_pil_t, Xp)}
    pid = [r["id"] for r in pool]
    for split, rows in (("pilot", pilot), ("dev", dev)):
        for meth in METHODS:
            mp = {r["id"]: [pid[j] for j in PK[split][meth][i]] for i, r in enumerate(rows)}
            (OUT / f"{split}_top3_{meth}.json").write_text(json.dumps(mp, indent=1), encoding="utf-8")
    nonuse = sum(not usable[j] for L in PK["pilot"]["tfidf"] + PK["dev"]["tfidf"] for j in L)
    same = sum(a == b for a, b in zip(PK["dev"]["tfidf"], PK["dev"]["tfidf_cand"]))
    print(f"tfidf picks that are non-usable: {nonuse}; dev tfidf == tfidf restricted to candidates: {same}/{len(dev)}",
          flush=True)
    print(f"pilot files written {time.time() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- omega metrics
    per = {"dev": [], "pilot": []}
    for split, rows in (("dev", dev), ("pilot", pilot)):
        for i, r in enumerate(rows):
            d = {"id": r["id"], "ntok": int(C[split]["ntok"][i])}
            for meth in METHODS:
                js = PK[split][meth][i]
                d[f"{meth}_ids"] = [pid[j] for j in js]
                d[f"{meth}_om1"] = om(split, i, js[0])
                d[f"{meth}_om1body"] = canon.omega(C[split]["hash"][i], pool_sets[js[0]], int(C[split]["ntok"][i]),
                                                   start=int(C[split]["boff"][i]))
                d[f"{meth}_om3mean"] = float(np.mean([om(split, i, j) for j in js]))
                d[f"{meth}_om3union"] = om_union(split, i, js)
                d[f"{meth}_cosphi1"] = float(C["pool"]["phi"][js[0]] @ C[split]["phi"][i])
            per[split].append(d)

    # ---------------------------------------------------------------- copy baseline (render)
    if not args.no_render:
        from playwright.sync_api import sync_playwright
        from genui.checklist import ChecklistScorer
        from genui.metrics import Scorer
        with sync_playwright() as pw:
            kw = {}
            hs = sorted(Path(os.environ["LOCALAPPDATA"], "ms-playwright").glob(
                "chromium_headless_shell-*/chrome-headless-shell-win64/chrome-headless-shell.exe"))
            if hs and not Path(pw.chromium.executable_path).exists():
                kw["executable_path"] = str(hs[-1])  # installed browser build lags the playwright package
                print(f"  using {hs[-1]}", flush=True)
            br = pw.chromium.launch(headless=True, args=["--disable-gpu", "--renderer-process-limit=1"], **kw)
            cs, ms = ChecklistScorer(br), Scorer(br)
            for split, rows in (("dev", dev), ("pilot", pilot)):
                for i, r in enumerate(rows):
                    g = ms.analyze(r["gt_html_flow"])
                    for meth in METHODS:
                        html = pool[PK[split][meth][i][0]]["gt_html_flow"]
                        txt = "```html\n" + html + "\n```"
                        c = cs.score(txt, r["checklist"])
                        e = ms.score(txt, r, gt_analysis=g)
                        per[split][i][f"{meth}_check"] = c["check_score"]
                        per[split][i][f"{meth}_check_all"] = float(c["check_all"])
                        per[split][i][f"{meth}_elem_f1"] = e["elem_f1"]
                        per[split][i][f"{meth}_type_f1"] = e["type_f1"]
                    if i % 25 == 0:
                        print(f"  render {split} {i}/{len(rows)} rss {rss_check():.2f}GB", flush=True)
            br.close()

    for split in per:
        with open(OUT / f"per_{split}.jsonl", "w", encoding="utf-8") as f:
            for d in per[split]:
                f.write(json.dumps(d) + "\n")

    # ---------------------------------------------------------------- summary
    metrics = ["om1", "om1body", "om3mean", "om3union", "cosphi1", "check", "check_all", "elem_f1", "type_f1"]
    summ = {"alpha": alpha, "beta": beta, "by_alpha": res_by_alpha, "n_candidates": int(cand.sum()),
            "tfidf_nonusable_picks": int(nonuse), "elapsed_s": round(time.time() - t0)}
    for split in per:
        S = {}
        for meth in METHODS:
            for mt in metrics:
                k = f"{meth}_{mt}"
                if k not in per[split][0]:
                    continue
                x = [d[k] for d in per[split]]
                S[k] = boot_ci(x) if split == "dev" else [float(np.mean(x))]
        if split == "dev":
            for mt in metrics:
                for meth in METHODS[1:]:
                    k = f"{meth}_{mt}"
                    if k in per[split][0]:
                        S[f"diff_{meth}-tfidf_{mt}"] = boot_ci([d[k] - d[f"tfidf_{mt}"] for d in per[split]])
        summ[split] = S
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    for split in per:
        print(f"== {split}")
        for k, v in summ[split].items():
            print(f"  {k:32s} " + " ".join(f"{x:.4f}" for x in v))
    print(f"done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
