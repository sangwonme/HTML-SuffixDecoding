# Render-space retrieval of few-shot exemplars (2026-11-07)

Code: `src/genui/canon.py` (κ, ω, φ), `scripts/retr_struct.py` (all steps; ~75 s on CPU, 4 threads, BelowNormal, peak RSS 0.8 GB).
Run: `.venv/Scripts/python.exe -I scripts/retr_struct.py [--no-render]`. Outputs in `runs/retrieval/`.

## Claim being tested
One quantity, the overlap ω between the exemplar's UI and the target UI in the render quotient (surface differences removed), should control both:
- **latency**: under copy/speculative decoding from the exemplar, acceptance α grows with ω, and steps ≈ |y| / E[L_acc] with E[L_acc] = (1−α^(γ+1))/(1−α);
- **accuracy**: the exemplar acts as a structural prior.

So retrieval should maximize E[ω(e, y) | x], learned on train_pool where y is known, instead of query text similarity.

## Definitions (as implemented)
- **κ(y)**, the canonical form, is built from `gt_html_flow` (the HTML that goes into the few-shot prompt):
  - Parse with html5lib if installed. It is not installed here, so `lxml.html` is used.
  - Drop comments, whitespace-only text and `data-*` attributes (this includes `data-label`). Sort attributes.
  - Rename classes c0, c1, … by order of first appearance. The renaming is applied both in `class` attributes and in `<style>` selectors; declarations are left unchanged.
  - Serialize compactly with no doctype, then tokenize with the Gemma tokenizer (`models/gemma-4-E2B-it-qat-q4_0-unquantized/tokenizer.json`, CPU).
  - The train_pool mean is 490 tokens.
- **ω₄(e, y)**: the fraction of κ(y) tokens covered by some 4-gram that also occurs in κ(e). 4-grams are compared by uint64 rolling hash.
  - `om1body` is a diagnostic variant that counts only tokens from `<body>` on, which excludes the head/style boilerplate.
- **φ(y)**: built from `gt_html_abs` with no rendering. Absolute y is the sum of the `top` offsets of labeled ancestors. Two blocks are concatenated:
  - a (category × band) histogram, using 25 labels + OTHER. Bands are by element center: top <128px, middle, bottom >512px.
  - parent→child category bigrams. The parent is the nearest labeled ancestor, or ROOT.
  - Each block is L1-normalized and the concatenation is L2-normalized (802 dims).
- **φ̂(x)**: Ridge regression (sparse_cg) from TF-IDF(query_mid) to φ. It uses the exact vectorizer of `genui.retrieval.Retriever` (1–2-grams, min_df 2, sublinear) and is trained on the 14,516 usable train_pool rows.
  - α ∈ {0.3, 1, 3, 10} was tried and **α=10** was chosen. Dev mean cos(φ̂, φ) is 0.76.
- **Score**: s(x,e) = β·cos(φ̂(x), φ(e)) + (1−β)·cos_tfidf(x,e).
  - β was swept over 0…1 in steps of 0.05 on dev 100, maximizing mean ω₄ of the top-1 exemplar. **β = 0.9** was chosen; the curve is flat over 0.55–0.9 at 0.833–0.835.
- **Candidates**: usable rows with `len(gt_html_flow) ≤ 4000` (14,181 rows). This matches poc_rnsd's `max_html_chars=4000`.
- **TF-IDF baseline**: exactly the `build_prompts` code path of `poc_rnsd.py`.
  - An independent re-run of that code reproduces `pilot_top3_tfidf.json` **50/50**.
  - None of its picks are non-usable rows, and on dev its picks equal the candidate-restricted ranking 100/100.
- **Oracle**: for each target, the top 3 by ω₄ among the top-300 TF-IDF candidates. It uses the GT, so it is an upper bound and is **not a deployable method**.
- **Copy baseline**: the top-1 exemplar's GT flow HTML is used directly as the answer. It is scored with the target's query_mid checklist (`genui.checklist`) and with element F1 against the target's GT flow (`genui.metrics`).

## Results: dev 100 (usable), mean [bootstrap 95% CI, 2000 resamples]
| | TF-IDF (current) | struct (ours, β=0.9) | oracle (upper bound) |
|---|---|---|---|
| ω₄ top-1 | 0.807 [0.779, 0.834] | **0.836** [0.812, 0.857] | 0.886 [0.863, 0.907] |
| ω₄ top-1, body only | 0.633 [0.592, 0.677] | **0.697** [0.664, 0.731] | 0.794 [0.762, 0.826] |
| ω₄ top-3 mean of 3 | 0.796 [0.772, 0.819] | **0.830** [0.806, 0.852] | 0.878 [0.855, 0.898] |
| ω₄ top-3 union coverage | 0.869 [0.846, 0.889] | 0.866 [0.843, 0.886] | 0.904 [0.881, 0.923] |
| cos(φ(e₁), φ(y)) | 0.563 | 0.741 | 0.714 |
| copy: checklist score | **0.369** [0.306, 0.435] | 0.287 [0.232, 0.345] | 0.322 [0.253, 0.387] |
| copy: checklist all | **0.13** | 0.05 | 0.12 |
| copy: element F1 | 0.584 [0.538, 0.630] | **0.635** [0.600, 0.670] | 0.649 [0.602, 0.694] |
| copy: type F1 | 0.766 | **0.838** | 0.853 |

Paired differences, struct − TF-IDF, with 95% CI:

| Metric | Difference |
|---|---|
| ω₄ top-1 | +0.029 [+0.011, +0.048] |
| ω₄ top-1, body only | +0.064 [+0.032, +0.097] |
| ω₄ top-3 mean | +0.034 [+0.024, +0.045] |
| ω₄ top-3 union | −0.002 [−0.011, +0.006] |
| element F1 | +0.052 [+0.008, +0.097] |
| type F1 | +0.071 [+0.035, +0.109] |
| **checklist** | **−0.083 [−0.138, −0.029]** |

## Pilot (held out from tuning), means
| | ω₄ top-1 | body ω₄ | top-3 mean | top-3 union | copy checklist | copy elem F1 |
|---|---|---|---|---|---|---|
| pilot 50: TF-IDF / struct / oracle | 0.751 / 0.795 / 0.846 | 0.563 / 0.646 / 0.745 | 0.751 / 0.792 / 0.840 | 0.821 / 0.826 / 0.866 | 0.372 / 0.262 / 0.338 | 0.543 / 0.622 / 0.685 |
| pilot first 10 (GPU run) | 0.817 / 0.844 / 0.901 | 0.636 / 0.686 / 0.799 | 0.801 / 0.840 / 0.892 | 0.878 / 0.871 / 0.916 | 0.442 / 0.242 / 0.342 | 0.512 / 0.639 / 0.671 |

## Reading
- **Structure retrieval raises copy coverage and structural accuracy.** Top-1 ω₄ rises on dev, and the body-only gain is about twice as large. The gain holds on the untouched pilot 50 (+0.044 top-1, +0.083 body). Element and type F1 of the copied exemplar also rise.
  - Struct closes about 36% of the TF-IDF→oracle gap on top-1 ω₄ and about 40% on body ω₄.
- **It loses on the checklist, which is content.** Checklist items are keywords such as titles and button labels. TF-IDF on the query matches those words, while the struct score at β=0.9 mostly ignores them.
  - Note that the ω-oracle also loses checklist points to TF-IDF (−0.048, CI touches 0). Maximizing ω trades content words for structure.
  - In real few-shot use the model rewrites the text from the query, so the copy-baseline checklist is a pessimistic proxy. Still, the trade-off is real and should be checked on Gemma outputs.
- **The union of the top-3 does not improve.** TF-IDF's three exemplars are more diverse, so their union already covers as much as struct's three. For speculative decoding with all 3 exemplars in context, the per-exemplar gain may not translate into more accepted drafts. This needs the GPU measurement.
- The β curve is flat between 0.55 and 0.9. A lower β such as 0.6 keeps almost all of the ω gain (0.834 vs 0.835) and may give up less checklist; this has not been measured yet.

## Added conditions (coordinator request, 21:40)
- **mix**: struct top-1 (β=0.9), followed by TF-IDF top-2 for content. A duplicate is replaced by the next TF-IDF rank (the poc_rnsd ranking). It pairs one structural exemplar with two content exemplars.
  - File: `runs/retrieval/pilot_top3_mix.json` (pilot 50, same format).
  - Its top-1 is the struct top-1, so the top-1 ω and the copy baseline equal struct's.
- **struct β=0.6**: the same score with β=0.6, chosen from the flat part of the β curve. This value was picked after seeing the dev curve.
  - Files: `*_top3_struct_b06.json`, kept for analysis only.

Dev 100 results, mean [95% CI]:

| | ω₄ top-1 | body ω₄ | top-3 mean | top-3 union | copy checklist | copy elem F1 |
|---|---|---|---|---|---|---|
| TF-IDF | 0.807 | 0.633 | 0.796 | 0.869 [0.846, 0.889] | 0.369 [0.306, 0.435] | 0.584 |
| struct β=0.9 | 0.836 | 0.697 | 0.830 | 0.866 | 0.287 | 0.635 |
| **mix** | 0.836 | 0.697 | 0.808 [0.785, 0.831] | **0.877** [0.854, 0.898] | 0.287 (= struct) | 0.635 (= struct) |
| **struct β=0.6** | 0.834 [0.809, 0.858] | 0.690 | 0.825 | 0.874 [0.850, 0.895] | **0.368** [0.307, 0.433] | **0.661** [0.626, 0.698] |
| oracle | 0.886 | 0.794 | 0.878 | 0.904 | 0.322 | 0.649 |

Paired differences against TF-IDF, with 95% CI:

| Metric | mix − TF-IDF | β=0.6 − TF-IDF |
|---|---|---|
| top-3 union | +0.009 [+0.004, +0.013] | +0.005 [−0.002, +0.011] |
| top-3 mean | +0.012 [+0.007, +0.017] | +0.029 [+0.019, +0.040] |
| ω₄ top-1 | struct value: +0.029 | +0.027 [+0.010, +0.046] |
| checklist | struct value: −0.083 | −0.002 [−0.052, +0.047] |
| element F1 | struct value: +0.052 | +0.077 [+0.036, +0.121] |

Pilot means, TF-IDF / struct / mix / β=0.6:

| Pilot set | Metric | TF-IDF | struct | mix | β=0.6 |
|---|---|---|---|---|---|
| pilot 50 | top-3 union | 0.821 | 0.826 | 0.832 | 0.831 |
| pilot 50 | top-3 mean | 0.751 | 0.792 | 0.767 | 0.787 |
| pilot 50 | ω₄ top-1 | 0.751 | 0.795 | 0.795 | 0.787 |
| pilot 50 | copy checklist | 0.372 | 0.262 | 0.262 | 0.317 |
| pilot 50 | copy elem F1 | 0.543 | 0.622 | 0.622 | 0.630 |
| pilot first 10 | top-3 union | 0.878 | 0.871 | 0.890 | 0.875 |
| pilot first 10 | copy checklist | 0.442 | 0.242 | 0.242 | 0.325 |

What this shows:
- **mix is the only condition whose top-3 union is significantly above TF-IDF**, by a small margin (+0.009). It also keeps struct's top-1 gain.
- **On dev, β=0.6 removes the checklist loss** (−0.002) while keeping most of the ω gain and giving the largest element F1 gain.
  - On pilot 50, which was not used for tuning, it still loses 0.055 checklist (no CI at n=50). Because β=0.6 was chosen after looking at dev, its dev numbers are optimistic.

## Limitations
- β and α were tuned on the same dev 100 that is reported, so the dev struct numbers are slightly optimistic. Pilot 50 is the held-out check.
- html5lib is not installed, so the lxml.html fallback is used. Both are lenient and the GT is well-formed, so the effect should be small.
- All rows share head/style boilerplate, which keeps the full-document ω₄ around 0.8. Body-only ω₄ is the more discriminative number.
- φ uses RICO labels and abs positions. φ̂ is linear in query TF-IDF (cos 0.76 to the true φ). Bands are by element center, and elements below 640px fall into the bottom band.
- The oracle only searches the top-300 TF-IDF candidates, so it is a lower bound on the true ω-oracle.
- The link from ω to acceptance α and latency is a hypothesis. It is to be measured by the GPU agent with these files.
- Playwright's expected chromium build (1248) is not installed. The scorer falls back to the installed headless shell 1243; no browsers were installed.

## Files
- `runs/retrieval/pilot_top3_{tfidf,struct,oracle}.json`: all 50 pilot ids in mid/pilot.jsonl order, each mapped to 3 train_pool ids. Pass them to `poc_rnsd.py --exemplars`.
- `runs/retrieval/pilot_top3_mix.json` (pilot 50) and `{pilot,dev}_top3_{mix,struct_b06}.json`.
- `runs/retrieval/dev_top3_{tfidf,struct,oracle}.json`: the same format for dev.
- `runs/retrieval/per_{dev,pilot}.jsonl` (per-sample metrics), `summary.json` (CIs, β curves per α), `run.log`, `cache_kappa_phi.pkl` (feature cache; delete it to recompute).

## Latency oracle gap (can fast exemplars be learned?)
Background: on pilot 50 with L2 decoding, the oracle-ω exemplars were slower than the TF-IDF exemplars by 1.40 s (95% CI [+0.41, +2.42]). Copy overlap ω is therefore a poor proxy for latency. This section measures real latency for each candidate exemplar on the same request.

### Preparation (CPU, `scripts/retr_cands6.py`)
- `runs/retrieval/pilot_cands6.json` maps each of the 50 pilot ids, in mid/pilot.jsonl order, to 6 different train_pool ids.
- Slots, in order:
  - tfidf1, tfidf2, tfidf3 (the poc ranking)
  - struct1 (β=0.9)
  - oracle1 (argmax ω₄ within the TF-IDF top-300)
  - short: the best TF-IDF-ranked candidate in the top-20 whose GT flow token count is at or below the pool's 25th percentile (370 tokens). Three requests had no such candidate and use the shortest of the top-20.
- Duplicates are replaced by the next TF-IDF rank: struct1 in 3 requests, oracle1 in 5. The replacement is marked in `role` as `*->tfidf_next`.
- Mean exemplar length in Gemma tokens of raw flow HTML: tfidf1 505, struct1 568, **oracle1 717**, short 328. Longer exemplars plausibly explain part of oracle's slowdown.
- Features that can be used at deployment (model input allowed) are in `pilot_cands6_feats.jsonl`:
  - tfidf_sim; struct_sim = cos(φ̂, φ); struct_score
  - n_tokens (raw HTML); css_frac (style tokens / all); pre_body_frac (tokens before `<body>` / all)
  - n_elems (body descendants); n_classes (unique); n_top_blocks (children of body, looking inside a single wrapper); category_match
- Analysis-only values, never model inputs, are in `pilot_cands6_analysis.jsonl`: ω₄(cand, GT) and the GT length.
- The analysis is `scripts/retr_cands6_analyze.py`, which reads `runs/poc/E2B_L2_cands6.jsonl`. It was tested on synthetic data.

### Results (E2B, L2 decoding, k=1 exemplar per request, pilot 50 × 6 candidates = 300 runs)
- The GPU-side scorer crashed at row 61 (`Page crashed`), so `runs/poc/E2B_L2_cands6.jsonl` has only 60 scored rows.
  - I re-scored all 300 rows from the read-only `E2B_L2_cands6_raw.jsonl` with `scripts/retr_cands6_score.py`, writing to `runs/retrieval/cands6_scored.jsonl`.
  - The scores agree with the GPU scorer on 59/60 rows. The one disagreement is in request s2w_68423, which is where the GPU-side page crash happened.
  - One row (s2w_68423 / s2w_54805) crashed the page twice and is scored 0.
- 24 of the 300 runs hit max_new=2048, spread evenly across slots (3–5 per slot).
- Analysis: `scripts/retr_cands6_analyze.py --results runs/retrieval/cands6_scored.jsonl` → `runs/retrieval/cands6_analysis.json`.

**1. Within-request gap.** Means over 50 requests, with bootstrap 95% CI:

| Quantity | Value |
|---|---|
| T with the TF-IDF top-1 exemplar | 9.33 s |
| T with the fastest of the 6 | 4.73 s |
| T with the slowest of the 6 | 17.12 s |
| **min-oracle − tfidf1** | **−4.60 s [−6.21, −3.11]** (relative −37% [−45, −30]) |
| max − tfidf1 | +7.79 s [+5.52, +10.40] |
| Requests where tfidf1 is already the fastest | 20% |
| Gap without the 18 requests that had any max_new hit (n=32) | −4.29 s on 8.26 s |

No fixed slot differs from tfidf1 on average. dT ranges from −0.10 s (short) to +0.83 s (struct1), and every CI covers 0. In this k=1 run, oracle1 is +0.08 s [−1.76, +1.98] (the earlier +1.40 s was measured with 3 exemplars).

**2. Decomposition.** log T = log N_out − log(tok/fwd) + log t_fwd, with t_fwd = T/n_fwd. Within-request variance of log T is 0.220.

| Term | Share of within-request variance |
|---|---|
| Output length N_out | **75%** |
| Acceptance efficiency (tok/fwd) | 25% |
| Per-forward time | ≈0% |

So the latency differences between exemplars come mostly from how long the output is, and only a quarter from how well the drafts are accepted.

**3. Predictability from deployable features.**
- Within-request Spearman of each feature against T is weak everywhere (|ρ̄| ≤ 0.13):
  - n_classes +0.11 [0.00, 0.23] is the only CI that excludes 0, and only barely.
  - struct_sim −0.13 [−0.27, +0.02]; n_tokens +0.08; tfidf_sim −0.01.
  - Even ω₄ vs GT, which is analysis-only, gives −0.09 [−0.21, +0.03].
- The pairwise logistic ranker with leave-one-request-out reaches a pairwise accuracy of **0.571 [0.525, 0.616]**, where chance is 0.5. Mean coefficients (positive = faster): n_tokens −0.54, n_elems +0.36, css_frac −0.27, struct_sim +0.24.

| Selection | T (s) | ΔT vs tfidf1 [CI] | checklist | Δcheck [CI] | elem F1 |
|---|---|---|---|---|---|
| TF-IDF top-1 | 9.33 | 0 | 0.893 | 0 | 0.549 |
| struct top-1 | 10.10 | +0.78 [−1.34, +2.85] | 0.838 | −0.055 [−0.167, +0.053] | 0.502 |
| ω-oracle top-1 | 9.41 | +0.08 [−1.76, +1.98] | 0.870 | −0.023 | 0.553 |
| short | 9.23 | −0.10 [−2.34, +1.92] | 0.850 | −0.043 | 0.469 |
| **learned (LOO)** | 8.74 | −0.59 [−2.61, +1.33] | 0.890 | −0.003 | 0.537 |
| learned + quality ≥ tfidf1 (*) | 8.16 | −1.17 [−2.96, +0.55] | 0.945 | +0.052 [+0.010, +0.105] | 0.553 |
| min-latency oracle | **4.73** | **−4.60 [−6.21, −3.11]** | 0.932 | +0.038 [−0.025, +0.110] | 0.554 |
| min-latency + quality ≥ tfidf1 (*) | 4.90 | −4.43 [−6.06, −2.95] | 0.967 | +0.073 | 0.561 |

(*) The quality filter uses the measured checklist, so it is outcome information and is shown only as a reference.

**4. Verdict: the gap is large but not predictable.**
- The fastest exemplar saves 37–49% of the time, and choosing it costs no quality: the min-latency oracle's checklist is +0.04 (CI covers 0), so there is no quality–latency trade-off.
- The deployable features recover only 13% of the gap (−0.59 s, CI covers 0).
- Because 75% of the variance is output length, which exemplar makes the model write a shorter answer is close to chaotic under greedy decoding. Some of the gap is "luck of the trajectory" that a static exemplar descriptor cannot see.

Caveats:
- n=50, a single run per (request, candidate).
- The min over 6 is optimistically biased by run-to-run timing noise. The noise in t_fwd is small (≈0% variance share), but output length can change discontinuously with the exemplar.
- Possible next steps: predict N_out directly (for example a short prefix rollout or a length predictor), or run 2–3 candidates in parallel and take the first to finish.

## Sequential vs myopic oracle (gate 3, CPU only)
Question: when choosing the draft position, does planning ahead (sequential) reduce the total number of forward passes compared with maximizing the current step's accepted length (myopic)?

### Setup
- Script: `scripts/retr_seq_oracle.py`. Result: `runs/retrieval/seq_oracle.json`. Log: `seq_oracle.log`.
- Runs on 1 core at BelowNormal priority, in about 90 s.
- `scripts/study_draft_sim.py` is used by import only, so the targets, annotator, DOM rule, γ=16 and acceptance rule (longest common prefix + 1 bonus token) are identical to the study simulator.
- Data (no subsampling needed):
  - poc greedy C1, C0, C4g, C7g: n=40
  - gemma rich_fs E2B/E4B/12B: n=30 (re-tokenized responses, approximate)
- Validation against the study simulator:
  - My myopic oracle reproduces `run_policy("oracle")` on 69/70 samples. The exception is one 12B sample that hits max_new=2048 (303 vs 301). The study oracle allows 1 accepted token even when the cap leaves g ≤ 0; mine does not.
  - My DOM reproduces `run_policy("dom")` on 70/70 samples.
- Key structural fact: in greedy replay the context at output position pos is fixed (prompt + y[:pos]). The future therefore depends only on pos, a position choice matters only through its accepted length a, and the next state is pos+a+1.
  - "Top-M positions" therefore means the top-M **distinct** accepted lengths; positions with equal a are equivalent.
  - This also allows an **exact** sequential optimum by dynamic programming: f(pos) = 1 + min_{a∈A(pos)} f(pos+a+1), where A(pos) is every achievable accepted length plus 0. This corresponds to M = all and H = ∞.

### Results (forwards per sample, mean)
| | poc 40 | gemma fs 30 |
|---|---|---|
| myopic oracle (any position) | 103.38 | 107.00 |
| sequential, approximate: M ∈ {4, 8} × H ∈ {1, 3, 6}, all 6 settings | 103.38 | 107.00 |
| **sequential, exact DP** | **103.38** | **107.00** |
| Paired difference myopic − exact | 0 on all 70 samples (mean 0, median 0, CI [0, 0]) | |
| Reduction (per-sample mean / total) | 0.0% / 0.0% | 0.0% / 0.0% |

**Why the difference is exactly zero.** Myopic is provably optimal for the any-position oracle. Let a* be the best accepted length at pos and r(pos) = pos + a* + 1 the furthest reachable point. For any pos' in (pos, r(pos)], the same source shifted by pos'−pos still matches the rest of y up to r(pos)−1. Its remaining length is at most a*, so the γ cap does not bind, and the source stays inside the context. So r(pos') ≥ r(pos): the reach is monotone. By the jump-game exchange argument, taking the maximum jump every time is optimal.

**Executable side (DOM rule) as a reference.** Candidates are DOM's top-M ranked positions, and lookahead is scored with the true target, so every variant below is an **oracle**.

| | poc 40 | gemma 30 |
|---|---|---|
| DOM (executable, myopic) | 176.5 | 204.8 |
| DOM top-4, best-of by true acceptance (H=0) | 150.0 | 165.0 |
| DOM top-8, best-of (H=0) | 139.2 | 153.0 |
| DOM top-8, lookahead H = 1 / 3 / 6 | 139.55 / 139.47 / 139.53 | 153.5 / 153.6 / 153.6 |
| DOM top-8, exact DP | 139.2 | 152.9 |

- Lookahead versus the H=0 best-of with the same M is **slightly worse**: −0.2 to −0.6 forwards, with CIs below 0 (for example poc M8 H3: −0.28 [−0.42, −0.15]; 0 samples better, 11 worse).
- Exact DP over DOM's top-8 versus best-of-8 differs by 0.00 on poc and +0.07 [0.00, 0.20] on gemma (0.04%).
- Lookahead H=1/3/6 versus myopic DOM shows −20% (poc) and −25% (gemma) total. That gain comes **entirely from choosing among the top-M by true acceptance (best-of-M)**; planning ahead contributes nothing.

### Verdict
**Sequential is not meaningfully better than myopic.** The difference is 0 for the any-position oracle (it is provably optimal) and at most 0.04% for the restricted DOM candidate sets. The 3% threshold fails, so **C1 reduces to single-step utility maximization** (maximize the current step's expected accepted length). The remaining room for improvement is in ranking candidates by expected acceptance: DOM best-of-8 is 139 forwards against DOM's 177, and the oracle is 103.

## Ranking vs construction gap (diagnostic only)
Question: the oracle gap between the standard rule (187 forwards) and the oracle position (103) — does it come from choosing better among the candidates SuffixDecoding already finds (ranking), or from candidates SuffixDecoding cannot find (construction)?

### Setup
- Script: `scripts/retr_rank_construct.py`. Result: `runs/retrieval/rank_construct.json` (per sample + summary). Log: `rank_construct.log`.
- 2 threads at BelowNormal priority, about 2 minutes.
- `study_draft_sim.py` and `study_suffix_sim.py` are used by import only. γ=16, acceptance = longest common prefix + 1 bonus, greedy replay.
- Data: poc greedy n=40, gemma fs n=30 (approximate).
- My SD loop reproduces `study_suffix_sim.run_sd` (λ=0, K=1) on **70/70** samples.

Each layer is evaluated as a myopic oracle restricted to that layer: at every step it takes the candidate with the largest true acceptance.

| Layer | Candidates |
|---|---|
| SD | Executable SuffixDecoding, the study implementation |
| O1a | SD's own candidate set (occurrences of the longest suffix match m, last 400), capped at SD's draft length ceil(2m) |
| O1b | The same set, with the draft length raised to γ=16 |
| **O1** | Every position SD can see lexically: every start whose previous token equals the current last token (suffix match n≥1), 16 tokens |
| **O2** | Every position in the context (the oracle position) |
| O3 | O2 plus constructed drafts: (i) a closing-tag sequence built from the full open-tag stack; (ii) structure jumps (top-8 element starts ranked by `sim_score`, no suffix match required). (ii) copies from the context, so it is already a subset of O2. O3 − O2 therefore measures only (i). |
| SD+jump | Executable rule: when SD's longest suffix match is m ≤ m₀, draft from the top-1 structure jump instead |

The O1a/O1b split was not computed for one degenerate gemma 12B sample (2052 tokens, repetitive), which exceeded a 20 s time budget. That split therefore covers 29 of the 30 gemma samples.

### Forwards per sample (mean)
| | poc 40 | gemma 30 |
|---|---|---|
| std (n=3..1, study) | 187.2 | 217.9 |
| SD | 194.9 | 213.7 |
| O1a: SD set, SD length | 178.0 | 174.6 (n=29) |
| O1b: SD set, γ=16 | 157.3 | 158.7 (n=29) |
| **O1: lexical, all n≥1** | **154.2** | **149.9** |
| **O2: any position** | **103.4** | **107.0** |
| O3: + closing tags / jumps | 103.05 | 107.0 |
| SD+jump, m₀=0 | 194.3 | 213.3 |
| SD+jump, m₀=1 | 213.1 | 232.3 |
| SD+jump, m₀=2 | 232.8 | 260.6 |

### Gap decomposition
| Share of SD → O2 | poc | gemma |
|---|---|---|
| ranking (SD → O1) | **45%** (40.8 fwd, CI [31.1, 51.1], median 26.5) | **60%** (63.8 fwd, CI [44.7, 84.2], median 32.5) |
| construction (O1 → O2) | **55%** (50.8 fwd, CI [33.3, 69.5], median 21.0) | **40%** (42.9 fwd, CI [25.7, 61.7], median 22.0) |

- Measured from std → O2 instead, the shares are:
  - poc: std → SD −9% (SD is slower than std here), ranking 49%, construction 61%
  - gemma: std → SD +4%, ranking 58%, construction 39%
- Within ranking, as a share of std → O2 (poc / gemma subset):

| Part of the ranking gap | poc | gemma |
|---|---|---|
| Path choice within SD's own set (SD → O1a) | 20% | 30% |
| SD's draft-length cap ceil(2m) (O1a → O1b) | 25% | 29% |
| Occurrences of shorter suffixes (O1b → O1) | 4% | 9% |

- Paired construction − ranking per sample:

| Group | Mean [95% CI] | Median |
|---|---|---|
| poc | +10.0 [+1.3, +19.3] | −3.5 |
| gemma | −20.9 [−26.0, −16.5] | |
| pooled 70 | −3.2 [−9.8, +3.6] | −7.5 |

  The poc mean is pulled up by a few long C0 samples.
- Generation outside the context adds almost nothing. O3 − O2 is 0.33 forwards on poc (CI [0.13, 0.58]) and 0 on gemma. Closing-tag drafts beat the best context copy on only 30 / 15 trajectory steps, all at tag boundaries.
- The executable structure jump does not help. At m₀=0 it saves 0.6 forwards on poc (0.3%) and 0.4 on gemma (0.2%), significant but negligible. At m₀=1 and m₀=2 it is clearly worse: −18 to −47 forwards, with 36–40 of 40 poc samples worse. The steps where the jump beat SD were all at `boundary` (tag boundaries): 24 on poc and 11 on gemma at m₀=0.

### Verdict
**Construction is not larger than ranking overall.** The two are roughly 50/50: pooled 70 samples −3.2 [−9.8, +3.6], with the median favoring ranking.
- On poc, construction is larger on average (55% vs 45%), driven by a few long samples. On gemma, ranking is larger (60% vs 40%).
- About half of the ranking gap is simply SD's draft-length cap, and another 20–30% is path choice within SD's own candidate set.
- Structure-generated drafts from outside the context are worth ≈0.

Limitation: the "structure jump" is an approximation (top element starts by structural-annotation similarity), not a true exemplar DOM alignment. All numbers are oracle diagnostics.

### O1 vs O2 verdict under the updated decision rules (01:40)
O1 and O2 are **ideal upper bounds that use the future ground-truth tokens**. They are not executable algorithms. The learned ranker numbers come from `runs/study/ranker.json` (hgb_full, per-sample paired). Script: `scripts/retr_o1o2_ctx.py`. Result: `runs/retrieval/o1o2_ctx.json`.

| forwards per sample | poc 40 | gemma fs 30 |
|---|---|---|
| std | 187.2 | 217.9 |
| SD | 194.9 | 213.7 |
| DOM rule | 176.5 | 204.8 |
| **learned ranker (hgb_full)** | **172.8** | **186.1** |
| O1 upper bound (suffix-match candidates) | 154.2 | 149.9 |
| O2 upper bound (any context position) | 103.4 | 107.0 |
| O1 − O2 | 50.8 [34.9, 69.4] (−33% of O1) | 42.9 [26.1, 61.4] (−29%) |
| ranker − O1 | 18.7 [12.6, 25.5] (11% of ranker) | 36.3 [23.3, 51.0] (19%) |
| ranker − O2 | 69.4 [48.2, 95.1] (40%) | 79.1 [49.5, 111.8] (43%) |

How each rule applies:
1. **"O1 and O2 are similar" does not apply.** O2 needs 29–33% fewer forwards than O1, and the CI excludes 0.
2. **"O2 is much better than O1" applies, so the limits of candidate retrieval (suffix matching) are analyzed below.**
3. **"Both upper bounds are close to the current learned ranker" does not apply.** The ranker is 11–19% above O1 and 40–43% above O2, so further work is not put on hold.
   - The room left inside the existing candidates is moderate (ranker → O1).
   - The room outside suffix-matched candidates is larger (O1 → O2).

**Where suffix matching misses.** These are steps on the O2 trajectory where the any-position upper bound beats the suffix-match upper bound:
- How often it happens: 2088 of 4095 steps on poc (51%), 1358 of 3180 on gemma (43%). These steps account for 7306 / 5494 extra accepted tokens.
- **The previous token never appeared earlier in the context** in **70%** (poc) / **44%** (gemma) of these steps. Right after new content (a new word, number or CSS value), suffix matching has no anchor at all, while the continuation that follows exists elsewhere in the context.
- Context of the next token, weighted by extra tokens:

| Context | poc | gemma |
|---|---|---|
| tag boundary `<`, `</`, `">` | 41% | 36% |
| CSS value | 24% | 24% |
| CSS property | 18% | 20% |
| text | 9% | 5% |
| attribute | 7% | 14% |

- The most frequent previous tokens are digits (0/1/2/3/5/8), `<`, `</`, `">`, `</div>`, `header`, `padding` and `width`. These are the transitions after a numeric value or a tag token.
- Where the best source sits: on poc it is own output 54% / prompt (exemplars) 46%; on gemma it is 43% / 57%.

**Conclusion.** The bigger limit is candidate retrieval, not ranking. A suffix-anchored candidate is missing mostly right after new values or text, when the next token is a tag boundary or a CSS property/value. A candidate generator anchored on structure (lexer state, CSS property, tag path) rather than on the last token is the direction worth testing. The earlier simple structure-jump rule did not capture this (see above), and the ranker still has 11–19% room inside the existing candidates.

## Official SuffixDecoding vs reproduction (2026-11-08)
Question: how far is our SuffixDecoding reproduction (`scripts/study_sd_formal.py`, POC_study.md §12) from the official Snowflake ArcticInference implementation, and can the official code run here?

### Source
- Official source: `git clone --depth 1` of `snowflakedb/ArcticInference`, main at commit `aca5d9a` (2026-09-08), Apache-2.0.
  - It is kept in an isolated scratch folder outside the project and treated as untrusted.
  - Read as text only: `csrc/suffix_decoding/suffix_tree.{h,cc}`, `arctic_inference/suffix_decoding/{cache.py, simulator.py}`, `arctic_inference/vllm/{config.py, model_runner.py}`, `docs/suffix-decoding.rst`.
- Port script: `scripts/retr_sd_official_check.py`. Result: `runs/retrieval/sd_official_check.json`. Log: `sd_official_check.log`.

### Can the official code run here?
- **Serving path: no.** SuffixDecoding ships as a vLLM plugin (`vllm.general_plugins` entry point; pinned to vllm 0.26.0 and torch 2.11). vLLM does not run on native Windows, and our decoding stack is transformers + bitsandbytes NF4 (custom loop in `poc_rnsd.py`). An end-to-end official run would need Linux/WSL and vLLM, and the model would no longer be bitsandbytes NF4.
- **Suffix tree alone: possible in principle, not done.**
  - `csrc/suffix_decoding` is a small CPU-only C++20 + nanobind module (`_C`, CMake). The official repo also has a replay simulator, `arctic_inference/suffix_decoding/simulator.py::suffix_decode`, which does exactly what our greedy-target replay does.
  - MSVC BuildTools 14.44 and CMake exist on this PC, and nanobind was installed into an isolated venv.
  - **Configuring and building were refused by the permission policy (no building or running external code), and I did not work around it.**
  - With user approval, the next step would be: build `_C` in the isolated folder, load it from there (not installed into `.venv`), and feed `SuffixDecodingCache` the same prompt and target sequences.
- Instead, I ported the official rules in our own code and measured each difference against the reproduction on the study's main data: poc greedy + gate2 outputs, n=114, the same app-level cross-fit global tree, factor 1.

### Differences (official vs reproduction) and measured impact
"Δ" is the paired difference reproduction − variant in forwards per request (positive = the official rule uses fewer forwards). Reproduction = 109.3 fwd, which equals `runs/study/sd_formal.json` `sd_f1` on **114/114**. The tree reproduction also matches 114/114.

| Item | Official (source) | Reproduction | Δ fwd [95% CI] |
|---|---|---|---|
| Tree / occurrence model | Suffix tree over each sequence, depth ≤ 64; node count = number of suffixes passing through, including suffixes that end there | Occurrence counting on token arrays; ends at the sequence end are counted in the denominator | Equivalent (no Δ; matches the formulation) |
| Path draft and score | Follow `head_child` (highest count); prob = Π count(child)/count(node) = count(P+d)/count(P); stop when prob < 0.1; score = Σ prob | Same, by telescoping | Equivalent |
| Draft length | `min(max_spec_tokens, ⌊m·factor + offset⌋)`; path depth ≤ 64 − m | Same, with MAX_SPEC = 16 | – |
| **MAX_SPEC** | vLLM plugin: `num_speculative_tokens` defaults to **16** when unset (`config.py`). Standalone simulator: `max_spec_tokens` 0 → `max_tree_depth` = **64** | 16 | 64 instead of 16: **6.1 [5.1, 7.4]** (−5.6%), but verify tokens per forward rise 6.16 → 6.94 |
| Match-length loop | `match_len < len(context)`, context = last 64 tokens, so m ≤ 63; best score with `>=` (longer m wins ties) | m ≤ 64; `>=` | m ≤ 63: **0** (no sample changed) |
| Tie among equal-count children | Sibling list sorted by count. On increment a node moves before the head of its new count group (count ≥ 2: most recently incremented first). New children are appended at the tail (count 1: creation order, i.e. earliest occurrence first) | Most recent occurrence always | count-1 ties by earliest occurrence: −0.02 [−0.25, +0.24]; 73 samples change by ±, mean 0 |
| local vs global | `draft1 if score1 >= score2` (local wins ties) | Same | Equivalent |
| **Global tree content** | All cached responses, **including the active request's own response** (`add_active_response` also extends the global tree); prompts only in the local tree | Other apps' responses only (app-level cross-fit, plus 240 other cands6 outputs); own response only in local | Adding own response to global: **1.24 [0.77, 1.73]** (−1.1%) |
| Cache eviction | FIFO over cached requests, `suffix_cache_max_requests` default 100,000 | No eviction (only a few hundred responses) | No effect at this size |
| Same-app responses in global | Allowed (any previous request) | Excluded (cross-fit), which is conservative | Not measured; would make SD stronger |
| Tree draft | Best-first heap: one token per pop, children pushed at node ends with prob ≥ 0.1; budget = max tokens | Expands all children of a node at once, then continues by heap | **−0.11 [−0.54, +0.29]**, no difference (107.8 vs 107.9) |
| Tree verification | Walk the children of the accepted node for each target token (longest matching path) | Same (`accept_tree`) | Equivalent |
| First step | Official simulator speculates already from the prompt (step 0) | Prefill gives the first token, then speculation | Counting convention only (≤ 1 step) |
| Combined (tie1 + m63 + own-global, MAX_SPEC 16) | | | **1.18 [0.63, 1.74]** (−1.1%): 108.1 vs 109.3 fwd |

### What this means
- **The reproduction is faithful in algorithm.** Draft construction, scoring and tie rules give the same forward counts within ±0.2. The tree difference is also nil.
- **Two configuration differences matter.**
  1. **The own response in the global tree** makes the official SD about 1.1% better (1.2 fwd).
     - That is about the same size as the reported SD+learned gain at factor 1: 1.3 [0.7, 1.8] (POC_study §12).
     - The learned ranker would also get these extra candidates, so the gain does not simply cancel. But **the factor-1 claim "SD + learned beats SD" should be re-run with the official global rule before it is cited.**
     - The factor-16 claim (2.4 [1.9, 3.0]) has more margin.
  2. **MAX_SPEC.** The standalone default of 64 gives −5.6% forwards at +13% verify tokens. The vLLM default of 16 is what our runs use, so 16 stays the main setting; 64 should be reported as a sensitivity check.
- Not tested: same-app reuse in global (realistic serving would allow it, so the official SD would be even stronger) and FIFO eviction.
- **Next step (needs user approval):** build the official `_C` in isolation and replay it on the same 114 requests, to confirm the ported rules token by token.
