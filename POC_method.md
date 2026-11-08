# RNSD: Render-Null relaxed Speculative Decoding (PoC)

Status: 2026-11-07, draft. The theory and the algorithm are final. **The results tables are filled in from `runs/poc/E2B_summary.md`, which is produced by `scripts/poc_eval.py`.**

## 1. Theory: decode in render space, not in token space

The user never consumes the HTML string y. They consume the rendered view

  v = R(y),  R = paint ∘ layout ∘ style ∘ parse,

and R is **many-to-one**: whitespace between tags, comments, optional end tags, quote style and consistent class renaming all leave v unchanged.

**Entropy decomposition.**

  H(Y|X) = H(V|X) + H(Y|V,X).

The second term is entropy that does not affect the render. A token-exact decoder spends forward passes resolving it, so this term is wasted decoding.

**ε-perceptual equivalence.**

  y ~ε y' ⇔ D(R(y), R(y')) ≤ ε,  D = 1 − SSIM of the 360×640 viewport screenshots.

The pixel-difference ratio is recorded alongside D.

**Theorem 1 (acceptance on the quotient space).** Speculative sampling normally accepts with probability 1 − TV(p, q), where p is the target distribution and q the draft distribution over token strings. Suppose instead we run it on the quotient Y/ker R, that is, on the pushforwards R#p and R#q. Then:

- the acceptance probability becomes 1 − TV(R#p, R#q) ≥ 1 − TV(p, q), by the data-processing inequality for the deterministic map R;
- the distribution of the output render V equals R#p, so the method is **lossless on V**.

**Greedy counterpart.** Accept a draft token whenever the draft continuation stays in the same equivalence class as the greedy output.

**How the rule levels relate to R.**

- **L1 (parse)** and **L2 (style)** compute the kernel of the first stages of R *locally*, from the incremental lexer state and the tag stack. They are exact equivalences: an output produced with only L1 and L2 relaxations must render identically to greedy. Checking this after the fact is the key test of the PoC.
- **L4** (CSS numeric and color values, accepted when the draft token is in the target's top-5 with p ≥ τ·p_greedy) is an ε-approximation, so it is **lossy**.
- The levels are nested, L1 ⊂ L2 ⊂ L4. Their acceptance rates should therefore increase monotonically, α_L1 ≤ α_L2 ≤ α_L4, and this is measured below.

## 2. Contexts and rules

| Symbol | Meaning |
|---|---|
| R(y) | Headless 360×640 render: screenshot plus normalized element list (`genui.metrics`) |
| S_null | Lexer contexts where a token cannot change the render |
| S_tol | CSS numeric values (padding, margin, gap, font-size, width, height, …) and color values |

- **RNSD (original run, `rnsd`):**
  - (a) S_null = inter-tag whitespace, comments, and attribute values of data-*, id, aria-*, title.
  - (b) S_tol, accepted when d is in the target's top-5 and p(d) ≥ τ·p(g), τ = 0.1.
- **RNSD+ levels (cumulative):**
  - **L1 parser equivalence:**
    - inter-tag whitespace swaps (both tokens are whitespace);
    - attribute quote style (`="` ↔ `='`);
    - optional end tags for li, p, td, tr, option: the target writes `</x` while the draft starts a sibling `<x`, or the reverse. The innermost open tag comes from a tag stack.
  - **L2 binding-aware class renaming:** class names are variables bound by CSS selectors.
    - Context: inside a `class="…"` value, or at a `.selector` inside `<style>`.
    - If the draft and target emit different identifiers, accept and record σ: target → draft.
    - Afterwards, a target token g in dom(σ) counts as matching draft σ(g).
    - Reject if σ would stop being a bijection, i.e. a name maps to two names or two names map to one.
    - This property is specific to markup with bound selectors and has no analogue in SQL or JSON value generation.
  - **L4:** L2 plus the narrowed S_null and S_tol.
    - The narrowed S_null keeps only whitespace, comments and **data-\*** values.
    - aria-* and title were removed: they carry accessibility meaning, title is shown as a tooltip, and our text-recall metric reads them.
    - id was removed because it may be referenced by CSS or anchors.
    - Difference from `rnsd`: `rnsd` used the wide S_null; L4 uses the narrowed one.

## 3. Algorithm (greedy target, zero-cost draft)

```
C  <- prompt tokens (system + 3 retrieved few-shot HTML examples + query)
prefill target on C -> g0 ; y <- [g0] ; sigma <- {}
while not EOS and |y| < 2048:
    d[0..k-1] <- ngram_lookup(C ++ y, n = 3..1, gamma <= 16)      # most recent match, 0 cost
    logits <- target.forward([y_last] ++ d, kv_cache)              # 1 forward, k+1 positions
    g[i] <- argmax logits[i]
    j <- 0
    while j < k:
        if d[j] == g[j] or d[j] == sigma(g[j]):  accept, j++ ; continue      # exact (PLD)
        s <- lexer_state(decode(y ++ d[:j]))                                    # incremental HTML lexer
        if rule_L(s, d[j], g[j], logits[j], sigma): accept, j++ ; continue      # relaxed (RNSD)
        break
    y <- y ++ d[:j] ++ [g[j]]                  # bonus token from the target
    kv_cache.crop(-(k - j))                    # real rollback (DynamicCache past-recording)
```

E[tokens/step] = (1 − α'^(γ+1)) / (1 − α') uses the positionwise empirical α'. The measured tokens per forward are reported next to it.

## 4. Setup

- Model: Gemma 4 E2B QAT checkpoint, loaded as bitsandbytes NF4 with bf16 compute and SDPA attention.
- Decoding: greedy, max_new 2048.
- Data: the same 10 pilot screens with query_mid. Few-shot uses 3 examples retrieved by TF-IDF over train_pool query_mid; each example's output is its GT flow HTML.
- All conditions run on one hand-written decode loop, so greedy and the speculative variants share the same code path and kernels.
- Conditions are interleaved per sample.
- `greedy_b` is a same-time greedy reference for L1, L2 and L4.

## 5. Results

### 5.1 Stage 1: greedy vs PLD vs RNSD (wide S_null)
E2B, pilot 10, TF-IDF few-shot. Source: `runs/poc/stage1/E2B_summary.md`.

| cond | out tok | total p50 | total max | forwards | tok/fwd | speed-up p50 | identical to greedy | L1 DOM eq | D p50 / max | checklist | elem F1 | relaxed fires |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy (C1) | 435 | 27.1 s | 38.9 s | 435 | 1.00 | 1.00 | - | - | 0 | 1.00 | 0.502 | - |
| PLD (C2) | 443 | 6.6 s | 10.1 s | 104 | 4.58 | **4.47x** | 3/10 | 3/10 | 0.020 / 0.099 | 1.00 | 0.503 | - |
| RNSD (wide S_null) | 414 | 5.3 s | 6.8 s | 80 | 5.24 | **5.60x** | 0/10 | 0/10 | 0.085 / 0.177 | 1.00 | 0.550 | a=24, b=11 |

**Observations**
1. **PLD gives 4.5x on its own.** This is evidence that the retrieved screens and the output share structure, i.e. there is a lot of redundancy.
2. **PLD is lossless only in exact arithmetic.**
   - 7 of 10 outputs diverge from greedy, at token 35–566.
   - The cause is bf16/NF4 kernel differences between a (k+1)-token verify forward and a 1-token forward, which flip near-tie argmaxes.
   - The rendered result is close but not identical (D ≤ 0.10).
3. **RNSD vs PLD.** RNSD was not slower than PLD on any sample. On 2 samples it was within 10% (s2w_42624: 6.70x vs 6.71x; s2w_68423: 3.28x vs 2.88x).
   - Lexer overhead is negligible: one regex pass per *mismatch*, not per token.
   - The speed-up comes from longer accepted runs, 5.24 vs 4.58 tok/fwd.
4. **The checklist metric saturates.** Every condition scores 1.00 on these 10 screens, so it cannot separate the conditions.

The wide S_null (data-*, id, aria-*, title) accepted render-invisible but DOM-visible substitutions such as `data-label="Text"` → `"Toolbar"`. This is why its L1 DOM equivalence is 0/10. The L4 condition uses the narrowed S_null instead.

### 5.2 Stage 2: RNSD+ levels L1 / L2 / L4 vs same-time greedy_b
Source: `runs/poc/stage2/E2B_summary.md`. Same 10 screens; conditions are interleaved per sample.

| cond | total p50 | forwards | tok/fwd | ᾱ' (pos-weighted) | E[tok/step] formula | L1 DOM eq | L2 style eq | D p50 / max | checklist | elem F1 | fires |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy_b | 28.1 s | 435 | 1.00 | - | 1 | 10/10 vs greedy (bit-identical) | 10/10 | 0 | 1.00 | 0.502 | - |
| L1 | 6.4 s | 104 | 4.58 | 0.830 | 5.63 | 3/10 | 5/10 | 0.020 / 0.099 | 1.00 | 0.503 | **0** (no L1 event occurred) |
| L2 v1 (naive rename) | 6.2 s | 87 | 5.34 | 0.870 | 6.95 | 0/10 | 0/10 | 0.102 / 0.203 | 1.00 | 0.532 | rename 14, σ-match 3 |
| L4 (L2 v1 + narrow S_null + S_tol) | 5.7 s | 87 | 5.63 | 0.874 | 7.12 | 0/10 | 0/10 | 0.105 / 0.157 | **0.883** | 0.571 | a2 19, rename 10, b 17 |

**Findings**
1. **α_L increases monotonically.** ᾱ' goes 0.830 → 0.870 → 0.874, and tok/fwd goes 4.58 → 5.34 → 5.63. This is consistent with L1 ⊂ L2 ⊂ L4.
2. **Repeated greedy is deterministic.** greedy_b is token-identical to greedy on 10/10 samples, so plain decoding is reproducible. The only source of non-identity is the multi-token verify forward. The L1 run is token-identical to PLD (L1 never fired), and like PLD it matches greedy on only 3/10 samples.
3. **L1 events never occurred.** None of inter-tag whitespace swaps, quote style, or optional end tags happened. The few-shot examples are minified, single-quote-style HTML, so the draft and the target always agree on those surface choices. L1 equals PLD here.
4. **The naive L2 rule was NOT render-lossless (0/10 style-equal).** This is a negative result, and it is informative. The v1 rule accepted *any* identifier swap inside `class="…"`, for example `class="icon"` → `class="row"` and `class="ph"` → `class="pager"`. But the CSS rules were already emitted in `<style>` before the body, so those names were **bound**. Swapping a bound name at a use site changes the computed style.
   - Correct α-renaming must rename only **fresh** names: the definition site and all uses together, before any rule binds the name.
   - **v2 (binding-aware) fix:** reject a rename when the target or draft identifier is already bound, meaning it appears as a complete `.selector` in `<style>` or in a complete `class` value earlier in the output.
   - C3 and all C4/C5/pilot-50 runs use v2.
5. **L4 is lossy, as expected for an ε-rule.** The checklist drops from 1.00 to 0.883. Element F1 is not a reliable signal here: it rose slightly, but that is noise at n = 10.

### 5.3 C0–C7 table (one page)
Source: `runs/poc/E2B_Ctable.md`.
- Every row uses the same 10 pilot screens. Speed-up is relative to C1 greedy, as a per-sample ratio (p50).
- "Render = own greedy" compares each RNSD-L2 row with the greedy run that used the same exemplars. For C5 there is no oracle-greedy run, so C5 is compared with C1 greedy.

| row | few-shot | decoding | checklist | elem F1 | type F1 | total p50 | tok/fwd | speed-up vs C1 | render = own greedy (DOM / style) | D p50 |
|---|---|---|---|---|---|---|---|---|---|---|
| C0 | none | greedy | 0.867 | 0.480 | 0.675 | 59.4 s | 1.00 | 0.47x | - | - |
| C1 | TF-IDF | greedy | 1.000 | 0.502 | 0.747 | 27.1 s | 1.00 | 1.00x | - | - |
| C2 | TF-IDF | PLD | 1.000 | 0.503 | 0.757 | 6.6 s | 4.58 | 4.47x | 3/10 / 5/10 | 0.020 |
| C3-naive | TF-IDF | RNSD-L2 v1 | 1.000 | 0.532 | 0.763 | 6.2 s | 5.34 | 4.32x | 0/10 / 0/10 | 0.102 |
| C3 | TF-IDF | RNSD-L2 v2 | 1.000 | 0.526 | 0.763 | 5.8 s | 5.53 | 4.61x | 2/10 / 3/10 | 0.042 |
| C4g | struct | greedy | 1.000 | 0.502 | 0.754 | 24.1 s | 1.00 | 1.04x | - | - |
| **C4** | **struct** | **RNSD-L2 v2** | 1.000 | 0.509 | 0.754 | **5.3 s** | 5.65 | **5.66x** | **6/10 / 6/10** | **0** |
| C5 | oracle | RNSD-L2 v2 | 0.967 | 0.653 | 0.841 | 4.7 s | 5.97 | 5.58x | 0/10 vs C1 greedy | 0.150 |
| C6, C7g, C7 | | | see `E2B_Ctable.md` (run after 21:40) | | | | | | | |

Zero-shot (C0) is about 2x slower than C1 and less accurate: its output is twice as long and its checklist is 0.867. Few-shot exemplars help both accuracy and latency; this is the ω claim.

**Pilot-50 paired bootstrap** (L2 v2 decoding, B = 10,000). Source: `runs/poc/E2B_ci.md`.

| | checklist | all items | elem F1 | type F1 | total s | tok/fwd |
|---|---|---|---|---|---|---|
| C1-L2 (TF-IDF) | 0.948 [0.903, 0.985] | 0.88 | 0.602 | 0.803 | 5.56 | 5.25 |
| C4-L2 (struct) | 0.982 [0.960, 1.000] | 0.94 | 0.612 | 0.813 | 5.71 | 5.22 |
| C5-L2 (oracle) | 0.962 | 0.88 | 0.633 | 0.852 | 6.96 | 5.38 |
| **C4 − C1** | **+0.033 [+0.002, +0.072] \*** | +0.060 [−0.02, +0.14] | +0.009 (ns) | +0.010 (ns) | +0.15 s (ns) | −0.04 (ns) |
| C5 − C1 | +0.013 (ns) | 0 | +0.031 (ns) | **+0.049 [+0.010, +0.088] \*** | **+1.40 s \*** | +0.13 (ns) |

**Reading the results**
- **Checklist saturation.** On pilot-50, C1 is not saturated (0.948). C4 is near saturation (0.982); it is marked "saturated" in `E2B_ci.md` because its mean is ≥ 0.98.
- **Struct retrieval (C4)** improves the content checklist significantly, by +3.3 points, at the same latency. This is the "same or better accuracy" condition.
- **RNSD-L2 decoding** gives a 4.6–5.7x wall-clock speed-up over greedy with the same exemplars. With struct exemplars, 6/10 renders are style-identical to their own greedy output and D p50 = 0.
- **Remaining non-identity** has two sources:
  - multi-token verify numerics: PLD alone is identical on only 3/10;
  - any accepted rename changes the conditioning of later tokens.
  The equivalence holds per token, but the target's continuation is not equivariant to renaming. This is the main limit of the "lossless on V" claim for a greedy target.
- **Oracle exemplars (C5)** raise structural similarity (type F1 +0.049\*) but are about 1.4 s slower, because their outputs are longer.

### 5.4 Perceptual latency (study's `perceptual_latency`)
Source: `runs/poc/E2B_perc.md`. Same 10 screens.
- Token arrival times are reconstructed from the decode trace: all tokens accepted in verify step s arrive at ttft + s·(T − ttft)/S, where S is the number of verify steps.
- t50 = the time at which the partial render reaches 50% of the final-view SSIM gain.
- head_share = the fraction of output tokens that come before `<body`.

| row | T p50 | **t50 p50** | t50 vs C1 | Lperc/T | t50/T | head_share |
|---|---|---|---|---|---|---|
| C0 zero-shot greedy | 59.4 s | 48.2 s | 2.35x | 0.857 | 0.860 | 0.711 |
| C1 TF-IDF greedy | 27.1 s | **20.3 s** | 1.00 | 0.805 | 0.806 | 0.492 |
| C2 TF-IDF PLD | 6.6 s | 4.5 s | 0.23 | 0.759 | 0.779 | 0.491 |
| C3 TF-IDF RNSD-L2 | 5.8 s | 3.5 s | 0.18 | 0.708 | 0.673 | 0.451 |
| **C4 struct RNSD-L2** | 5.3 s | **3.3 s** | **0.15** | 0.753 | 0.746 | 0.517 |
| C7g interleaved-exemplar greedy | 24.7 s | 16.2 s | 0.87 | 0.668 | 0.672 | 0.263 |
| C7 interleaved-exemplar RNSD-L2 | 5.1 s | 4.0 s | 0.20 | 0.691 | 0.694 | 0.270 |

**Headline.** The user sees half the screen after **20.3 s with C1 and 3.3 s with C4**. Accuracy is the same or better: on pilot-50 the C4 checklist is +0.033, a significant difference.

**Interleaving (C7).**
- Under greedy decoding, interleaved exemplars change what the model generates. head_share drops from 0.49 to 0.26, and t50/T drops from 0.81 to 0.67, so t50 goes from 20.3 s to 16.2 s.
- Under RNSD-L2 the gain disappears: C7 has t50 4.0 s, versus 3.5 s for C3. The CSS head is copied almost verbatim from the exemplars, so speculative decoding already emits it in a few forwards. Moving it later saves almost nothing, and the interleaved form slightly lowers the draft match rate (4.18x vs 4.61x speed-up).

## 6. Gate 2: DOM-aware drafting vs standard PLD on real GPU wall-clock

**Question.** Does DOM-aware drafting reduce real wall-clock time compared with standard PLD, after counting the cost of proposing drafts (parsing and matching)?

**Setup.**
- Verification is exact (L2 and other relaxations are off).
- Model: E2B NF4, greedy target, TF-IDF top-3 exemplars, `query_mid`.
- Each condition runs 3 times, and the condition order is shuffled per sample and per repetition. There are 3 warm-up runs.
- The measuring process runs at Normal priority. Other agents run BelowNormal CPU simulations on 1–2 cores at the same time.

**How the DOM drafter works.** `poc_rnsd.DomDrafter` imports `Lexer`, `sim_score` and `Index` from `study_draft_sim.py`, so it uses the same rule as the simulator.
- The annotation state and the n-gram index are updated incrementally from new tokens only; nothing is re-parsed per step.
- Offline replay of the C1 greedy outputs gives forward counts that match the simulator on **10/10** samples.

### 6.1 Pilot 10 (3 reps)
Source: `runs/poc/gate2/gate2_summary.md`.

| metric (mean over requests) | PLD | DOM | PLD − DOM [95% CI] |
|---|---|---|---|
| wall-clock total | 6.64 s | 6.19 s | **+0.45 s [+0.22, +0.71]** |
| target forwards | 103.7 | 96.4 | +7.3 [+4.2, +10.9] |
| tokens / forward | 4.58 | 5.02 | −0.43 [−0.68, −0.22] |
| verify time | 6.38 s | 5.91 s | +0.47 s |
| draft proposal time | 0.016 s (0.15 ms/step) | 0.039 s (0.32 ms/step, plus 8 ms prompt init) | −0.023 s |
| output tokens | 442.7 | 445.8 | n.s. |

- Relative wall-clock gain: **+6.9% [+3.5, +11.0]**. Relative forward reduction: +7.3% [+4.2, +10.9]. The simulator predicted −7.9%.
- Drafting is cheap: 0.6% of wall-clock for DOM and 0.2% for PLD.
- Peak GPU allocation is 6.9 GB in both conditions.
- Repeated runs of the same condition give identical outputs (20/20). PLD and DOM outputs are identical in 5/10 samples; the others differ because of multi-token verify numerics.
- The run-to-run CV of wall-clock has a median of 9.2%, which is why repetitions are needed.
- **Gate 2 (pilot 10): PASSED.** The wall-clock CI is above 0.

### 6.2 Pilot 50 (3 reps, 300 runs)
Source: `runs/poc/gate2_p50/gate2_summary.md`.

| metric (mean over requests) | PLD | DOM | PLD − DOM [95% CI] |
|---|---|---|---|
| wall-clock total | 5.85 s | 5.48 s | **+0.370 s [+0.181, +0.572]** |
| target forwards | 96.9 | 90.0 | +6.86 [+3.60, +10.24] |
| tokens / forward | 4.98 | 5.52 | −0.54 [−0.71, −0.38] |
| verify time | 5.60 s | 5.21 s | +0.39 s [+0.20, +0.59] |
| draft proposal time | 0.013 s (0.14 ms/step) | 0.034 s (0.30 ms/step, plus 7.7 ms prompt init) | −0.021 s |
| output tokens | 422.5 | 428.4 | −5.9 (n.s.) |
| TTFT | 0.228 s | 0.228 s | 0 |

- **Relative wall-clock gain: +6.95% [+4.06, +10.00]. Relative forward reduction: +7.77% [+4.76, +10.93].**
- Drafting cost is 0.62% of wall-clock for DOM and 0.23% for PLD, so almost all of the forward reduction carries over to wall-clock.
- Peak GPU allocation is 7.0 GB in both conditions. Process peak RSS is 10.1 GB, set during model load.
- **Simulator vs measured.** For every measured output, the simulator's predicted forward count equals the measured count: **300/300** (150 PLD, 150 DOM). The incremental drafter also equals the simulator on 300/300. Simulator results therefore transfer exactly to forward counts, and the only extra cost in wall-clock is the measured draft overhead.
- **Noise.** Outputs are identical across reps on 100/100 request-condition pairs. The median run-to-run CV of wall-clock is 1.75%; the pilot-10 run, measured while other jobs were starting, had 9.2%.
- PLD and DOM outputs are identical on 22/50 requests. The others diverge because of multi-token verify numerics; the output-length difference is not significant.
- **Gate 2 (pilot 50): PASSED.** The wall-clock paired-difference CI is above 0. On real GPU wall-clock, including drafting cost, DOM-aware drafting is about 7% faster than standard PLD.

### 6.3 Gate 2b: learned ranker (MLP over DOM top-8 candidates) vs DOM rule, real GPU wall-clock
Source: `runs/poc/gate2b/gate2_summary.md`.

**Setup.**
- Pilot 50, 3 reps, conditions {PLD, DOM, RANKER} with shuffled order per sample and rep. Exact verification, 3 warm-ups.
- RANKER is `study_ranker.RankerDrafter`. Its weights come from the cross-fit fold that did not train on the request's app (`ranker_fold_map.json`), so there is no app leakage.

| metric (mean over requests) | PLD | DOM | RANKER |
|---|---|---|---|
| wall-clock total | 6.53 s | 6.12 s | **5.83 s** |
| target forwards | 96.9 | 90.0 | **85.5** (simulator predicted 86.3) |
| tokens / forward | 4.98 | 5.52 | 5.85 |
| draft cost per step | 0.16 ms | 0.34 ms (+9 ms init) | 0.53 ms (+4.6 ms init) |
| draft share of wall-clock | 0.23% | 0.64% | 0.85% |

| paired difference | wall-clock [95% CI] | wall-clock % | forwards [95% CI] | forwards % |
|---|---|---|---|---|
| **DOM − RANKER** | **+0.293 s [+0.049, +0.531]** | **+5.13% [+1.83, +8.10]** | +4.54 [+0.82, +8.14] | +5.46% [+2.05, +8.49] |
| PLD − RANKER | +0.702 s [+0.434, +0.980] | +11.87% [+8.49, +15.24] | +11.40 [+7.14, +16.08] | +13.12% [+9.62, +16.65] |
| PLD − DOM (Gate 2 replication) | +0.410 s [+0.204, +0.623] | +6.76% [+3.73, +9.89] | +6.86 [+3.60, +10.24] | +7.77% [+4.76, +10.93] |

- Outputs are identical across reps for every condition (50/50 each). Between policies, outputs are identical on 21–22 of 50 requests because of multi-token verify numerics. Output length does not differ significantly.
- The run-to-run CV of wall-clock has a median of about 6.5%, higher than the 1.75% in Gate 2. CPU simulations by other agents were running at the same time. Even so, the effect is significant with 3 reps × 50 requests.
- Peak GPU is 7.0 GB for every policy, so the ranker adds no GPU memory.
- **Simulator / replay vs measured forwards: 450/450 exact** (PLD, DOM and RANKER). Replaying each measured output with the same policy reproduces the measured forward count exactly. Forward-count predictions from the CPU simulator therefore carry over without error.
- **Gate 2 replicated: PASSED. Gate 2b: PASSED.** The DOM − RANKER wall-clock CI is above 0. The learned structure-conditioned drafter beats the DOM rule on real wall-clock, including its own overhead. PLD → DOM → RANKER is monotone, and RANKER is about 12% faster than PLD in total.

### 6.4 P1: robustness to HTML representation changes (no retraining)
The transforms come from `src/genui/perturb.py`, and each one renders pixel-identically to the original:
- **fmt**: newlines, indentation, attribute order and quote style
- **css**: `<style>` rules inlined into `style=` attributes
- **dom**: `display:contents` wrappers added
- **cls**: classes renamed

The RANKER uses the original-format cross-fit weights and is not retrained.

**Condition A (prompt exemplars and draft source both transformed; GPU, pilot 50, 1 rep).** Sources: `runs/poc/p1A/<t>/gate2_summary.md`, `runs/poc/p1A/p1A_eval.md`.
- *fmt*

  | | PLD | DOM | RANKER |
  |---|---|---|---|
  | wall-clock | 17.2 s | 15.4 s | 16.1 s |
  | target forwards | 275 | 241 | 248 |
  | tokens / forward | 2.43 | 2.80 | 2.56 |
  | output tokens | 655 | 652 | 621 |

  - PLD − DOM = **+6.4% [+0.4, +12.7]**, so the DOM rule still helps.
  - DOM − RANKER = **−11.5% [−27.8, −1.1]**; in seconds it is −0.76 s [−1.61, +0.11]. The learned ranker loses to the rule.
  - Acceptance collapses for every policy, from about 5 tokens/forward to 2.5.
- *css*: skipped by instruction, because like fmt it increases the token count; the simulated condition B covers it. Only 1–2 requests were run before the stop.
- *dom*, *cls*: running; results go in §6.4.2.

**Cause analysis, fmt** (`scripts/p1A_cause.py`; the RANKER trajectory is replayed on the measured outputs). Source: `runs/poc/p1A/cause_fmt.md`.
1. **Format drift of the target.** The model output follows the fmt style and also pretty-prints the CSS inside `<style>` (`* {\n  box-sizing: …`). But the fmt transform leaves `<style>` minified in the exemplars. Drafts copied from the exemplar CSS therefore stop matching, and mean accepted length falls from 3.99 to 1.51 for the MLP and from 3.50 to 1.62 for the rule. Candidates are still available: the n-gram hit rate is 0.85 versus 0.79 and the mean number of candidates is 4.0 versus 3.3. This is **transform bias**: the generated format is a mixture the exemplars do not contain.
2. **Ranker vs rule at the same state.** On original outputs the MLP beats the rule on 11.9% of multi-candidate steps and loses on 4.8%. Under fmt it wins on 6.8% and loses on 9.4%. Most losses happen in CSS property and value contexts (cssprop 140, cssval 96 of 400) and at `<html\n  <head>`-style line-break boundaries, where the MLP jumps to a structurally similar but differently formatted block.
3. **Feature distribution shift** (candidate means at multi-candidate steps, orig → fmt):
   - depth_cur 3.68 → 2.43
   - sib_diff 0.97 → 0.59
   - path_suf 1.66 → 1.21
   - lex_eq 0.85 → 0.74
   - dom_score 6.87 → 6.08

   The history inputs (`acc_last`, `acc_mean3`) also move far out of the training range, because mean acceptance drops from about 4 to 1.5. The MLP is miscalibrated off-distribution, while the hand-written rule is not.

**Condition B (only the draft source transformed; ESTIMATED wall-clock, no GPU).**
- Forwards come from the study simulator (`p1_perturb_report.json` simB, greedy replay; this replay matched the measured forwards exactly on 300/300 and 450/450 in Gates 2/2b).
- Wall-clock is ESTIMATED as TTFT + (F−1)·(measured per-forward time) + measured draft cost.
- The estimator reproduces the Gate 2b measurements within 0.5% (ratios 0.998 / 0.999 / 1.005).
- Source: `runs/poc/p1B/p1B_estimate.md`.

| source | PLD est. | DOM est. | RANKER est. | DOM − RANKER (est.) | PLD − RANKER (est.) |
|---|---|---|---|---|---|
| orig | 6.51 s | 6.12 s | 5.86 s | +0.26 s [+0.21, +0.31] (+5.2%) | +11.4% |
| fmt | 9.43 s | 8.85 s | 8.62 s | +0.23 s [+0.16, +0.31] (+2.8%) | +9.2% |
| css | 13.45 s | 12.78 s | 12.68 s | +0.10 s [+0.05, +0.15] (+0.8%) | +5.7% |
| dom | 6.63 s | 6.29 s | 6.06 s | +0.23 s [+0.19, +0.27] (+4.4%) | +9.4% |
| cls | 7.21 s | 6.78 s | 6.53 s | +0.25 s [+0.20, +0.30] (+4.3%) | +10.2% |

**Reading A against B.**
- When only the reference changes (B), the learned ranker keeps its edge over the rule for every transform. The edge shrinks for token-inflating transforms (css +0.8%).
- When the target format also changes (A, fmt), the ranker falls below the rule.
- The failure is therefore driven by **target-side distribution shift**: lower acceptance and out-of-range history and structure features. A changed reference alone does not cause it.
- Implication for Stage 1: train on mixed-format targets, or normalize format-dependent features (depth and sibling under whitespace, history scaled by the running acceptance level).

#### 6.4.2 Condition A: all four transforms (GPU, pilot 50, 1 rep)
Sources: `runs/poc/p1A/<t>/gate2_summary.md`, `p1A_eval.md`, `cause_fmt.md`, `cause_dom_cls_css.md`. css was run fresh in the night queue; the stopped partial run is kept in `css_partial_stopped`.

| source | tok/fwd PLD / DOM / RANKER | wall-clock PLD / DOM / RANKER | **PLD − DOM** [95% CI] | **DOM − RANKER** [95% CI] | PLD − RANKER |
|---|---|---|---|---|---|
| orig (gate2b) | 4.98 / 5.52 / 5.85 | 6.53 / 6.12 / 5.83 s | +6.8% [+3.7, +9.9] | +5.1% [+1.8, +8.1] | +11.9% |
| fmt | 2.43 / 2.80 / 2.56 | 17.2 / 15.4 / 16.1 s | +6.4% [+0.4, +12.7] | **−11.5% [−27.8, −1.1]** | +2.9% (n.s.) |
| css | 5.17 / 6.18 / 6.38 | 9.46 / 8.17 / 8.07 s | **+11.9% [+6.4, +16.8]** | 0.0% [−6.9, +5.3] | +13.4% [+7.8, +18.4] |
| dom | 5.44 / 5.89 / 6.11 | 7.07 / 6.57 / 6.34 s | +6.9% [+4.2, +9.6] | +3.7% [−0.9, +8.3] | +10.8% [+6.5, +15.1] |
| cls | 3.94 / 4.23 / 4.45 | 7.04 / 6.46 / 6.33 s | +6.2% [+1.2, +10.7] | +2.6% [−2.1, +6.6] | +9.6% [+4.4, +14.2] |

**Format adherence.** The outputs do follow the new representation (`p1A_eval.md`). The figures below are exemplar → output.
- fmt: newlines per 1,000 chars go from 17.3 in the exemplars to 39.1 in the outputs, against 0.3 in the original. The outputs pretty-print even the `<style>` CSS, which the exemplars keep minified.
- css: the inline-style share is 0.79 in the exemplars and 0.74 in the outputs, against 0.11 in the original.
- dom: the `display:contents` wrapper share is 0.18 in the exemplars and 0.13 in the outputs.
- cls: the share of standard class names is 0 in the exemplars and 0.19 in the outputs, against 0.89 in the original.

**Quality.** The checklist stays between 0.89 and 0.96 for every set and policy, against about 0.92 for the original. Element F1 is 0.54–0.62. There is no systematic quality loss from the transforms or from the drafting policy. The cls set is lowest at 0.544 element F1, against 0.59–0.62 for the others.

**Reading the results**
- **The DOM rule keeps a significant gain over PLD under every representation change**, from +6.2% to +11.9%.
  - The gain is largest for css (+11.9%), where inline styles create many repeated attribute strings at different structural positions. This is exactly the ambiguity that structural selection resolves.
  - The draft cost stays below 1% of wall-clock.
- **The learned ranker (original-format weights, not retrained):**
  - It significantly loses only under fmt.
  - Under css, dom and cls it is level with or slightly ahead of the rule; the means are +0 to +3.7%, none significant at n = 50 × 1 rep.
  - Same-state counterfactual replay (`cause_dom_cls_css.md`) shows the MLP still beats the rule more often than it loses on multi-candidate steps:

    | set | MLP better | MLP worse |
    |---|---|---|
    | dom | 15.3% | 5.3% |
    | cls | 7.6% | 4.3% |
    | css | 10.9% | 6.5% |
    | fmt | 6.8% | 9.4% (reversed) |

  - The feature shift under dom (depth_diff 0.64 → 1.03, sib_diff 0.97 → 1.68) does not hurt. The shift under fmt (depth_cur and sib_diff drop because of whitespace text; history features fall far below the training range) does.
  - At wall-clock level the remaining ranker advantage shrinks for two reasons. Its per-step cost is about 0.2 ms higher than the rule's. And once a trajectory diverges, later steps change.
- **Failure taxonomy of MLP < rule steps** (lexical state at the step):
  - fmt: mostly CSS property and value contexts and line-break boundaries.
  - cls: CSS values, boundaries and properties (48 / 22 / 20 of 131).
  - The MLP jumps to a structurally similar block whose formatting or class vocabulary differs from the current output.

#### 6.4.3 Condition B (draft source only): GPU measurement, compared with the simulated estimate
Measured on pilot 50, 1 rep, starting 03:04 after the Parkinson yield. Sources: `runs/poc/p1B/<t>/gate2_summary.md` and `quality.md`. The prompt is the original, so outputs follow the original format. Only the draft source changes.

| source | measured wall-clock PLD / DOM / RANKER | **DOM − RANKER measured** [95% CI] | DOM − RANKER estimated (§6.4) | PLD − DOM measured | PLD − RANKER measured | tok/fwd PLD / DOM / RANKER |
|---|---|---|---|---|---|---|
| fmt | 8.72 / 7.94 / 7.75 s | +2.5% [+0.7, +4.3] | +2.8% | +7.2% [+4.2, +10.4] | +9.6% [+6.4, +13.0] | 3.13 / 3.40 / 3.52 |
| css | 11.98 / 11.39 / 11.48 s | −1.3% [−5.6, +1.9] (n.s.) | +0.8% | +4.3% [+2.1, +6.4] | +3.7% [+0.0, +5.8] | 2.08 / 2.19 / 2.23 |
| dom | 6.07 / 5.88 / 5.66 s | +5.7% [+0.5, +9.7] | +4.4% | +3.5% [+0.6, +5.9] | +9.2% [+4.4, +13.2] | 4.88 / 5.20 / 5.66 |
| cls | 6.40 / 6.09 / 5.83 s | +4.6% [+2.2, +7.1] | +4.3% | +4.7% [+2.9, +6.5] | +9.2% [+6.3, +11.9] | 4.51 / 4.82 / 5.11 |

- **The estimate predicts the measured relative effects well.** DOM − RANKER differs from the estimate by at most 2 points, and the sign matches except for css, where both values are near 0.
- Absolute measured times are 7–11% below the estimate. The estimate used the gate2b per-forward time, which was measured while CPU simulations were running.
- When only the reference changes, **the learned ranker keeps a significant edge over the rule for fmt, dom and cls**. For css it ties.
- The PLD → DOM gain stays significant under every transform.
- Quality is unchanged: the checklist is 0.923–0.928 for every policy and transform, and element F1 is 0.595–0.616.
- Together with condition A, this localizes the fmt failure to **target-side format drift**, not to a perturbed reference.

### 6.5 Independent test (E): train-split ranker on unseen test requests
This section is **separate from the cross-fit pilot results above**.
- **Training data (E(i))**: 150 train-pool requests with 150 distinct apps, none shared with pilot, dev or test. Retrieval excludes the request itself and same-app exemplars. They were decoded greedily with PLD (`runs/poc/E/train150/E2B_pld.jsonl`). The study agent then retrained one MLP on these targets (`runs/study/ranker_mlp_trainsplit.npz`).
- **Evaluation (E(ii))**: 100 test requests from 92 apps, none shared with pilot. Conditions are {PLD, DOM, RANKER (train-split)} × 1 rep with shuffled order, exact verification, and TF-IDF top-3 exemplars. Sources: `runs/poc/E/test100/gate2_summary.md` and `quality.md`.

| metric (mean over requests) | PLD | DOM | RANKER |
|---|---|---|---|
| wall-clock total | 5.28 s | 5.17 s | **4.86 s** |
| target forwards | 88.5 | 86.0 | **80.5** |
| tokens / forward | 5.20 | 5.74 | 6.22 |
| output tokens | 399 | 433 | 424 |
| draft cost per step | 0.14 ms | 0.28 ms | 0.47 ms |
| checklist | 0.944 | 0.938 | 0.943 |
| element F1 / type F1 | 0.591 / 0.823 | 0.586 / 0.813 | 0.587 / 0.813 |

| paired difference (n = 100) | wall-clock | wall-clock % | forwards % |
|---|---|---|---|
| **DOM − RANKER** | **+0.310 s [+0.083, +0.581]** | **+5.90% [+3.15, +8.72]** | +6.36% [+3.53, +9.25] |
| PLD − RANKER | +0.419 s [+0.149, +0.655] | +9.51% [+5.03, +13.29] | +10.87% [+6.23, +14.77] |
| PLD − DOM | +0.109 s [−0.179, +0.352] | +2.30% [−2.93, +6.70] (n.s.) | +3.21% [−2.21, +7.77] (n.s.) |

Quality paired differences are all within ±0.01 and none is significant. For example, RANKER − DOM on the checklist is +0.005 [−0.007, +0.020].

**Reading the results**
- **On unseen test requests the train-split ranker beats both baselines significantly**: +5.9% over the DOM rule and +9.5% over PLD in wall-clock, with no quality change. This replicates Gate 2b out of sample.
- **The plain DOM rule was not significantly faster than PLD on this test set** (+2.3%, CI includes 0).
  - Its tok/fwd advantage is still clear (5.74 vs 5.20).
  - But its outputs happen to be about 34 tokens longer, because multi-token verify numerics make the outputs diverge from PLD on 48 of 100 requests. Output length is not caused by the policy; at n = 100 × 1 rep this variance hides the 3% forward gain.
  - The tok/fwd gain itself (+10%) is in line with the pilot.
  - A repeat (3 reps) or a fixed-output comparison (forward counts on identical outputs, as in condition B or the replay) would settle the DOM-vs-PLD point on test.

### 6.6 External data (F): WebSight v0.2 eval 50
- **Setup**: 50 WebSight v0.2 eval pages (Tailwind HTML written by another LLM, a different source from our S2W converter). Prompts use TF-IDF top-3 exemplars from the 150 WebSight pool pages (`runs/retrieval/external_websight_prompts.jsonl`). Conditions {PLD, DOM, RANKER} × 1 rep, shuffled order, exact verification. RANKER uses the train-split weights (`runs/study/ranker_mlp_trainsplit.npz`), not retrained. Ran 05:09–06:47. Sources: `runs/poc/F_websight/gate2_summary.md`, `websight_quality.md`.

| metric (mean over requests) | PLD | DOM | RANKER |
|---|---|---|---|
| wall-clock total | 39.79 s | 38.28 s | 38.28 s |
| target forwards | 683.9 | 651.3 | 649.5 |
| tokens / forward | 2.06 | 2.16 | 2.15 |
| output tokens | 1420 | 1412 | 1402 |
| hit max_new (2,048) | 5/50 | 2/50 | 4/50 |
| draft cost per step | 0.15 ms | 0.36 ms | 0.49 ms |
| render_ok | 1.000 | 1.000 | 1.000 |
| element F1 / type F1 / text recall | 0.454 / 0.695 / 0.547 | 0.467 / 0.704 / 0.573 | 0.469 / 0.703 / 0.556 |

| paired difference (n = 50) | wall-clock | wall-clock % | forwards % |
|---|---|---|---|
| **PLD − DOM** | **+1.512 s [+0.143, +2.805]** | +3.22% [−0.24, +6.46] | **+4.17% [+0.71, +7.38]** |
| PLD − RANKER | +1.507 s [+0.491, +2.499] | +3.50% [+0.85, +5.97] | +4.76% [+2.15, +7.18] |
| DOM − RANKER | −0.005 s [−1.073, +1.089] | −0.53% [−3.35, +2.22] (n.s.) | −0.20% [−3.02, +2.54] (n.s.) |

**Reading the results**
- **The DOM rule transfers to an outside HTML source.** It is significantly faster than PLD in seconds and in forwards. The per-request percentage CI just includes 0; as in condition B, the verdict is taken in seconds (Gate 2 passed in `gate2_summary.md`).
- **The learned ranker's gain does not transfer.** It is no faster than the rule (−0.53%, n.s.). Its gain over PLD (+3.5%) is the rule's gain. The ranker was trained only on S2W converter output, so its extra gain over the rule is so far shown only inside that format (Gate 2b not passed).
- **Quality is unchanged**: element F1 within 0.015, all 150 outputs render.
- **Caveats**
  - Drafting works much worse here: 2.1 tok/fwd vs 5–6 on S2W.
  - Outputs are far longer than the GT (about 1,410 vs 525 tokens), so a request takes about 39 s; 2–5 per condition hit max_new.
  - Outputs are identical between policies on only 1–2 of 50 requests, so output-length noise enters the wall-clock comparison. n = 50 × 1 rep.
  - The GT replay proxy predicted −6.1% forwards for DOM (DATASET.md "External data (prep)"); the GPU run gives −4.2%, same direction, smaller.
  - The cause of the ranker's loss (feature distribution shift, as in the fmt failure) has not been analysed yet.

## 7. Limitations

- n = 10. NF4 is an approximation of q4_0.
- The PLD "lossless" property holds in exact arithmetic only. A forward over k+1 tokens uses different bf16 kernels than a 1-token forward, so near-tie argmax flips can make PLD diverge from greedy. We measured this; see `identical`.
- σ is applied to token pieces, not whole identifiers.
- The lexer is a regex approximation with no full HTML5 tokenizer, and script and textarea are ignored.
