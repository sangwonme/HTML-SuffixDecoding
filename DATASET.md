# Dataset card: S2W-Query2HTML (v0.1, 2026-11-07)

This dataset is a **(natural-language user query → UI HTML)** task derived from Screen2Words (RICO Android screens). Everything is built deterministically from the source data: no LLM rewriting, a fixed seed, and a build script that can be re-run (running it twice gives identical JSONL files, checked with sha1).

- Source: HF `rootsautomation/RICO-Screen2Words` (CC-BY-4.0). This is the RICO-SCA subset with 22,417 screens and 5 human summaries per screen. The official train/val/test splits have **no app overlap** (0 shared apps, checked).
- Build: `.venv/Scripts/python.exe -I scripts/build_dataset.py` (about 12 s), then `scripts/validate_dataset.py` (about 2 min).
- License: derived from CC-BY-4.0 data. Cite Screen2Words (Wang et al., UIST 2021) and RICO (Deka et al., UIST 2017).

## Files (`data/processed/`)
| File | Rows | Source split | Purpose |
|---|---|---|---|
| `test.jsonl` | 500 | official test | Main evaluation. Stratified by category. |
| `pilot.jsonl` | 50 | **subset of test.jsonl** | Pilot for timing and sanity checks. Prompt tuning must not use it. |
| `dev.jsonl` | 100 | official val | Prompt tuning and few-shot debugging. |
| `train_pool.jsonl` | 14,522 | official train (all kept) | Retrieval baseline and few-shot example pool. |
| `screenshots/<id>.jpg` | 600 | | Original RICO screenshots, for test and dev only. |
| `build_report.json`, `validation_report.json` | | | Filter counts, split sizes, render results, length statistics. |
| `../review/cmp_XX_<id>.png` | 20 | | Original, GT abs, and GT flow placed side by side (first 20 test rows). |

### Fields
`id` (`s2w_<screenId>`), `screen_id`, `app`, `category`, `source_summary`, `source_summary_index`, `all_summaries`, `query_short`, `query_rich`, `key_texts`, `element_counts` (componentLabel → count), `gt_html_flow`, `gt_html_abs`, `screenshot_path` (null in train_pool).

## Construction rules
1. **Tree.** The input tree is RICO `semantic_annotations` (componentLabel, bounds, text, iconClass). We use it instead of the raw view hierarchy because layout wrappers are already removed.
   - Invisible nodes are dropped: zero or negative area, or fully off-screen (68 nodes in total).
   - Children are sorted top-to-bottom, then left-to-right.
2. **Summary choice (consensus rule).** Each of the 5 captions gets a score: its mean token-Jaccard similarity to the other 4. We take the highest-scoring caption (ties go to the lower index) that has at least 4 words and at least 2 non-generic words. The generic stoplist includes words like page, screen, displaying, app, options, list. The chosen caption index is stored in `source_summary_index`. On test it is spread across indices 0–4: 71/120/125/105/79.
3. **`query_short` template.**
   - Caption starts with an article → `Create {s}.`
   - One of the first 3 words is a screen noun (page, screen, pop-up, list, menu, ...) → `Create a/an {s}.`
   - Otherwise → `Create a mobile app screen showing {s}.`
4. **`query_rich`** is `query_short` plus two more lines:
   - `Elements: <count> <label>s, ...` (sorted by count, highest first)
   - `Key texts (top to bottom): "..."; ...` (up to 15 unique texts from Text, Text Button, Input, Toolbar, Checkbox, Radio, and Switch nodes, each cut to 40 characters)
5. **GT HTML.** Two versions are generated from the same tree. Every element carries `data-label="<componentLabel>"`.
   - **flow** (for structure comparison): unlabeled wrappers are collapsed. A container becomes `div.row` when all its children share one horizontal band, and `div.col` otherwise. Consecutive List Items are wrapped in `<ul class=row|col>`. Images and icons get their scaled size.
   - **abs** (for visual comparison): each labeled node is `position:absolute` at its bounds scaled by 0.25 (1440×2560 → 360×640), placed relative to its labeled parent.
   - Void elements (input, img) cannot hold children, so their children are emitted as following siblings. A `p` or `button` that has children becomes a `div`, so the HTML parser does not auto-close it.

### Component mapping
| componentLabel | HTML |
|---|---|
| Text | `<p>` (`<div>` if it has children) |
| Text Button | `<button>` (`<div>` if it has children) |
| Input | `<input type=text placeholder="{text}">` |
| Image | `<img alt="{text}" class=ph>` (gray placeholder) |
| Icon | `<span class=icon aria-label="{iconClass}">` |
| List Item | `<li>` inside `<ul>` |
| Toolbar | `<header>` |
| Card | `<article>` |
| Checkbox / Radio Button | `<input type=checkbox>` / `<input type=radio>` |
| On/Off Switch | `<input type=checkbox role=switch>` |
| Slider / Number Stepper / Date Picker | `<input type=range|number|date>` |
| Modal | `<div role=dialog>` |
| Drawer | `<aside>` |
| Multi-Tab | `<nav role=tablist>` |
| Bottom Navigation | `<nav>` |
| Button Bar | `<div role=group>` |
| Pager Indicator | `<div class=pager>` |
| Background Image | `<div class="bgimage ph">` |
| Advertisement | `<aside class=ad>` |
| Web View / Map View | `<div class="webview ph">` / `<div class="map ph">` |
| Video | `<video class=ph>` |
| (other / unlabeled) | `<div>` or collapsed |

## Filtering (applied in order, per screen)
| Rule | train | val | test |
|---|---|---|---|
| Input | 15,743 | 2,364 | 4,310 |
| F0 JSON parse error | 0 | 0 | 0 |
| F1 landscape | 0 | 0 | 0 |
| F2 fewer than 3 labeled elements | 241 | 28 | 61 |
| F3 more than 150 labeled elements | 1 | 1 | 2 |
| F4 no visible text in text-bearing elements | 337 | 56 | 83 |
| F5 Web View, Map View, Video, or Ad covering ≥50% of the screen | 293 | 47 | 91 |
| F6 low-quality summary (no caption with ≥4 words and ≥2 content words) | 221 | 35 | 79 |
| F7 duplicate within app (same label and text sequence) | 128 | 19 | 29 |
| **Kept** | **14,522** | **2,178** | **3,965** |

Background Image was deliberately left out of F5, because it often sits behind real forms such as login screens.

## Sampling
- Proportional stratification by `category` (largest-remainder allocation), with seeded sampling inside each category.
  - test uses seed 20261107, pilot seed +1 (drawn from test), dev seed +2.
- Test covers 27 categories. The largest are Social 39, Travel 38, Entertainment 36, and Communication 35.

## Validation (headless Chromium, network blocked)
- **Render success is 100%** for both flow and abs on every split, including all 14,522 train_pool rows.
  - Success means: no JS error, a non-empty body, and a DOM `[data-label]` count equal to the sum of `element_counts`.
- Lengths on test (Gemma 4 tokenizer):

| Field | mean | p50 | p95 | max |
|---|---|---|---|---|
| query_short tokens | 10.4 | 10 | 13 | 15 |
| query_rich tokens | 82.7 | 73 | 145 | 205 |
| gt_html_flow tokens | 571 | 476 | 1,103 | 4,655 |
| gt_html_abs tokens | 942 | 746 | 2,070 | 4,764 |
| labeled elements | 16.9 | 12 | 44 | 72 |
| key texts | 7.1 | 6 | 15 | 15 |

## Known limitations
- Summaries are short and sometimes ungrammatical ("page displays to create a profile card"). Templating keeps their errors.
- `query_short` underspecifies the screen, so many different UIs are valid answers. `query_rich` is the main setting for that reason.
- The GT contains no style at all (no colors, fonts, or icons beyond gray placeholders). Visual similarity to GT therefore penalizes nicely styled outputs.
- RICO labels can be odd. For example, quote text is sometimes labeled Text Button. The GT inherits these labels.
- The flow layout is a heuristic: horizontally scrolled carousels can become columns, and background images come before the content that sits on them.
- The status bar and navigation bar are not part of the GT. Bounds keep their y offset (root usually starts at y=84 px, which is 21 css px).
- Key texts are cut at 40 characters, and input values become placeholders.
- `pilot` is a subset of `test`, which keeps pilot cost low. Tune prompts only on `dev`.

## query_mid + checklist (v0.2 addition, 2026-11-07)
A third query level for every sample. Files: `data/processed/mid/{pilot,dev,test,train_pool}.jsonl` (joined to the main files by `id`; the original JSONL files are unchanged). Build: `.venv/Scripts/python.exe -I scripts/build_query_mid.py` (about 1.5 min, CPU only, deterministic: re-run gives identical sha1).
- **Why**: `query_rich` gives element counts and all key texts, so element F1 mostly measures copying; `query_short` is one vague line. `query_mid` = purpose (a human caption, sometimes with the app's store name) + 2–4 salient elements, with no counts and no long verbatim text. Many UIs satisfy it, so the **main metric is checklist satisfaction**; element/type F1 is secondary.
- **Example**: "Create a page showing activation code to continue on app. Include a top bar with a title about token, a button labeled "Cancel", and a "Continue" button." → `[{"type":"Toolbar","keyword":"token"},{"type":"Text Button","keyword":"cancel"},{"type":"Text Button","keyword":"continue"}]`
- **Fields**: `id, query_mid, checklist, checklist_mode (keyword|mixed|type_only), n_items, n_keyword_items, few_salient (<2 keyword items), purpose_summary_index, app_name, leak_ok, leak{digits,count_words,verbatim,unmentioned}, usable, gt_self, n_candidates, n_candidates_gt_valid, category, screenshot_path`. Evaluate only rows with `usable=true`.
- **Item selection**: candidates from GT `data-label` nodes: Input placeholder, short Text Button labels (≤4 words), toolbar title, checkbox/switch/radio label, bottom-nav/tab label, a distinctive (low document-frequency) word from long button/text, short headings (max 1 Text item); then type-only items (Input, List, Modal, Drawer, Image, Card, ...). Every candidate is first checked on the GT render; only those the GT satisfies under `metrics.NORMALIZE_JS` are kept (about 96% pass; rejects are mostly small images that the normalizer calls Icon, icon-only buttons, and checkbox labels outside the control's parent).
- **Checklist scoring** (`src/genui/checklist.py`): item satisfied if ≥`min_count` elements of an accepted normalized category contain the keyword (NFKC, lowercase, punctuation→space, substring, space-free fallback; text includes innerText plus placeholder/aria-label/alt/value/title, and for form controls their label/short parent text). Type `Text` = keyword anywhere in the page (`ATTR_TEXT_JS`). On/Off Switch also accepts a plain checkbox. `check_score` = share of items satisfied, `check_all` = all satisfied, `check_lenient` = keyword anywhere ignoring the type pairing.
- **Leak checks** (automatic, per row): no digits; no count words (two…twenty, outside quoted labels); no 5-word n-gram copied from any GT text; every item's keyword and type word appear in the query. Failing rows retry with other templates and other human captions, else `leak_ok=false`.

| split | rows | usable | leak pass | GT self (flow) | GT self (abs, independent) | few_salient |
|---|---|---|---|---|---|---|
| pilot | 50 | 50 | 100% | 1.000 | 1.000 | |
| dev | 100 | 100 | 100% | 1.000 | 1.000 | 22 |
| test | 500 | 500 | 100% | 1.000 | 1.000 | 131 |
| train_pool | 14,522 | 14,516 | 99.99% (1 verbatim) | 1.000 | – | 3,545 (5 empty checklists) |

- Query length ≈ 26 words. Items per row (test): 2: 215, 3: 215, 4: 65, 1: 5.
- Review sheet: `data/review/mid_samples.md` (20 random dev+test rows with screenshot paths).
- Limitations: RICO labels share sheets as Drawer, so some queries say "side drawer menu" for a share pop-up; type-only items (e.g. "a button", "an image") are easy; keyword-token phrasing ("a button mentioning name") can read oddly; captions keep their grammar errors.

## External data (prep, 2026-11-08)
Small external subsets for a later generalization check of the draft-selection results (does the gain depend on our converter's HTML format?). These files are CPU preparation only; no GPU runs have been made with them yet. Scripts: `scripts/prep_external.py` and `scripts/prep_external_replay.py`. Downloaded files are **untrusted**: they are only parsed with lxml (`python -I`), never executed or rendered here.

| | WebSight | Design2Code |
|---|---|---|
| Source | HF `HuggingFaceM4/WebSight`, config v0.2 (1,922,671 rows) | HF `SALT-NLP/Design2Code` (484 pages) |
| License | CC-BY-4.0 | ODC-BY |
| Format | `text` (HTML using the Tailwind 2.2.19 CDN), `llm_generated_idea`, `image` | `{id}.html` + `{id}.png` (images replaced by a placeholder) |
| Downloaded | 200 rows, text fields only (no images), via the datasets-server rows API: 100 from offset 0 and 100 from offset 1,000,000. Raw copy: `data/raw/WebSight/websight_v0.2_rows200.json` (0.4 MB) | All 484 `.html` files (35.6 MB), no png. Raw copy: `data/raw/Design2Code/` |
| Query | `llm_generated_idea`, the prompt the page was generated from | No NL description exists, so the query is a deterministic template built from the page: title, first heading, up to 2 section headings, up to 3 short link/button labels (similar to query_mid) |
| HTML length (Gemma tokens) | p50 503, p90 693, max 971 | p50 25,712, p90 63,100, max 92,519 |
| Fits generation (≤1,800 tokens; max_new is 2,048) | 200/200 | **16/484** |
| Fits as an exemplar (≤4,000 chars, same cap as poc_rnsd) | 200/200 | 15/484 (**0 in the pool**) |
| Split (seed 20261108) | eval 50 / pool 150 | eval 16 / pool 468 |

Output: `data/processed/external/{websight,design2code}.jsonl` (fields: `id, source, query, html, html_chars, n_tokens, fits_gen, fits_exemplar, split, key_texts, n_elements`) and `prep_report.json`. `design2code.jsonl` (35 MB) is excluded from git and can be regenerated.

**Pipeline prep (WebSight)**
- Retrieval: TF-IDF over the pool queries, top-3, 4,000-char cap. Result: `runs/retrieval/external_websight_top3.json`.
- Prompts: same chat layout as poc_rnsd (SYSTEM, 3 × (query, fenced exemplar HTML), query). Written to `runs/retrieval/external_websight_prompts.jsonl` (`id, msgs, example_ids`).
  - A GPU run needs an option in the decoding script to read these prompts instead of the pilot. That change belongs to the PoC agent and is not made here.
- Replay proxy, with the GT HTML standing in for model output: `runs/retrieval/external_replay.json`. γ=16, the same simulator as POC_study §9. Forwards per request, mean:

| | n | T (tokens) | std PLD | DOM rule | any-position oracle | std − DOM [95% CI] | oracle vs std |
|---|---|---|---|---|---|---|---|
| WebSight eval | 50 | 525 | 202.8 (2.59 tok/fwd) | 190.4 | 101.3 | 12.4 [10.7, 14.1] (−6.1%) | −50% |
| S2W pilot (same proxy) | 50 | 653 | 179.5 (3.64 tok/fwd) | 173.1 | 99.9 | 6.5 [5.0, 7.9] (−3.6%) | −44% |

  - In this proxy the DOM rule's gain does **not** disappear on a different HTML source: it is −6.1% on WebSight vs −3.6% on S2W. Less can be copied by the standard rule on WebSight (2.6 vs 3.6 tokens per forward).
  - This is a proxy only. Real model outputs differ from GT, so the GPU run with real outputs decides.

**Limitations and plan**
- Design2Code pages are too long for the current budget (E2B, max_new 2,048): only 16 fit, and the pool has no exemplar under the 4,000-char cap.
  - Options: zero-shot or cross-source exemplars (WebSight pool) for those 16 pages; a larger max_new; or cutting pages into sections. Not done yet.
- WebSight relies on the Tailwind CDN, which our renderer blocks (network off), so renders are unstyled.
  - Forward counts are not affected. Visual metrics would be; text recall over `key_texts` is the usable accuracy proxy.
  - S2W checklists do not exist for these sources (no `data-label` categories).
- WebSight HTML is LLM-generated (synthetic) and may overlap with model pre-training data.
- Next step: a GPU run of WebSight eval 50 × {PLD, DOM, ranker} with these prompts. Then replay the real outputs with `study_draft_sim` / `study_ranker`, using the frozen cross-fit ranker with no retraining.
