# HTML-SuffixDecoding (GenerativeUI)

**Structure-Conditioned Speculative Draft Selection for HTML Generation.**

When an LLM generates a mobile UI as HTML, the output repeats tokens heavily, both from the few-shot exemplars and from itself. Speculative decoding by copying (prompt lookup, SuffixDecoding) therefore helps a lot. However, the same token pattern recurs in different DOM contexts, so purely lexical matching often copies from the wrong position.

This project builds a (natural-language query → UI HTML) task from Screen2Words/RICO. It measures latency and accuracy with local Gemma 4 models, and studies **choosing the draft position using the structure of the partial DOM**: a DOM rule and a lightweight learned ranker.

New agents and contributors should start with **[RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md)** and **[AGENTS.md](AGENTS.md)**.

## Current status (2026-11-08 morning) — recommendation: Conditional Go
- **Main result.** On 100 unseen test requests (92 apps, none shared with pilot; one rep, E2B NF4), a structure-conditioned draft ranker trained only on train-split apps is faster than both baselines, with no quality change (`POC_method.md` §6.5):
  - +5.9% [+3.2, +8.7] wall-clock vs the DOM rule
  - +9.5% [+5.0, +13.3] vs prompt lookup
- **Pilot, 3 reps** (`POC_method.md` §6.2–6.3): DOM rule +6.95% vs PLD; ranker +5.1% vs the DOM rule.
- **Robustness (P1, §6.4).**
  - The DOM rule stays significantly faster than PLD under all four pixel-identical representation changes (+6% to +12%).
  - The learned ranker (original-format weights) loses to the rule when the output format changes to pretty-printed HTML (fmt: −11.5%).
- **Against formal SuffixDecoding** (simulation, official global-cache rule, `POC_study.md` §12b–15):
  - Adding DOM features to SD's candidate selection saves only 1–3% forwards (factor 16), and about 0.5% or n.s. at the default factor 1.
  - SD's global cache is the bigger lever. The oracle with global positions is 46.9 forwards, against 94.9 for tuned SD.
- **In progress:** WebSight generalization run (`runs/poc/F_websight/`). Data prep is in `DATASET.md` "External data (prep)".
- **Morning report:** `REPORT_2026-11-08.md` (`paper/report_2026-11-08.pdf`). **Paper draft:** `paper/genui_paper.pdf`.
- **Awaiting user decisions:** approval to build the official ArcticInference suffix-tree module, a code license, and the data release scope.

## Folder layout
```
GenerativeUI/
  README.md  RESEARCH_CONTEXT.md  AGENTS.md     # entry points (start here)
  PLAN.md  NOTES.md                             # plan / decision log (Korean)
  DATASET.md                                    # dataset card
  POC_method.md  POC_study.md  RETRIEVAL.md     # experiment reports
  REPORT_2026-11-08.md                          # morning report (summary + recommendation)
  src/genui/        metrics.py checklist.py retrieval.py canon.py interleave.py perturb.py
  scripts/          build_* / eval_* / bench_gemma.py / poc_* (GPU decoding) / study_* (CPU simulation)
                    / retr_* (retrieval and oracle analysis) / gate2* / p1* (GPU chains)
  data/processed/   pilot, dev, test jsonl + mid/ (query_mid, checklists); train_pool.jsonl.gz (decompress before use)
  data/review/      comparison images and review sheet
  runs/             all raw results: baselines, baselines_mid, gemma, poc (GPU), study (simulation), retrieval
  paper/            genui_paper (English paper draft), report_2026-11-08, earlier drafts (tex/pdf)
```
Not in git: `models/` (Gemma weights, 47 GB), `data/raw/` (2.4 GB), `data/processed/train_pool.jsonl` (81 MB; shipped as `train_pool.jsonl.gz`, 7.7 MB), `data/processed/screenshots/`, `.venv/`, `.cache/`, `*.pkl`. See `runs/repo_audit.md`.

## Quick reproduction
Reference environment: Windows 11, RTX 4070 12 GB, Python 3.11 managed by uv. All CPU-only analyses (study, retrieval) run without a GPU.

```bash
# 1) environment
uv venv --python 3.11 .venv
uv pip install --python .venv torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv transformers==5.22.2 bitsandbytes==0.50.2 accelerate==1.15.0 tokenizers==0.23.3 \
    huggingface_hub==2.4.0 safetensors==0.8.0 numpy==2.4.6 scipy==1.17.1 scikit-learn==1.9.1 pandas==3.1.1 \
    pyarrow==26.0.1 pillow==13.0.0 lxml==6.1.3 playwright==1.64.1 psutil==7.2.2 matplotlib==3.11.2 \
    scikit-image==0.26.0 tqdm==4.70.1
.venv/Scripts/python.exe -m playwright install chromium-headless-shell   # headless only, no browser window

# 2) data (HF rootsautomation/RICO-Screen2Words, CC-BY-4.0, about 2.5 GB) -> data/processed
.venv/Scripts/python.exe -I scripts/download_data.py
.venv/Scripts/python.exe -I scripts/build_dataset.py      # deterministic, about 12 s; rebuilds train_pool.jsonl
#   (shortcut for train_pool only: gzip -dc data/processed/train_pool.jsonl.gz > data/processed/train_pool.jsonl
#    sha1 dad43c770029a4ea6fa87674190926e557f83ca2)
.venv/Scripts/python.exe -I scripts/validate_dataset.py   # render check, about 2 min
.venv/Scripts/python.exe -I scripts/build_query_mid.py    # query_mid and checklists, about 1.5 min

# 3) models (only for GPU decoding; E2B is enough for the PoC)
.venv/Scripts/python.exe -I scripts/download_models.py E2B [E4B 12B]

# 4) examples
.venv/Scripts/python.exe -I scripts/eval_mid_baselines.py --splits pilot,dev,test   # non-LLM baselines (CPU)
.venv/Scripts/python.exe -I scripts/study_draft_sim.py                              # draft-position simulator (CPU)
.venv/Scripts/python.exe -I scripts/retr_struct.py                                  # exemplar retrieval study (CPU)
.venv/Scripts/python.exe -I scripts/poc_rnsd.py --model E2B --n 50 --conds pld,dom,ranker --reps 3 --shuffle --outdir runs/poc/<name>   # GPU
```
The versions above are the ones installed in the original `.venv`. The exact GPU command lines used for each result are in the run logs under `runs/poc/` and in the chain scripts (`scripts/gate2*_chain.sh`, `scripts/p1*_chain.sh`).

## Data and licenses
- Derived from **Screen2Words** (Wang et al., UIST 2021) and **RICO** (Deka et al., UIST 2017) through the HF mirror `rootsautomation/RICO-Screen2Words` (CC-BY-4.0). Please cite both.
- Gemma 4 weights are not included (Apache-2.0 per the model card; "Gemma" is a Google trademark).
- No license has been chosen yet for the project code.
