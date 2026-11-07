#!/usr/bin/env bash
# P1 condition B: prompt keeps ORIGINAL exemplars; only the draft source is transformed. After p1A.
# Order fmt, css first (priority), then dom, cls. RANKER: existing cross-fit weights + fold map.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe -I"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a runs/poc/p1B/chain.log; }
log "waiting for p1A"
until grep -q "p1A done" runs/poc/p1A/chain.log 2>/dev/null; do sleep 15; done
for t in fmt css dom cls; do
  mkdir -p runs/poc/p1B/$t
  log "start $t"
  $PY scripts/poc_rnsd.py --n 50 --conds pld,dom,ranker --reps 1 --shuffle \
      --draft-source-html runs/retrieval/pilot_exemplar_html_$t.json --outdir runs/poc/p1B/$t > runs/poc/p1B/$t/run.log 2>&1
  log "$t rc=$?"
  $PY scripts/gate2_analyze.py runs/poc/p1B/$t > runs/poc/p1B/$t/analyze.log 2>&1
done
log "p1B done"
