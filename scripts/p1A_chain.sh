#!/usr/bin/env bash
# P1 condition A: exemplars AND draft source transformed (fmt, css, dom, cls) x {PLD, DOM, RANKER} x pilot 50 x 1 rep.
# Starts after gate2b chain is done (gate2b untouched). RANKER uses the existing cross-fit weights + fold map.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe -I"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a runs/poc/p1A/chain.log; }
log "waiting for gate2b"
until grep -q "gate2b done" runs/poc/gate2b/chain.log 2>/dev/null; do sleep 15; done
for t in fmt css dom cls; do
  mkdir -p runs/poc/p1A/$t
  log "start $t"
  $PY scripts/poc_rnsd.py --n 50 --conds pld,dom,ranker --reps 1 --shuffle \
      --exemplar-html runs/retrieval/pilot_exemplar_html_$t.json --outdir runs/poc/p1A/$t > runs/poc/p1A/$t/run.log 2>&1
  log "$t rc=$?"
done
for t in fmt css dom cls; do $PY scripts/gate2_analyze.py runs/poc/p1A/$t > runs/poc/p1A/$t/analyze.log 2>&1; done; log "analyze done"
$PY scripts/p1A_eval.py > runs/poc/p1A/eval.log 2>&1; log "eval rc=$?"
log "p1A done"
