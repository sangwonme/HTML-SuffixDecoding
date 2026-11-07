#!/usr/bin/env bash
# Gate 2b: PLD vs DOM vs RANKER (learned MLP, app-held-out fold), exact verification, pilot 50 x 3 reps, shuffled.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe -I"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a runs/poc/gate2b/chain.log; }
log "start"
$PY scripts/poc_rnsd.py --n 50 --conds pld,dom,ranker --reps 3 --shuffle --outdir runs/poc/gate2b > runs/poc/gate2b/run.log 2>&1; log "run rc=$?"
$PY scripts/gate2_analyze.py runs/poc/gate2b > runs/poc/gate2b/analyze.log 2>&1; log "analyze rc=$?"
$PY scripts/gate2_check.py measured runs/poc/gate2b > runs/poc/gate2b/check.log 2>&1; log "check rc=$?"
log "gate2b done"
