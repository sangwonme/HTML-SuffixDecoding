#!/usr/bin/env bash
# Gate 2: PLD vs DOM-aware draft (exact verification), 3 reps, shuffled order per sample/rep. pilot 10 then pilot 50.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe -I"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a runs/poc/gate2/chain.log; }
log "start pilot10"
$PY scripts/poc_rnsd.py --n 10 --conds pld,dom --reps 3 --shuffle --outdir runs/poc/gate2 > runs/poc/gate2/run.log 2>&1; log "pilot10 rc=$?"
if [ "$(date +%H%M)" -lt 2350 ] || [ "$(date +%H)" = "00" ]; then
  $PY scripts/poc_rnsd.py --n 50 --conds pld,dom --reps 3 --shuffle --outdir runs/poc/gate2_p50 > runs/poc/gate2_p50/run.log 2>&1; log "pilot50 rc=$?"
fi
log "gate2 chain done"
