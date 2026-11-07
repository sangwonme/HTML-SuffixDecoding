#!/usr/bin/env bash
# After chain4 (all GPU runs done): perceptual-latency table (CPU only).
cd "$(dirname "$0")/.." || exit 1
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a runs/poc/chain5.log; }
log "chain5 waiting"
until grep -q "chain4 done" runs/poc/chain4.log 2>/dev/null; do [ "$(date +%H%M)" -ge 2315 ] && break; sleep 15; done
.venv/Scripts/python.exe -I scripts/poc_perc.py E2B > runs/poc/perc.log 2>&1; log "perc rc=$?"
log "chain5 done"
