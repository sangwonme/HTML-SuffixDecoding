#!/usr/bin/env bash
# Latency oracle gap (k=1 exemplar, 6 candidates per request, RNSD-L2). After chain5; no new sample after 22:55.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe -I"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a runs/poc/chain6.log; }
log "chain6 waiting"
until grep -q "chain5 done" runs/poc/chain5.log 2>/dev/null; do [ "$(date +%H%M)" -ge 2255 ] && { log "deadline before chain5 done"; exit 0; }; sleep 15; done
until [ -f runs/retrieval/pilot_cands6.json ]; do [ "$(date +%H%M)" -ge 2255 ] && { log "cands6 missing at deadline"; exit 0; }; sleep 15; done
log "cands6: $($PY -c "import json;d=json.load(open('runs/retrieval/pilot_cands6.json'));print(len(d),'requests',sum(len(v) for v in d.values()),'cands')")"
$PY scripts/poc_cands.py --stop 2255 > runs/poc/E2B_runCands6.log 2>&1; log "cands GPU rc=$? $(grep -E 'requests completed|stop time' runs/poc/E2B_runCands6.log | tail -1)"
$PY scripts/poc_cands.py --score >> runs/poc/E2B_runCands6.log 2>&1; log "score rc=$?"
log "chain6 done"
