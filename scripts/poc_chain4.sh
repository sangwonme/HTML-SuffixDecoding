#!/usr/bin/env bash
# C7 after chain3: TF-IDF exemplars shown with interleaved-CSS HTML; greedy + L2 on pilot 10. Deadline guard 23:05.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe -I"
X=runs/retrieval/pilot_exemplar_html_interleaved.json
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a runs/poc/chain4.log; }
late() { [ "$(date +%H%M)" -ge 2305 ]; }
log "chain4 waiting"
until grep -q "chain3 done" runs/poc/chain3.log 2>/dev/null; do late && { log "deadline before chain3 done"; exit 0; }; sleep 15; done
until [ -f "$X" ]; do late && { log "interleaved exemplar file missing at deadline"; exit 0; }; sleep 15; done
log "interleaved ids: $($PY -c "import json;print(len(json.load(open('$X'))))")"
late || { $PY scripts/poc_rnsd.py --n 10 --conds greedy,L2 --exemplar-html "$X" --tag _C7 > runs/poc/E2B_runC7.log 2>&1; log "C7 rc=$?"; }
$PY scripts/poc_eval.py --model E2B > runs/poc/eval4.log 2>&1; log "eval rc=$?"
$PY scripts/poc_table.py E2B >> runs/poc/eval4.log 2>&1; log "table rc=$?"
log "chain4 done"
