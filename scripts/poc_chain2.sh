#!/usr/bin/env bash
# After C0: wait for exemplar files, then C4 (greedy+L2), C5 (L2) on pilot 10, then pilot-50 L2 runs, then eval.
# Each poc_rnsd.py run holds ../.gpu_busy during its measured loop. Deadline guard: stop launching new runs after 23:05.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe -I"
S=runs/retrieval/pilot_top3_struct.json
O=runs/retrieval/pilot_top3_oracle.json
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a runs/poc/chain2.log; }
late() { [ "$(date +%H%M)" -ge 2305 ]; }
until [ -f runs/poc/E2B_runC0.log ] && grep -qE "^done|Traceback|SystemExit" runs/poc/E2B_runC0.log; do sleep 10; done
log "C0 finished"
until [ -f "$S" ] && [ -f "$O" ]; do late && { log "exemplar files missing at deadline"; break; }; sleep 15; done
if [ -f "$S" ]; then
  log "struct ids: $($PY -c "import json;print(len(json.load(open('$S'))))")  oracle ids: $($PY -c "import json;print(len(json.load(open('$O'))))" 2>/dev/null)"
  late || $PY scripts/poc_rnsd.py --n 10 --conds greedy,L2 --exemplars "$S" --tag _C4 > runs/poc/E2B_runC4.log 2>&1; log "C4 rc=$?"
  [ -f "$O" ] && { late || $PY scripts/poc_rnsd.py --n 10 --conds L2 --exemplars "$O" --tag _C5 > runs/poc/E2B_runC5.log 2>&1; log "C5 rc=$?"; }
fi
late || $PY scripts/poc_rnsd.py --n 50 --conds L2 --tag _P50C1 > runs/poc/E2B_runP50C1.log 2>&1; log "P50C1 rc=$?"
[ -f "$S" ] && { late || $PY scripts/poc_rnsd.py --n 50 --conds L2 --exemplars "$S" --tag _P50C4 > runs/poc/E2B_runP50C4.log 2>&1; log "P50C4 rc=$?"; }
[ -f "$O" ] && { late || $PY scripts/poc_rnsd.py --n 50 --conds L2 --exemplars "$O" --tag _P50C5 > runs/poc/E2B_runP50C5.log 2>&1; log "P50C5 rc=$?"; }
$PY scripts/poc_eval.py --model E2B > runs/poc/eval.log 2>&1; log "eval rc=$?"
$PY scripts/poc_table.py E2B >> runs/poc/eval.log 2>&1
$PY scripts/poc_ci.py E2B L2_P50C1 L2_P50C4 $( [ -f runs/poc/E2B_L2_P50C5.jsonl ] && echo L2_P50C5 ) >> runs/poc/eval.log 2>&1; log "ci rc=$?"
log "chain2 done"
