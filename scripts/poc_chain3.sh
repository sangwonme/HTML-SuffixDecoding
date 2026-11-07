#!/usr/bin/env bash
# Appended after chain2 (chain2 untouched): C6-L2 on pilot 50 with mixed exemplars, then re-run eval/table/CI.
# Priority: C6 over P50C5. If C6 has not started by 22:45 because chain2 is still on P50C5, stop that P50C5 run.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe -I"
M=runs/retrieval/pilot_top3_mix.json
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a runs/poc/chain3.log; }
late() { [ "$(date +%H%M)" -ge 2305 ]; }
log "chain3 waiting"
until grep -q "chain2 done" runs/poc/chain2.log 2>/dev/null; do
  if [ "$(date +%H%M)" -ge 2245 ] && grep -q "P50C4 rc" runs/poc/chain2.log 2>/dev/null && ! grep -q "P50C5 rc" runs/poc/chain2.log; then
    pid=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*poc_rnsd.py*_P50C5*' -and \$_.Name -like 'python*' }).ProcessId" | tr -d '\r' | head -1)
    [ -n "$pid" ] && { log "22:45 priority rule: stopping P50C5 pid=$pid for C6"; powershell -NoProfile -Command "Stop-Process -Id $pid -Force"; rm -f ../.gpu_busy; }
  fi
  late && { log "deadline reached before chain2 done"; break; }
  sleep 15
done
until [ -f "$M" ]; do late && { log "mix exemplars missing at deadline"; exit 0; }; sleep 15; done
log "mix ids: $($PY -c "import json;print(len(json.load(open('$M'))))")"
late || { $PY scripts/poc_rnsd.py --n 50 --conds L2 --exemplars "$M" --tag _P50C6 > runs/poc/E2B_runP50C6.log 2>&1; log "P50C6 rc=$?"; }
$PY scripts/poc_eval.py --model E2B > runs/poc/eval3.log 2>&1; log "eval rc=$?"
$PY scripts/poc_table.py E2B >> runs/poc/eval3.log 2>&1
CI="L2_P50C1 L2_P50C4"
[ -f runs/poc/E2B_L2_P50C6.jsonl ] && CI="$CI L2_P50C6"
[ -f runs/poc/E2B_L2_P50C5.jsonl ] && [ "$(grep -c '"id"' runs/poc/E2B_L2_P50C5.jsonl)" -ge 50 ] && CI="$CI L2_P50C5"
$PY scripts/poc_ci.py E2B $CI >> runs/poc/eval3.log 2>&1; log "ci rc=$? ($CI)"
log "chain3 done"
