#!/usr/bin/env bash
# CPU analyses that follow the GPU queue: quality (checklist, element F1) per finished run dir.
# Runs at BelowNormal-equivalent cadence (one at a time); does not touch the GPU.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe -I"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a runs/poc/analysis.log; }
done_dir() { [ -f "$1/run.log" ] && grep -q "^done" "$1/run.log"; }
todo="runs/poc/p1A/fmt runs/poc/p1A/dom runs/poc/p1A/cls runs/poc/p1A/css runs/poc/p1B/fmt runs/poc/p1B/css runs/poc/p1B/dom runs/poc/p1B/cls runs/poc/E/test100"
for d in $todo; do
  until done_dir "$d"; do sleep 30; done
  if [ "$d" = "runs/poc/E/test100" ]; then
    $PY scripts/quality_eval.py $d runs/poc/E/test100_rows.jsonl data/processed/test.jsonl > $d/quality.log 2>&1
  else
    $PY scripts/quality_eval.py $d > $d/quality.log 2>&1
  fi
  log "quality $d rc=$?"
done
log "analysis watch done"
