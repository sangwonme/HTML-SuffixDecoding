#!/usr/bin/env bash
# F: external generalization (WebSight v0.2 eval 50), {PLD, DOM, RANKER} x 1 rep, after the main GPU queue.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe -I"
O=runs/poc/F_websight
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $O/chain.log; }
log "waiting for 'queue done'"
until grep -q "queue done" runs/poc/queue.log 2>/dev/null; do sleep 30; done
W=runs/study/ranker_mlp_trainsplit.npz
NOTE="trainsplit"
if [ ! -f "$W" ]; then W=runs/study/ranker_mlp_fold0.npz; NOTE="FALLBACK cross-fit fold0 (trainsplit weights missing)"; fi
log "GPU start F (ranker weights: $W, $NOTE)"
$PY scripts/poc_rnsd.py --prompts-file runs/retrieval/external_websight_prompts.jsonl --n 50 --conds pld,dom,ranker \
    --reps 1 --shuffle --ranker-weights "$W" --outdir $O > $O/run.log 2>&1
log "F rc=$?"
rm -f ../.gpu_busy
log "GPU released"
$PY scripts/gate2_analyze.py $O > $O/analyze.log 2>&1; log "analyze rc=$?"
$PY scripts/websight_eval.py $O > $O/quality.log 2>&1; log "quality rc=$?"
log "F done"
