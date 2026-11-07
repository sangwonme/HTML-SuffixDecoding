#!/usr/bin/env bash
# Night GPU queue (sequential). Order: P1-A finish (dom, cls via p1A_chain) -> css(A) -> yield to Parkinson
# (wait for ../.gpu_parkinson_done newer than yield start, max 45 min) -> P1-B x4 -> E(i) train150 PLD -> wait for
# trainsplit ranker weights -> E(ii) test100 {PLD, DOM, RANKER}. CPU analyses run during the yield window.
cd "$(dirname "$0")/.." || exit 1
PY=".venv/Scripts/python.exe -I"
Q=runs/poc/queue.log
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $Q; }
log "queue waiting for p1A chain (dom, cls)"
until grep -q "p1A done" runs/poc/p1A/chain.log 2>/dev/null; do sleep 15; done
# ---- 1. css (condition A), fresh run; keep the partial stopped run
[ -d runs/poc/p1A/css ] && mv runs/poc/p1A/css runs/poc/p1A/css_partial_stopped
mkdir -p runs/poc/p1A/css
log "GPU start P1-A css"
$PY scripts/poc_rnsd.py --n 50 --conds pld,dom,ranker --reps 1 --shuffle \
    --exemplar-html runs/retrieval/pilot_exemplar_html_css.json --outdir runs/poc/p1A/css > runs/poc/p1A/css/run.log 2>&1
log "P1-A css rc=$?"
$PY scripts/gate2_analyze.py runs/poc/p1A/css > runs/poc/p1A/css/analyze.log 2>&1
log "P1-A complete (4 transforms)"
# ---- 2. yield GPU to Parkinson
rm -f ../.gpu_busy
YS=$(date +%s)
log "GPU YIELD to Parkinson (waiting for ../.gpu_parkinson_done, max 45 min)"
( $PY scripts/p1A_eval.py > runs/poc/p1A/eval_all.log 2>&1; \
  $PY scripts/p1A_cause.py dom cls css > runs/poc/p1A/cause_rest.log 2>&1; echo "[$(date +%H:%M:%S)] CPU analyses (eval, cause dom/cls/css) done" >> $Q ) &
while true; do
  if [ -f ../.gpu_parkinson_done ] && [ "$(stat -c %Y ../.gpu_parkinson_done)" -ge "$YS" ]; then log "Parkinson done signal"; break; fi
  [ $(( $(date +%s) - YS )) -ge 2700 ] && { log "yield timeout 45 min, resuming"; break; }
  sleep 20
done
wait
log "GPU RESUME"
# ---- 3. P1-B (draft source only), 4 transforms
for t in fmt css dom cls; do
  mkdir -p runs/poc/p1B/$t
  log "GPU start P1-B $t"
  $PY scripts/poc_rnsd.py --n 50 --conds pld,dom,ranker --reps 1 --shuffle \
      --draft-source-html runs/retrieval/pilot_exemplar_html_$t.json --outdir runs/poc/p1B/$t > runs/poc/p1B/$t/run.log 2>&1
  log "P1-B $t rc=$?"
  $PY scripts/gate2_analyze.py runs/poc/p1B/$t > runs/poc/p1B/$t/analyze.log 2>&1
done
log "P1-B complete"
# ---- 4. E(i) training outputs: 150 train requests (apps disjoint from pilot/dev/test), PLD greedy
mkdir -p runs/poc/E/train150
log "GPU start E(i) train150 PLD"
$PY scripts/poc_rnsd.py --rows runs/poc/E/train150_rows.jsonl --n 150 --conds pld --reps 1 \
    --outdir runs/poc/E/train150 > runs/poc/E/train150/run.log 2>&1
log "E(i) rc=$? -> runs/poc/E/train150/E2B_pld.jsonl (for study retraining)"
rm -f ../.gpu_busy
log "GPU idle: waiting for runs/study/ranker_mlp_trainsplit.npz"
until [ -f runs/study/ranker_mlp_trainsplit.npz ]; do sleep 30; done
sleep 10
# ---- 5. E(ii) independent test: 100 test requests (apps disjoint from pilot), {PLD, DOM, RANKER trainsplit}
mkdir -p runs/poc/E/test100
log "GPU start E(ii) test100"
$PY scripts/poc_rnsd.py --rows runs/poc/E/test100_rows.jsonl --n 100 --conds pld,dom,ranker --reps 1 --shuffle \
    --ranker-weights runs/study/ranker_mlp_trainsplit.npz --outdir runs/poc/E/test100 > runs/poc/E/test100/run.log 2>&1
log "E(ii) rc=$?"
rm -f ../.gpu_busy
log "GPU released (queue finished)"
$PY scripts/gate2_analyze.py runs/poc/E/test100 > runs/poc/E/test100/analyze.log 2>&1
log "queue done"
