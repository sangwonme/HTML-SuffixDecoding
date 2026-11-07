#!/usr/bin/env bash
# Sequential pilot chain (concurrency 1). Holds ../.gpu_busy for the whole chain (removed on exit).
cd "$(dirname "$0")/.." || exit 1
PY=.venv/Scripts/python.exe
N=${N:-10}
FLAG=../.gpu_busy
echo "GenerativeUI Gemma pilot chain pid=$$ since $(date -Is)" > "$FLAG"
trap 'rm -f "$FLAG"' EXIT
export GENUI_CHAIN=1
SPECS=${SPECS:-"E2B:rich_zs E4B:rich_zs 12B:rich_zs E2B:short_zs E2B:rich_fs"}
for spec in $SPECS; do
  m=${spec%%:*}; c=${spec##*:}
  echo "=== START $m $c n=$N $(date -Is)"
  $PY -I scripts/bench_gemma.py --model "$m" --cond "$c" --n "$N" 2>&1 | grep --line-buffered -v -i -E "triton|redirects|Loading weights|expandable_segments|torch.empty"
  echo "=== END $m $c rc=${PIPESTATUS[0]} $(date -Is)"
done
