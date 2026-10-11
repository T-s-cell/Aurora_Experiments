#!/bin/bash
# Watcher (theta): wait for GPU1 free VRAM >= 6 GiB, then preflight;
# ONLY on preflight PASS: pack preflight evidence, then start the full run.
# Per the standing ruling: preflight PASS -> evidence saved -> full run
# starts automatically; any preflight failure blocks the full run.
cd "$(dirname "$0")"
mkdir -p logs
TOKEN=9c164222d4b9
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  echo "$(date '+%F %T') GPU1 free ${FREE} MiB" >> logs/watcher.log
  if [ "$FREE" -ge 6000 ]; then break; fi
  sleep 120
done
echo "$(date '+%F %T') GPU1 free enough -> preflight" >> logs/watcher.log
./wrapper_theta.sh preflight.py --i-approve-frozen-protocol "$TOKEN" \
  > logs/preflight.log 2>&1
RC=$?
echo "$(date '+%F %T') preflight exit=$RC" >> logs/watcher.log
if [ "$RC" -ne 0 ]; then
  echo "$(date '+%F %T') PREFLIGHT FAILED - full run blocked" >> logs/watcher.log
  exit 1
fi
./wrapper_theta.sh audit_pack.py --stage preflight \
  > logs/pack_preflight.log 2>&1
echo "$(date '+%F %T') preflight evidence packed; starting full run" \
  >> logs/watcher.log
./wrapper_theta.sh run_infer.py --i-approve-frozen-protocol "$TOKEN" \
  > logs/infer.log 2>&1
echo "$(date '+%F %T') infer exit=$?" >> logs/watcher.log
