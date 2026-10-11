#!/bin/bash
# Watcher (dt GPU0, RTX 5060 Ti, exclusive): parallel copy of the frozen
# run. Same frozen protocol/token; preflight PASS -> evidence kept -> full
# run starts automatically (standing ruling). No audit_pack step here (the
# frozen-code redaction false positive aborts it; evidence is packed
# locally from pulled artifacts instead). Shards land in this directory's
# predictions/ and are pulled back to local predictions_dt/.
cd "$(dirname "$0")"
mkdir -p logs
# WSL2: nvidia-smi lives outside the default non-interactive ssh PATH.
export PATH="/usr/lib/wsl/lib:$PATH"
TOKEN=9c164222d4b9
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  echo "$(date '+%F %T') GPU0 free ${FREE} MiB" >> logs/watcher.log
  if [ "$FREE" -ge 6000 ]; then break; fi
  sleep 120
done
echo "$(date '+%F %T') GPU0 free enough -> preflight" >> logs/watcher.log
./wrapper_dt.sh preflight.py --i-approve-frozen-protocol "$TOKEN" \
  > logs/preflight.log 2>&1
RC=$?
echo "$(date '+%F %T') preflight exit=$RC" >> logs/watcher.log
if [ "$RC" -ne 0 ]; then
  echo "$(date '+%F %T') PREFLIGHT FAILED - full run blocked" >> logs/watcher.log
  exit 1
fi
echo "$(date '+%F %T') preflight evidence kept (pack locally); starting full run" \
  >> logs/watcher.log
./wrapper_dt.sh run_infer.py --i-approve-frozen-protocol "$TOKEN" \
  > logs/infer.log 2>&1
echo "$(date '+%F %T') infer exit=$?" >> logs/watcher.log
