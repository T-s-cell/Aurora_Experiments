#!/bin/bash
# Preflight watcher: wait for a free GPU, run all preflight gates, freeze
# the protocol (batch plan measured first), build the PREFLIGHT audit pack
# and STOP. The 36-group full run starts only after the user reviews that
# pack and explicitly approves.
set -u
cd "$(dirname "$0")"
export PATH=/usr/lib/wsl/lib:$PATH
mkdir -p logs results

log() { echo "[watcher $(date +%H:%M:%S)] $*" | tee -a logs/watcher.log; }

wait_gpu() {
    while true; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [ "${used:-99999}" -lt 2500 ]; then
            log "GPU free (used=${used}MiB)"
            return 0
        fi
        sleep 60
    done
}

wait_gpu

log "env snapshot (pip freeze both envs + diff)"
/home/wlt/miniconda3/envs/aurora_tmmd_v1/bin/pip freeze > logs/pip_freeze_aurora_tmmd_v1.txt 2>/dev/null
/home/wlt/miniconda3/envs/torch_50_series/bin/pip freeze > logs/pip_freeze_torch_50_series.txt 2>/dev/null
diff logs/pip_freeze_torch_50_series.txt logs/pip_freeze_aurora_tmmd_v1.txt > logs/env_diff.txt || true
for pkg in torch torchvision numpy; do
    v_base=$(grep -m1 "^${pkg}==" logs/pip_freeze_torch_50_series.txt || true)
    v_clone=$(grep -m1 "^${pkg}==" logs/pip_freeze_aurora_tmmd_v1.txt || true)
    if [ "$v_base" != "$v_clone" ]; then
        log "FATAL: $pkg changed in clone env: base='${v_base}' clone='${v_clone}'"
        exit 1
    fi
    log "env guard OK: $v_clone"
done

log "verify_refs (pre-GPU gates)"
./wrapper_dt.sh verify_refs.py 2>&1 | tee logs/verify_refs.log
if ! grep -q "RESULT: PASS" logs/verify_refs.log; then
    log "verify_refs FAILED — stopping"; exit 1
fi

log "preflight (data gate -> batch probes -> freeze -> Security:12 mini group)"
./wrapper_dt.sh preflight.py 2>&1 | tee logs/preflight.log
if ! grep -q "RESULT: PASS" logs/preflight.log; then
    log "preflight FAILED — stopping"; exit 1
fi

TOKEN=$(./wrapper_dt.sh -c "import json;print(json.load(open('results/preflight_report.json'))['protocol_token'])")
log "frozen protocol token: $TOKEN"

log "building PREFLIGHT audit pack"
./wrapper_dt.sh audit_pack.py --stage preflight 2>&1 | tee logs/audit_pack_preflight.log
PACK=$(ls -t pack_out/Aurora_TimeMMD_zeroshot_preflight_*.tar.gz 2>/dev/null | head -1)
if [ -z "${PACK:-}" ]; then
    log "audit pack MISSING — stopping"; exit 1
fi
log "preflight pack: $PACK"

log "STOPPING FOR USER REVIEW. After approval run:"
log "  ./wrapper_dt.sh run_all.py --i-approve-frozen-protocol $TOKEN"
log "  ./wrapper_dt.sh evaluate.py"
log "  ./wrapper_dt.sh audit_pack.py --stage final"
