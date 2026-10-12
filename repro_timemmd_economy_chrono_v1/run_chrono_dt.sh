#!/bin/bash
# One-shot driver for repro_timemmd_economy_chrono_v1 (EXP-020) on dt.
# Chain: wait GPU -> env guard -> prepare_data (sort) -> verify_refs ->
# preflight (data gate -> batch probe -> freeze) -> FIRST OFFICIAL GROUP
# Economy:12 -> remaining groups Economy:6/8/10 -> evaluate -> FINAL audit
# pack -> STOP (awaiting the user's freeze review; no git actions here).
set -u
cd "$(dirname "$0")"
export PATH=/usr/lib/wsl/lib:$PATH
mkdir -p logs results

log() { echo "[driver $(date +%H:%M:%S)] $*" | tee -a logs/watcher.log; }

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

log "prepare_data (chronological sort + inverse-perm proof)"
./wrapper_dt.sh prepare_data.py 2>&1 | tee logs/prepare_data.log
if ! grep -q "RESULT: PASS" logs/prepare_data.log; then
    log "prepare_data FAILED — stopping"; exit 1
fi

log "verify_refs (pre-GPU gates)"
./wrapper_dt.sh verify_refs.py 2>&1 | tee logs/verify_refs.log
if ! grep -q "RESULT: PASS" logs/verify_refs.log; then
    log "verify_refs FAILED — stopping"; exit 1
fi

log "preflight (data gate -> Economy batch probe -> freeze)"
./wrapper_dt.sh preflight.py 2>&1 | tee logs/preflight.log
if ! grep -q "RESULT: PASS" logs/preflight.log; then
    log "preflight FAILED — stopping"; exit 1
fi

TOKEN=$(./wrapper_dt.sh -c "import json;print(json.load(open('results/preflight_report.json'))['protocol_token'])")
log "frozen protocol token: $TOKEN"

log "FIRST OFFICIAL GROUP: Economy h=12"
./wrapper_dt.sh run_all.py --i-approve-frozen-protocol "$TOKEN" --only Economy:12 2>&1 | tee logs/run_all_h12.log
if ! test -f predictions/Economy__h12.done.json; then
    log "first official group Economy:12 FAILED — stopping"; exit 1
fi
log "Economy:12 OK — running remaining groups 6/8/10"
./wrapper_dt.sh run_all.py --i-approve-frozen-protocol "$TOKEN" 2>&1 | tee logs/run_all_rest.log
if ! grep -q "ALL 4 GROUPS OK" logs/run_all_rest.log; then
    log "remaining groups FAILED — stopping"; exit 1
fi

log "evaluate (metrics + chrono-vs-orig comparison)"
./wrapper_dt.sh evaluate.py 2>&1 | tee logs/evaluate.log
if ! grep -q "groups done: 4/4" logs/evaluate.log; then
    log "evaluate FAILED — stopping"; exit 1
fi

log "building FINAL audit pack"
./wrapper_dt.sh audit_pack.py --stage final 2>&1 | tee logs/audit_pack_final.log
PACK=$(ls -t pack_out/Aurora_TimeMMD_economy_chrono_final_*.tar.gz 2>/dev/null | head -1)
if [ -z "${PACK:-}" ]; then
    log "final audit pack MISSING — stopping"; exit 1
fi
log "final pack: $PACK"

log "ALL DONE. Awaiting the user's freeze review (no git actions)."
