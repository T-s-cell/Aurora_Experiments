#!/bin/bash
# Single-stage driver for repro_timemmd_visionts_economy_chrono_v1
# (EXP-022) on dt. Per the user's instruction there is NO mid-run
# approval: gates -> select P -> freeze -> all 4 groups -> evaluate ->
# FINAL audit packs -> STOP "awaiting the user's freeze review".
#
# all: wait GPU -> env guard -> prepare_data -> verify_refs -> preflight
#      (data gates -> weights gate -> geometry -> batch probe -> smoke)
#      -> select_period (GPU, 16 combos) -> protocol freeze -> run_all
#      (4 groups, identity gates vs EXP-020) -> evaluate (three-way
#      comparison + anchor mean gates) -> FINAL audit packs (main +
#      val evidence) -> STOP (no git actions here).
set -u
cd "$(dirname "$0")"
export PATH=/usr/lib/wsl/lib:$PATH
mkdir -p logs results

STAGE="${1:-}"
if [ "$STAGE" != "all" ]; then
    echo "usage: $0 all" >&2
    exit 1
fi

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

env_guard() {
    local PYBIN=/home/wlt/miniconda3/envs/diag_vts_env/bin/python
    "$PYBIN" -m pip freeze > logs/pip_freeze_diag_vts_env.txt 2>/dev/null
    local v
    v=$("$PYBIN" -c 'import platform;print("python=="+platform.python_version())')
    if [ "$v" != "python==3.10.20" ]; then
        log "FATAL: env guard: got $v want python==3.10.20"; exit 1
    fi
    log "env guard OK: $v"
    for pin in torch torchvision numpy timm einops scikit-learn; do
        v=$(grep -im1 "^${pin}==" logs/pip_freeze_diag_vts_env.txt || true)
        if [ -z "$v" ]; then
            log "FATAL: env guard: $pin not found in pip freeze"; exit 1
        fi
        log "env guard OK: $v"
    done
    grep -q "^torch==2.13.0+cu129$" logs/pip_freeze_diag_vts_env.txt || { log "FATAL: env guard: torch pin"; exit 1; }
    grep -q "^numpy==2.2.6$" logs/pip_freeze_diag_vts_env.txt || { log "FATAL: env guard: numpy pin"; exit 1; }
    "$PYBIN" -c "import sklearn" 2>/dev/null || { log "FATAL: env guard: scikit-learn missing in diag_vts_env"; exit 1; }
    log "env guard OK: scikit-learn importable"
}

wait_gpu
log "env guard (diag_vts_env pinned versions)"
env_guard

log "prepare_data (copy EXP-020 ascending Economy.csv + sort evidence + gates)"
./wrapper_dt.sh prepare_data.py 2>&1 | tee logs/prepare_data.log
if ! grep -q "data_prep.json written" logs/prepare_data.log; then
    log "prepare_data FAILED — stopping"; exit 1
fi

log "verify_refs (dual protocol anchors + CSV/mapping/original + md5/weights + anchor npz vs frozen done records)"
./wrapper_dt.sh verify_refs.py 2>&1 | tee logs/verify_refs.log
if ! grep -q "verify. ALL OK" logs/verify_refs.log; then
    log "verify_refs FAILED — stopping"; exit 1
fi

log "preflight (data gates -> weights gate -> geometry 16 combos -> batch probe -> smoke)"
./wrapper_dt.sh preflight.py 2>&1 | tee logs/preflight.log
if ! grep -q "PREFLIGHT OK" logs/preflight.log; then
    log "preflight FAILED — stopping"; exit 1
fi

log "select_period (GPU, candidates 1/3/6/12 x H 6/8/10/12, ascending val split)"
./wrapper_dt.sh select_period.py 2>&1 | tee logs/select_period.log
if ! grep -q "SELECT OK" logs/select_period.log; then
    log "select_period FAILED — stopping"; exit 1
fi

log "protocol freeze (embeds selected_p sha256 + argmin recheck)"
./wrapper_dt.sh protocol.py freeze 2>&1 | tee logs/freeze.log
if ! grep -q "frozen token" logs/freeze.log; then
    log "freeze FAILED — stopping"; exit 1
fi
TOKEN=$(./wrapper_dt.sh -c "import hashlib;print(hashlib.sha256(open('protocol.json','rb').read()).hexdigest()[:12])")
./wrapper_dt.sh protocol.py verify "$TOKEN" 2>&1 | tee -a logs/freeze.log
log "frozen protocol token: $TOKEN"

log "run_all (4 groups, fresh process per group, identity gates vs EXP-020)"
./wrapper_dt.sh run_all.py 2>&1 | tee logs/run_all.log
if ! grep -q "RUN_ALL OK" logs/run_all.log; then
    log "run_all FAILED — stopping"; exit 1
fi

log "evaluate (metrics + three-way comparison + anchor 4H-mean gates)"
./wrapper_dt.sh evaluate.py 2>&1 | tee logs/evaluate.log
if ! grep -q "EVALUATE OK" logs/evaluate.log; then
    log "evaluate FAILED — stopping"; exit 1
fi

log "building FINAL audit packs (main + val evidence)"
./wrapper_dt.sh audit_pack.py --stage final 2>&1 | tee logs/audit_pack_final.log
if ! grep -q "AUDIT_PACK OK" logs/audit_pack_final.log; then
    log "final audit pack FAILED — stopping"; exit 1
fi

log "ALL COMPLETE. STOP: awaiting the user's freeze review (no git actions)."
