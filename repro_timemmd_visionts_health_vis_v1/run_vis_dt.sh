#!/bin/bash
# Single-stage driver for repro_timemmd_visionts_health_vis_v1 (EXP-023)
# on dt: env guard -> verify_and_select (all gates + selection) ->
# plot (10 PNGs) -> report (README) -> audit pack.
# No model inference, no GPU wait. Any failing step stops with the
# scene preserved. No git actions here — pack goes back to the user
# for review first.
set -euo pipefail
cd "$(dirname "$0")"
export PATH=/usr/lib/wsl/lib:$PATH
mkdir -p logs results figures pack_out

PYBIN=/home/wlt/miniconda3/envs/diag_vts_env/bin/python

log() { echo "[driver $(date +%H:%M:%S)] $*" | tee -a logs/watcher.log; }

env_guard() {
    "$PYBIN" -m pip freeze > logs/pip_freeze_before.txt 2>/dev/null
    local v
    for pin in "python==3.10.20" "numpy==2.2.6" "scikit-learn==1.7.2"; do
        case "$pin" in
            python==*) v=$("$PYBIN" -c 'import platform;print("python=="+platform.python_version())') ;;
            *) v=$(grep -m1 "^${pin}$" logs/pip_freeze_before.txt || true) ;;
        esac
        if [ "$v" != "$pin" ]; then
            log "FATAL: env guard: got '${v:-absent}' want $pin"; exit 1
        fi
        log "env guard OK: $pin"
    done
    if ! "$PYBIN" -c "import matplotlib" 2>/dev/null; then
        log "matplotlib missing -> pip install 'matplotlib>=3.10,<4'"
        "$PYBIN" -m pip install "matplotlib>=3.10,<4" \
            > logs/pip_install_matplotlib.log 2>&1 || {
            log "FATAL: matplotlib install failed — see logs/"; exit 1; }
        log "matplotlib installed"
    fi
    "$PYBIN" -m pip freeze > logs/pip_freeze_after.txt 2>/dev/null
    v=$("$PYBIN" -c "import matplotlib;print(matplotlib.__version__)")
    log "env guard OK: matplotlib==$v"
}

if [ "${1:-}" != "all" ]; then
    echo "usage: $0 all" >&2
    exit 1
fi
t0=$(date +%s)

log "env guard (diag_vts_env pins + matplotlib provision)"
env_guard

log "verify_and_select (EXP-021 anchor gates + per-window metrics + selection)"
./wrapper_dt.sh verify_and_select.py 2>&1 | tee logs/verify_select.log
grep -q "VERIFY_SELECT OK" logs/verify_select.log || {
    log "verify_and_select FAILED — stopping"; exit 1; }

log "plot (10 two-panel PNGs)"
./wrapper_dt.sh plot_windows.py 2>&1 | tee logs/plot.log
grep -q "PLOT OK" logs/plot.log || { log "plot FAILED — stopping"; exit 1; }

log "report (README.md from actual run values)"
./wrapper_dt.sh make_report.py 2>&1 | tee logs/report.log
grep -q "REPORT OK" logs/report.log || {
    log "report FAILED — stopping"; exit 1; }

log "audit pack (single bundle, <50MB, MANIFEST + sources)"
./wrapper_dt.sh audit_pack.py 2>&1 | tee logs/pack.log
grep -q "AUDIT_PACK OK" logs/pack.log || {
    log "pack FAILED — stopping"; exit 1; }

log "ALL DONE in $(( $(date +%s) - t0 ))s — pack awaits user review (no git actions)"
