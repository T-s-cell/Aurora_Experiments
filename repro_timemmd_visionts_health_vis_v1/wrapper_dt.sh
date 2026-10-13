#!/bin/bash
# Common environment for repro_timemmd_visionts_health_vis_v1 on dt
# (same pattern as EXP-021; no GPU work happens here but exports are
# kept for env parity with the frozen experiment lineage).
set -u
cd "$(dirname "$0")"

PY=/home/wlt/miniconda3/envs/diag_vts_env/bin/python
if [ ! -x "$PY" ]; then
    echo "[wrapper] missing $PY" >&2
    exit 1
fi

# WSL2: non-interactive ssh PATH lacks nvidia-smi
export PATH=/usr/lib/wsl/lib:$PATH
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "$PY" "$@"
