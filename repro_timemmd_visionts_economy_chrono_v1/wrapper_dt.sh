#!/bin/bash
# Common environment for repro_timemmd_visionts_economy_chrono_v1 on dt
# (WSL2, RTX 5060 Ti, conda env diag_vts_env verified in EXP-018).
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
