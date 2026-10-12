#!/bin/bash
# Common environment for repro_timemmd_economy_chrono_v1 on dt (WSL2, RTX 5060 Ti).
set -u
cd "$(dirname "$0")"

PY=/home/wlt/miniconda3/envs/aurora_tmmd_v1/bin/python
if [ ! -x "$PY" ]; then
    echo "[wrapper] missing $PY — clone torch_50_series first" >&2
    exit 1
fi

# WSL2: non-interactive ssh PATH lacks nvidia-smi
export PATH=/usr/lib/wsl/lib:$PATH
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export AURORA_HF_SNAPSHOT=/home/wlt/.cache/huggingface/hub/models--DecisionIntelligence--Aurora/snapshots/c495b02c1b151be52a3c174237ed240aa66e6384

if [ ! -d "$AURORA_HF_SNAPSHOT" ]; then
    echo "[wrapper] HF snapshot missing: $AURORA_HF_SNAPSHOT" >&2
    exit 1
fi

exec "$PY" "$@"
