#!/bin/bash
# Execution wrapper for theta GPU1 (aurora env). All entry points
# (preflight / run_infer / verify / evaluate) MUST run through this so the
# determinism-critical environment is pinned before Python starts.
set -euo pipefail
PY=/dev_data/wlt/conda/envs/aurora/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")"
exec "$PY" "$@"
