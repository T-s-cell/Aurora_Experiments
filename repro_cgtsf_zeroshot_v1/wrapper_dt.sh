#!/bin/bash
# Execution wrapper for dt GPU0 (RTX 5060 Ti, torch_50_series env). All
# entry points (preflight / run_infer / verify / evaluate) MUST run through
# this so the determinism-critical environment is pinned before Python
# starts. NOTE: dt runs torch 2.13.0+cu129 (sm_120) per user instruction —
# not version-pinned to theta/iota; the frozen protocol's machine-specific
# runtime guards are self-consistent per server via its own preflight.
set -euo pipefail
PY=/home/wlt/miniconda3/envs/torch_50_series/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
cd "$(dirname "$0")"
exec "$PY" "$@"
