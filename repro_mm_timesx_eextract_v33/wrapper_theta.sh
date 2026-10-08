#!/usr/bin/env bash
# theta-side wrapper for the EXP-013 (M48T512_E33) round.
# Usage: bash wrapper_theta.sh <step>
#   budgets | compress | assemble | freeze | preflight-cpu | preflight | eval-dry | eval
#
# The LLM API key is injected ONLY via the environment (TIMESX_LLM_API_KEY,
# read from the 0600 raw-key file ~/.timesx_llm_api_key outside the repo).
# The key never appears in code, command lines, logs, git or audit packs.
set -euo pipefail
SUBDIR="$(cd "$(dirname "$0")" && pwd)"
PY=/dev_data/wlt/conda/envs/aurora/bin/python

export CUDA_VISIBLE_DEVICES=1
export CUBLAS_WORKSPACE_CONFIG=':4096:8'
export PYTHONHASHSEED=0
if [ -z "${TIMESX_LLM_API_KEY:-}" ] && [ -f "$HOME/.timesx_llm_api_key" ]; then
  export TIMESX_LLM_API_KEY="$(tr -d '\r\n' < "$HOME/.timesx_llm_api_key")"
fi
if [ -z "${TIMESX_LLM_API_KEY:-}" ]; then
  echo "[wrapper] TIMESX_LLM_API_KEY not set and ~/.timesx_llm_api_key unreadable" >&2
  exit 1
fi

case "${1:-}" in
  budgets)       cd "$SUBDIR" && exec "$PY" build_budgets_e33.py ;;
  compress)      cd "$SUBDIR" && exec "$PY" build_cache_e33.py --phase compress ;;
  assemble)      cd "$SUBDIR" && exec "$PY" build_cache_e33.py --phase assemble ;;
  freeze)        cd "$SUBDIR" && exec "$PY" build_cache_e33.py --phase freeze ;;
  preflight-cpu) cd "$SUBDIR" && exec "$PY" run_preflight_e33.py --skip-gpu ;;
  preflight)     cd "$SUBDIR" && exec "$PY" run_preflight_e33.py ;;
  eval-dry)      cd "$SUBDIR" && exec "$PY" run_eval_e33.py --dry-run ;;
  eval)
    # official run: bash wrapper_theta.sh eval <first 12 hex of sha256(protocol_e33.json)>
    TOKEN="${2:-}"
    if [ -z "$TOKEN" ]; then
      echo "[wrapper] eval needs the frozen-protocol approval token as arg 2" >&2
      exit 2
    fi
    cd "$SUBDIR" && exec "$PY" run_eval_e33.py --itl 48 --i-approve-frozen-protocol "$TOKEN" ;;
  *) echo "usage: bash wrapper_theta.sh <budgets|compress|assemble|freeze|preflight-cpu|preflight|eval-dry|eval>" >&2; exit 2 ;;
esac
