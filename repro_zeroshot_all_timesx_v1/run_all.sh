#!/usr/bin/env bash
# Stage runner. Each stage runs only when its .state marker is stale
# (stage_guard re-verifies output sha256 + declared input/code fingerprints).
# FORCE_<STAGE_UPPER>=1 forces a re-run (e.g. FORCE_FILL=1 bash run_all.sh fill).
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/wlt/miniconda3/envs/aurora_test/bin/python}

donefile() {  # keep in sync with stage_guard.py::MARKER
  case "$1" in
    freeze_ref) echo s1a_freeze_ref ;;
    verify)     echo s1b_verify_cache ;;
    sample)     echo s2_sample_frozen ;;
    preflight)  echo s3_preflight ;;
    fill)       echo s4_infer_fill ;;
    assemble)   echo s5a_assemble ;;
    evaluate)   echo s5b_evaluate ;;
    recompute)  echo recompute ;;
    accept)     echo accept ;;
    *)          return 1 ;;
  esac
}

force_flag() {  # FORCE_<UPPER-with-_->  (tr would also map '-' -> '_', so build explicitly)
  echo "FORCE_$(echo "$1" | tr '[:lower:]' '[:upper:]')"
}

run_stage() {
  local stage="$1"; shift
  local marker fv force overwrite
  marker="$(donefile "$stage")" || { echo "unknown stage: $stage"; return 2; }
  fv="$(force_flag "$stage")"
  force="${!fv:-0}"
  if [ -f ".state/${marker}.done" ] && [ "$force" != "1" ]; then
    if "$PY" stage_guard.py "$stage"; then
      echo "[run_all] $stage: up-to-date, skip"
      return 0
    fi
    echo "[run_all] $stage: stale state (inputs/code/outputs changed) — re-running"
    overwrite="--force"
  else
    if [ "$force" = "1" ]; then
      echo "[run_all] $stage: forced re-run (${fv}=1)"
    fi
    overwrite="--force"
  fi
  echo "[run_all] $stage: running"
  case "$stage" in
    freeze_ref) "$PY" freeze_ref.py $overwrite ;;
    verify)     "$PY" verify_cache.py $overwrite ;;
    sample)     "$PY" preflight.py --sample-only $overwrite ;;
    assemble)   "$PY" assemble.py $overwrite ;;
    evaluate)   "$PY" evaluate.py $overwrite ;;
    recompute)  "$PY" recompute_check.py ;;
    accept)     "$PY" verify_acceptance.py ;;
    *) echo "stage $stage runs on the GPU host (see RUN_COMMANDS.md)"; return 2 ;;
  esac
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[run_all] $stage: FAILED (rc=$rc) — stopping"
    return $rc
  fi
  echo "[run_all] $stage: OK"
}

main() {
  local what="${1:-all}"
  case "$what" in
    local)   run_stage freeze_ref && run_stage verify && run_stage sample ;;
    post)    run_stage assemble && run_stage evaluate && run_stage recompute && run_stage accept ;;
    all)     run_stage freeze_ref && run_stage verify && run_stage sample ;;
    *)       run_stage "$what" ;;
  esac
}

main "$@"
