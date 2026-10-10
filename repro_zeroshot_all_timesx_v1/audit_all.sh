#!/usr/bin/env bash
# Audit packer. Usage: bash audit_all.sh --preexec | --final
# Packs to /home/wlt/MMTS/temp/VisionTS_aurora_zeroshot_all_timesx_v1_{preexec|final}_audit_<ts>.tar.gz
# and asserts size < 50 MB. --preexec: code + frozen refs + S1/S2 evidence for
# user review BEFORE any GPU run. --final: everything incl. predictions.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
TEMP_DIR=/home/wlt/MMTS/temp
MODE="$1"
TS=$(date +%Y%m%d_%H%M%S)
STAGE="$([ "$MODE" = "--preexec" ] && echo preexec || echo final)"
OUT="$TEMP_DIR/VisionTS_aurora_zeroshot_all_timesx_v1_${STAGE}_audit_${TS}.tar.gz"
mkdir -p "$TEMP_DIR"

MEMBERS=(README.md RUN_COMMANDS.md DRYRUN_EVIDENCE.md protocol.py common.py
  stage_guard.py freeze_ref.py verify_cache.py preflight.py infer_fill.py
  assemble.py evaluate.py recompute_check.py verify_acceptance.py
  make_report.py run_all.sh audit_all.sh visionts_ref reuse_plan.json
  preflight_sample.json .state logs)

if [ "$MODE" != "--preexec" ] && [ "$MODE" != "--final" ]; then
  echo "usage: bash audit_all.sh --preexec|--final" >&2; exit 2
fi

in_members() {
  local f
  for f in "${MEMBERS[@]}"; do
    [ "$f" = "$1" ] && return 0
  done
  return 1
}

# final pack additionally carries the GPU verdicts, new cache and results
[ "$MODE" = "--final" ] && MEMBERS+=(preflight_report.json reuse_final.json
  cache_all results REPORT.md)
if [ "$MODE" = "--final" ]; then
  for f in preflight_report.json reuse_final.json cache_all results REPORT.md; do
    in_members "$f" || { echo "$f not in MEMBERS" >&2; exit 1; }
  done
fi

if [ "$MODE" = "--preexec" ]; then
  for f in visionts_ref/inventory.json visionts_ref/vars.json \
           visionts_ref/frozen_arrays.npz visionts_ref/preds.jsonl.gz \
           visionts_ref/split_manifest.json \
           reuse_plan.json preflight_sample.json .state/s1a_freeze_ref.done \
           .state/s1b_verify_cache.done .state/s2_sample_frozen.done; do
    [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }
  done
else
  for f in results/aurora_preds.jsonl.gz results/acceptance.json \
           results/recompute_check.json results/vts_gate.json \
           preflight_report.json reuse_final.json cache_all/manifest.json \
           REPORT.md; do
    [ -f "$f" ] || { echo "missing $f — run the pipeline first" >&2; exit 1; }
  done
fi

rm -rf /tmp/au_pack && mkdir -p /tmp/au_pack
for m in "${MEMBERS[@]}"; do
  if [ -e "$m" ]; then
    cp -r "$m" /tmp/au_pack/
  elif [ "$MODE" = "--final" ]; then
    echo "missing required final-pack member: $m" >&2; exit 1
  fi
done
# keep only evidence files from logs
find /tmp/au_pack/logs -type f ! -name '*.txt' ! -name '*.json' \
  ! -name '*.jsonl.gz' ! -name '*.log' -delete 2>/dev/null || true

( cd /tmp/au_pack && find . -type f ! -name SHA256SUMS -print0 | sort -z | \
    xargs -0 sha256sum > SHA256SUMS )
PACK_INFO=$(mktemp)
{
  echo "mode: $STAGE"
  echo "created: $(date -Is)"
  echo "host: $(uname -a)"
  echo "protocol_version: Aurora-timesx-all-native-A48-v1"
  echo "refs: VisionTS EXP-014 git 7b434dd (4 files sha-pinned); old EXP-010 cache reuse_plan.json"
} > "$PACK_INFO"
cp "$PACK_INFO" /tmp/au_pack/PACK_INFO.txt

tar -C /tmp/au_pack -czf "$OUT" .
SIZE=$(stat -c %s "$OUT")
if [ "$SIZE" -ge $((50 * 1024 * 1024)) ]; then
  echo "ERROR: $OUT is $SIZE bytes (>=50MB)" >&2; exit 1
fi
rm -rf /tmp/au_pack "$PACK_INFO"
echo "pack: $OUT ($((SIZE / 1024)) KB)"
sha256sum "$OUT"
