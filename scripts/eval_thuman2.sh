#!/usr/bin/env bash
set -e

PRED_DIR=${PRED_DIR:?pred dir required}
GT_DIR=${GT_DIR:?gt dir required}
OUT=${OUT:-output}
DEVICE=${DEVICE:-}
THRESHOLDS=${THRESHOLDS:-}

ARGS=()
[ -n "$THRESHOLDS" ] && ARGS+=(--thresholds $THRESHOLDS)
[ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")

python tools/evaluate.py \
  --pred_dir "$PRED_DIR" \
  --gt_dir "$GT_DIR" \
  --out "$OUT" \
  "${ARGS[@]}"
