#!/usr/bin/env bash
set -e

DATASET_MODULE=${DATASET_MODULE:?dataset module required}
DATASET_ROOT=${DATASET_ROOT:?dataset root required}
ANGLES=${ANGLES:?view angles required}
VAE_PATH=${VAE_PATH:?vae path required}
OUT=${OUT:-output}
DEVICE=${DEVICE:-}
FACE_CROP_SIZE=${FACE_CROP_SIZE:-}
BATCH_SIZE=${BATCH_SIZE:-}
ITERATIONS=${ITERATIONS:-}
LR=${LR:-}
GRAD_ACCUM=${GRAD_ACCUM:-}

ARGS=()
[ -n "$FACE_CROP_SIZE" ] && ARGS+=(--face_crop_size "$FACE_CROP_SIZE")
[ -n "$BATCH_SIZE" ] && ARGS+=(--batch_size "$BATCH_SIZE")
[ -n "$ITERATIONS" ] && ARGS+=(--iterations "$ITERATIONS")
[ -n "$LR" ] && ARGS+=(--lr "$LR")
[ -n "$GRAD_ACCUM" ] && ARGS+=(--grad_accum "$GRAD_ACCUM")
[ -n "$DEVICE" ] && ARGS+=(--device "$DEVICE")

python tools/train_synthesizer.py \
  --dataset_module "$DATASET_MODULE" \
  --dataset_root "$DATASET_ROOT" \
  --angles $ANGLES \
  --vae_path "$VAE_PATH" \
  --out "$OUT" \
  "${ARGS[@]}"
