#!/usr/bin/env bash
set -e

DATA_DIR=${DATA_DIR:-data}
mkdir -p "$DATA_DIR"/thuman2
mkdir -p "$DATA_DIR"/cape
mkdir -p "$DATA_DIR"/renderpeople

IMAGE=${IMAGE:?sample image required}
OUT="$DATA_DIR"/thuman2/smplx
python tools/estimate_smplx.py --image "$IMAGE" --out "$OUT"
