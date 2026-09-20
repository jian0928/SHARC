#!/usr/bin/env bash
set -e

DOWNLOAD_DIR=${DOWNLOAD_DIR:-assets/checkpoints}
mkdir -p "$DOWNLOAD_DIR"

wget -O "$DOWNLOAD_DIR/smplx.tar.gz" "${SMPLX_URL:?smplx url required}"
wget -O "$DOWNLOAD_DIR/pixie.tar.gz" "${PIXIE_URL:?pixie url required}"
wget -O "$DOWNLOAD_DIR/sd21.safetensors" "${SD21_URL:?sd2.1 url required}"
