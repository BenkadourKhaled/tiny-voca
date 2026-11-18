#!/usr/bin/env bash
# Copy the files the Arabic demo serves into public/assets (they are not stored in git).
#
# Usage:
#   bash fetch_assets.sh <source> [model.onnx] [config.json]
#
# <source> is either a local directory holding the exported model.onnx + config.json
# (from `python -m tiny_voca.export`), or an HTTP(S) base URL from which those two
# files can be fetched. No Hugging Face repository is hard-coded: the Arabic
# checkpoint is local until it is published. The pure-Python G2P is copied
# verbatim from src/tiny_voca/arabic_text.py; it is all the phonemizer needs.
set -euo pipefail
cd "$(dirname "$0")"
SOURCE="${1:?local directory or base URL with model.onnx and config.json}"
MODEL="${2:-model.onnx}"
CONFIG="${3:-config.json}"

mkdir -p public/assets
cp ../src/tiny_voca/arabic_text.py public/assets/arabic_text.py

if [ -d "$SOURCE" ]; then
  cp "$SOURCE/$MODEL" public/assets/model.onnx
  cp "$SOURCE/$CONFIG" public/assets/config.json
else
  case "$SOURCE" in
    http://*|https://*) ;;
    *) echo "source must be a local directory or an http(s) URL: $SOURCE" >&2; exit 2 ;;
  esac
  curl -fsSL -o public/assets/model.onnx "${SOURCE%/}/$MODEL"
  curl -fsSL -o public/assets/config.json "${SOURCE%/}/$CONFIG"
fi
ls -l public/assets
