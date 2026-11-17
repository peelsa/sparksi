#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR=/srv/example/builds
DO_LAUNCH="${DO_LAUNCH:-true}"
FP8_DENSE="${FP8_DENSE:-true}"
add_overlay() {
  local file="$1"
  OVERLAY_FILES+=(-v "$OV/$file:/opt/overlay/$file")
}
if $DO_LAUNCH && [[ "$FP8_DENSE" == "true" ]]; then
  OV="$SCRIPT_DIR/files/overlay"
fi
add_overlay first.patch
add_overlay second.patch
exec docker run --network host --gpus all \
  -v /srv/example/cache:/root/.cache/huggingface \
  "${OVERLAY_FILES[@]}" \
  example.invalid/vllm:patched \
  vllm serve /srv/example/cache/weights --host 0.0.0.0 --port 8014
