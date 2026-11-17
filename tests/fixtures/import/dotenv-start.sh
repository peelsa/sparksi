#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

cd /srv/example/builds/stack
if [ ! -f .env ]; then
  echo missing
else
  set -a
  . ./.env
  set +a
fi
exec docker run --network host --gpus all \
  -v /srv/example/cache:/root/.cache/huggingface \
  -e HF_TOKEN \
  "$IMAGE" \
  vllm serve "$MODEL" --served-model-name "$SERVED_MODEL_NAME" --port "$PORT"
