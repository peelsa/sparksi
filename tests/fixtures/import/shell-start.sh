#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
err() {
  echo "$*" >&2
}
log() {
  echo "$1"
}
for v in IMAGE PORT; do
  [ -z "${!v:-}" ] && err "$v is unset"
done
log "ready"
exec docker run --network host --gpus all \
  -v /srv/example/cache:/root/.cache/huggingface \
  -e HF_TOKEN \
  "$IMAGE" \
  vllm serve "$MODEL" --port "$PORT"
