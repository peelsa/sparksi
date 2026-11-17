#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

BIN_DIR=/srv/example/builds/llama/bin
export LD_LIBRARY_PATH="$BIN_DIR:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NESTED="${A:-${B:-x}}"
BIN=$BIN_DIR/llama-server
_CLI_HF_TOKEN="${HF_TOKEN:-}"
if [ -n "$_CLI_HF_TOKEN" ]; then
  setsid "$BIN" -m /srv/example/cache/example.gguf --host 0.0.0.0 --port 8014 -e HF_TOKEN="$_CLI_HF_TOKEN"
fi
