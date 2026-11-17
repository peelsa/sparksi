#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

BIN_DIR=/srv/example/builds/llama
export LD_LIBRARY_PATH=$BIN_DIR/lib:/usr/local/cuda/lib64
export HF_TOKEN
BIN=$BIN_DIR/bin/llama-server
setsid "$BIN" -m /srv/example/cache/example.gguf --host 0.0.0.0 --port 8014 -e HF_TOKEN -e HF_TOKEN=$HF_TOKEN >>/srv/example/logs/server.log 2>&1 &
echo leftover
