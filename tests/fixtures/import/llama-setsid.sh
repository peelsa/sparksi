#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

BIN=/srv/example/builds/llama/bin/llama-server
export LD_LIBRARY_PATH="/srv/example/builds/llama/lib"
export MODEL_ALIAS=example
setsid "$BIN" -m /srv/example/cache/example.gguf --host 0.0.0.0 --port 8014 -c 8192
