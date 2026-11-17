#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

ROOT=/srv/example/builds/llama
BIN="$ROOT/bin/llama-server"
export LD_LIBRARY_PATH="$ROOT/lib"
exec setsid "$BIN" -m /srv/example/cache/example.gguf --host 0.0.0.0 --port 8016
