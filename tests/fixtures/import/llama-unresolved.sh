#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

setsid "$BIN" -m /srv/example/cache/example.gguf --host 0.0.0.0 --port 8014
