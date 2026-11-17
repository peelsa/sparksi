#!/bin/sh
# SPDX-License-Identifier: Apache-2.0

export LD_LIBRARY_PATH=/srv/example/builds/llama/lib
exec /srv/example/builds/llama/bin/llama-server \
  -m /srv/example/cache/example.gguf \
  --host 0.0.0.0 \
  --port 8014 \
  -c 262144 \
  -np 1 \
  -ctk q8_0 \
  -ctv q8_0 \
  -fa on \
  -ngl all \
  --jinja \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --reasoning off \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 40 \
  --presence-penalty 0.1 \
  --alias example
