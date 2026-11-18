#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Install Playwright and its WebKit build for the webkit-scroll CI job.
# Usage: ci-install-webkit.sh <dir>   (needs PLAYWRIGHT_VERSION and PLAYWRIGHT_BROWSERS_PATH)
#
# Each command gets its own time limit and up to three tries with a growing
# wait, so a stalled download or apt mirror fails in minutes instead of
# holding the job for hours.
set -euo pipefail

dir="${1:?usage: ci-install-webkit.sh <dir>}"
version="${PLAYWRIGHT_VERSION:?PLAYWRIGHT_VERSION is not set}"
: "${PLAYWRIGHT_BROWSERS_PATH:?PLAYWRIGHT_BROWSERS_PATH is not set}"

# attempt <seconds per try> <command...>
attempt() {
  local limit=$1
  shift
  local try=1 wait=15 rc=0
  while true; do
    if timeout --kill-after=30 "$limit" "$@"; then
      return 0
    else
      rc=$?
    fi
    if [ "$try" -ge 3 ]; then
      echo "::error::gave up after $try tries (exit $rc): $*" >&2
      return "$rc"
    fi
    echo "::warning::try $try failed (exit $rc, 124 means it hit the ${limit}s limit); retrying in ${wait}s: $*" >&2
    sleep "$wait"
    try=$((try + 1))
    wait=$((wait * 2))
  done
}

mkdir -p "$dir"
cd "$dir"
[ -f package.json ] || npm init -y > /dev/null
attempt 180 npm install --no-audit --no-fund "playwright@${version}"
attempt 300 npx --no-install playwright install webkit
attempt 300 sudo npx --no-install playwright install-deps webkit
