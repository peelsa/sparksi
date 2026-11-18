#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Install Playwright and its WebKit build for the webkit-scroll CI job.
# Usage: ci-install-webkit.sh <dir>   (needs PLAYWRIGHT_VERSION and PLAYWRIGHT_BROWSERS_PATH)
#
# Each command gets its own time limit and up to three tries with a growing
# wait, so a stalled download or apt mirror is cut off and retried instead
# of holding the job for hours.
set -euo pipefail

dir="${1:?usage: ci-install-webkit.sh <dir>}"
version="${PLAYWRIGHT_VERSION:?PLAYWRIGHT_VERSION is not set}"
: "${PLAYWRIGHT_BROWSERS_PATH:?PLAYWRIGHT_BROWSERS_PATH is not set}"

# attempt [--sudo] <seconds per try> <command...>
#
# With --sudo, timeout runs inside sudo as root. sudo starts the command on
# its own pty, so a timeout outside sudo only stops sudo and leaves a running
# apt-get or dpkg behind, still holding the dpkg lock for the next try.
#
# Worst case per command is 3 tries x (limit + 30s kill grace) + 45s of
# waits, about 55 minutes for all three. The step's 30-minute cap in ci.yml
# wins over that and fails the job; the per-try limits are there to cut off
# a single stalled try early, not to fit inside the cap.
attempt() {
  local prefix=()
  if [ "$1" = --sudo ]; then
    prefix=(sudo)
    shift
  fi
  local limit=$1
  shift
  local shown="$*"
  if [ "${#prefix[@]}" -gt 0 ]; then
    shown="${prefix[*]} $shown"
  fi
  local try=1 wait=15 rc=0
  while true; do
    if "${prefix[@]}" timeout --kill-after=30 "$limit" "$@"; then
      return 0
    else
      rc=$?
    fi
    if [ "$try" -ge 3 ]; then
      echo "::error::gave up after $try tries (exit $rc): $shown" >&2
      return "$rc"
    fi
    echo "::warning::try $try failed (exit $rc, 124 means it hit the ${limit}s limit); retrying in ${wait}s: $shown" >&2
    sleep "$wait"
    try=$((try + 1))
    wait=$((wait * 2))
  done
}

# On a GitHub runner, make apt drop a stalled mirror connection after 30s and
# retry it, instead of waiting minutes per package, and wait up to 60s for
# the dpkg lock. apt keeps finished downloads, so a retried install-deps
# carries on where the last try stopped.
if [ "${GITHUB_ACTIONS:-}" = true ]; then
  printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\nDPkg::Lock::Timeout "60";\n' |
    sudo tee /etc/apt/apt.conf.d/80-ci-retries > /dev/null
fi

mkdir -p "$dir"
cd "$dir"
[ -f package.json ] || npm init -y > /dev/null
attempt 180 npm install --no-audit --no-fund "playwright@${version}"
# System libraries first, so the browser install does not warn that they are missing.
attempt --sudo 480 npx --no-install playwright install-deps webkit
attempt 300 npx --no-install playwright install webkit
