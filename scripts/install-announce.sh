#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Copy announce.py onto a node. Does not restart or enable anything unless
# --restart is passed, and never enables a unit.
#
# The file that lands on the node is the script as it is in the repo. Node id,
# controller address, and the node's configured client address stay in the
# unit environment (SPARKSI_NODE_ID, SPARKSI_CONTROLLER, and
# SPARKSI_ADVERTISE). Render that unit with sparksi install-unit --print.
# This script does not write those values into the copied file.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${ROOT}/src/sparksi/announce.py"
HOST=""
DEST=".local/bin/sparksi-announce"
RESTART=0
UNIT=""

usage() {
  cat <<'EOF'
usage: install-announce.sh --host node-a [--dest .local/bin/sparksi-announce] [--restart [--unit NAME.service]]

Copy src/sparksi/announce.py to a node with ssh -o BatchMode=yes.
An existing copy is saved as the same path plus .bak.
Nothing is restarted unless --restart is passed. The unit is never enabled.
--restart restarts --unit, or else the installed unit it finds:
dgx-announce.service when the node has it, otherwise sparksi-announce.service.

SPARKSI_ADVERTISE is not stored in the copied file. sparksi install-unit
--print --unit announce fills it from that node's configured tailscale
address. An empty address leaves wildcard URL checks off.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --dest)
      DEST="${2:-}"
      shift 2
      ;;
    --restart)
      RESTART=1
      shift
      ;;
    --unit)
      UNIT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${HOST}" ]]; then
  echo "install-announce.sh: --host is required" >&2
  exit 2
fi
if [[ "${HOST}" == -* || "${HOST}" =~ [[:space:][:cntrl:]] ]]; then
  echo "install-announce.sh: --host must be a host name, not an option" >&2
  exit 2
fi
if [[ "${HOST}" == -* || "${HOST}" == *" "* ]]; then
  echo "install-announce.sh: refusing host ${HOST}" >&2
  exit 2
fi
if [[ -z "${DEST}" || "${DEST}" == *" "* ]]; then
  echo "install-announce.sh: refusing dest ${DEST}" >&2
  exit 2
fi
if [[ -n "${UNIT}" && ! "${UNIT}" =~ ^[A-Za-z0-9][A-Za-z0-9@._-]*\.service$ ]]; then
  echo "install-announce.sh: refusing unit ${UNIT}" >&2
  exit 2
fi
if [[ ! -f "${SRC}" ]]; then
  echo "install-announce.sh: missing ${SRC}" >&2
  exit 2
fi

payload="$(base64 -w 0 "${SRC}")"
ssh -o BatchMode=yes -o ConnectTimeout=8 -- "${HOST}" bash -s -- "${DEST}" "${payload}" <<'EOS'
set -euo pipefail
dest="$1"
payload="$2"
case "${dest}" in
  /*) ;;
  *) dest="${HOME}/${dest#./}" ;;
esac
mkdir -p "$(dirname "${dest}")"
if [[ -e "${dest}" ]]; then
  cp -a "${dest}" "${dest}.bak"
fi
tmp="${dest}.tmp"
printf '%s' "${payload}" | base64 -d > "${tmp}"
chmod 0755 "${tmp}"
mv -f "${tmp}" "${dest}"
EOS

if [[ "${RESTART}" -eq 1 ]]; then
  if [[ -z "${UNIT}" ]]; then
    # Older nodes run the announcer as dgx-announce.service; new ones as sparksi-announce.service.
    units="$(ssh -o BatchMode=yes -o ConnectTimeout=8 -- "${HOST}" systemctl --user list-unit-files --no-legend dgx-announce.service sparksi-announce.service || true)"
    UNIT="sparksi-announce.service"
    if grep -qE '^dgx-announce\.service[[:space:]]' <<<"${units}"; then
      UNIT="dgx-announce.service"
    fi
  fi
  ssh -o BatchMode=yes -o ConnectTimeout=8 -- "${HOST}" systemctl --user restart "${UNIT}"
  echo "restarted ${UNIT} on ${HOST}"
fi
