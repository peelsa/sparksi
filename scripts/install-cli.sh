#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

# Install a ~/bin/sparksi launcher (and the dgx-mem alias). Does not touch systemd or models.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${HOME}/bin"
mkdir -p "$BIN"
for name in sparksi dgx-mem; do
  LAUNCHER="${BIN}/${name}"
  cat > "$LAUNCHER" <<PY
#!/usr/bin/env python3
import sys
sys.path.insert(0, "${ROOT}/src")
from sparksi.cli import main
if __name__ == "__main__":
    raise SystemExit(main())
PY
  chmod +x "$LAUNCHER"
  echo "installed ${LAUNCHER}"
done
"${BIN}/sparksi" --version
