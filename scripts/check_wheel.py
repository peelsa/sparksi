#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Build the wheel, install it in a clean venv, and run the first-run path.

A fresh `pip install` has no source checkout, so the CLI must find the shipped
example catalog inside the package. This script fails if the wheel is missing
package data, if `sparksi inventory` cannot load a catalog with no user config,
or if `sparksi serve` cannot answer on loopback.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "sparksi/examples/catalog.example.yaml",
    "sparksi/examples/sparksi.example.toml",
    "sparksi/static/index.html",
)


def clean_env(**extra: str) -> dict[str, str]:
    """Drop PYTHONPATH and friends so the venv cannot see the source tree."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "SPARKSI_", "DGX_MEM_", "PIP_"))}
    env.update(extra)
    return env


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd), flush=True)
    kw.setdefault("env", clean_env())
    return subprocess.run(cmd, check=True, text=True, **kw)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def get(url: str) -> int:
    with urllib.request.urlopen(url, timeout=15) as resp:  # noqa: S310 - loopback only
        resp.read()
        return int(resp.status)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sparksi-wheel-") as tmp_s:
        tmp = Path(tmp_s)
        dist = tmp / "dist"
        run([sys.executable, "-m", "pip", "wheel", "--no-deps", "-q", "-w", str(dist), str(ROOT)], cwd=tmp)
        wheels = sorted(dist.glob("sparksi-*.whl"))
        if len(wheels) != 1:
            print(f"expected one wheel, found {wheels}", file=sys.stderr)
            return 1
        names = set(zipfile.ZipFile(wheels[0]).namelist())
        missing = [n for n in REQUIRED if n not in names]
        if missing:
            print(f"wheel is missing package data: {missing}", file=sys.stderr)
            return 1

        venv = tmp / "venv"
        run([sys.executable, "-m", "venv", str(venv)])
        bindir = venv / "bin"
        run([str(bindir / "python"), "-m", "pip", "install", "-q", str(wheels[0])])

        home = tmp / "home"
        work = tmp / "work"
        for d in (home / ".config", home / "run", home / ".local" / "state", work):
            d.mkdir(parents=True, exist_ok=True)
        env = clean_env(
            PATH=f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
            HOME=str(home),
            XDG_CONFIG_HOME=str(home / ".config"),
            XDG_RUNTIME_DIR=str(home / "run"),
        )
        sparksi = str(bindir / "sparksi")
        run([sparksi, "--version"], cwd=work, env=env)
        inv = run([sparksi, "inventory"], cwd=work, env=env, capture_output=True, timeout=120)
        print(inv.stdout)
        if "catalog.example.yaml" in inv.stdout + inv.stderr:
            print("inventory could not load the shipped example catalog", file=sys.stderr)
            return 1

        port = free_port()
        cmd = [sparksi, "serve", "--port", str(port), "--bind", "127.0.0.1", "--no-announce",
               "--state-dir", str(home / "state")]
        print("+", " ".join(cmd), flush=True)
        proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 30
            while True:
                try:
                    get(base + "/")
                    break
                except OSError:
                    if proc.poll() is not None or time.monotonic() > deadline:
                        print("serve did not come up", file=sys.stderr)
                        return 1
                    time.sleep(0.5)
            for path in ("/", "/api/config", "/api/status", "/api/models"):
                code = get(base + path)
                print(f"GET {path} -> {code}")
                if code != 200:
                    return 1
        finally:
            proc.terminate()
            try:
                out, _ = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
            print(out[-2000:])
    print("wheel first-run path ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
