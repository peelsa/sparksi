# Sparksi

Sparksi is the Peelsa model manager for NVIDIA DGX Spark. It runs on one to
four Sparks, shows what each one is actually serving, and loads and unloads
models from saved recipes. It is part of the Xanthsi suite by Peelsa.

> Status: private, pre-release (0.6.0). Planned to be open-sourced under
> Apache-2.0.

## What it does

- Shows what is actually resident on each Spark, from UDP announce packets every 3 s, not SSH polling.
- Checks served-model port health (`/v1/models`, `/metrics`).
- Keeps per-model token counts and a 24-hour resource chart with durable telemetry files.
- Loads and unloads models from saved recipes for vLLM, SGLang, TensorRT-LLM and llama.cpp containers or binaries.
- Copies model weights from an archive store onto the Sparks.

Sparksi ships a dashboard and HTTP API (`sparksi serve`, port 7500) plus a CLI (`sparksi`). Dashboard pages: Overview, Model Manager, Library, Archive, Network, Options, Token totals, Activity.

## Quick start

Full steps are in [docs/getting-started.md](docs/getting-started.md).

```sh
git clone <repo-url> sparksi && cd sparksi
python3 -m venv .venv && .venv/bin/pip install -e .
mkdir -p ~/.config/sparksi
cp config/sparksi.example.toml ~/.config/sparksi/sparksi.toml
cp config/catalog.example.yaml ~/.config/sparksi/catalog.yaml
"${EDITOR:-vi}" ~/.config/sparksi/sparksi.toml ~/.config/sparksi/catalog.yaml
.venv/bin/sparksi status
.venv/bin/sparksi serve   # http://127.0.0.1:7500
```

## Sparks: one to four

- One Spark works fully: load any model that fits one Spark.
- Two to four Sparks: run on any single Spark or on any pair; multi-Spark runs always use exactly two Sparks.
- On the Network page only the last Spark can be removed, and never the head.

## Safety

- The lock is ON by default. Load, Unload, Restart, Network edits and saving a GPU max-clock cap on Overview need Unlock (15 minutes from the page).
- Load, Unload and Restart each ask first, with the model, the machines and the port.
- The browser sends fields and never a command string.
- The dashboard listens on loopback plus one tailnet address and always refuses `0.0.0.0`, `::` and `*`. Other refused addresses come from a deny list; see [Listen addresses](docs/config.md#listen-addresses).
- Services the catalog marks as protected have no Start or Stop on the dashboard unless you opt one in under `[services.<id>]`, and recipe runs refuse to start or stop them. Built-in rules also protect the node announcer. The optional `video-service` preset protects a video service and its port (8188); it is off by default, see [`[protect]`](docs/config.md#protect).
- The `load`, `evict` and `install` CLI verbs refuse.

`dgx-mem` is installed as a deprecated alias of `sparksi` for one release.

## Documentation

- [docs/getting-started.md](docs/getting-started.md) — install and first run
- [docs/config.md](docs/config.md) — `sparksi.toml`, 1-4 Sparks
- [docs/pages.md](docs/pages.md) — each dashboard page
- [docs/recipes.md](docs/recipes.md) — recipes, naming, defaults
- [docs/lock.md](docs/lock.md) — the lock and confirmations
- [docs/cli.md](docs/cli.md) — commands and exit codes
- [docs/deploy.md](docs/deploy.md) — user systemd units, announcer, firewall, SSH
- [docs/architecture.md](docs/architecture.md)
- [docs/telemetry.md](docs/telemetry.md)
- [docs/stage.md](docs/stage.md)
- [docs/state-machines.md](docs/state-machines.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)
- [SECURITY.md](SECURITY.md)
- [llms.txt](llms.txt)

## Development

```sh
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
ruff check src tests scripts
python3 scripts/check_identifiers.py
python3 scripts/check_stdlib.py
```

## License

Apache License 2.0. See [LICENSE](LICENSE). Copyright Peelsa.