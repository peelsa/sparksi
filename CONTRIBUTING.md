# Contributing to Sparksi

Thanks for helping. This page covers setup, the checks CI runs, and the rules every change follows.

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

Requires Python 3.12 or newer. Tests run on 3.12, 3.13 and 3.14.

Some page tests need Node 20 and Playwright. The WebKit job uses Playwright 1.55.

## Checks

CI runs on pull requests and pushes to the branches listed in `.github/workflows/ci.yml` (always `main`):

- **identity-guard** — `python scripts/check_identifiers.py .`
- **stdlib-guard** — `python scripts/check_stdlib.py .` — `src/sparksi/agent.py` and `announce.py` import only the standard library
- **lint** — `ruff check src tests scripts` (ruff 0.17.2, rules E and F)
- **ui-colors** — `python scripts/check_ui_colors.py`
- **design-detect** — `scripts/check_design.py` on the dashboard HTML
- **gitleaks** — secret scan
- **test** — `python -m pytest`
- **webkit-scroll** — WebKit page regressions

Before pushing, run locally:

```sh
python3 scripts/check_identifiers.py .
python3 scripts/check_stdlib.py .
ruff check src tests scripts
python3 scripts/check_ui_colors.py
.venv/bin/pytest
```

## Writing code

- Site values (hosts, addresses, paths, user names) come only from `sparksi.toml` through `sparksi.config`. Never hard-code them.
- The node agent and announcer stay standard-library only.
- The browser sends fields, never shell strings. argv lists are never joined for a shell.
- When touching placement or rendering, keep two-Spark output byte-for-byte unchanged. The engine builders only know head, worker and the both pair.
- Tests use the generic example config: spark-a / spark-b, documentation addresses.

## Public-safe rules

No real host names, tailnet or LAN addresses, home directories or personal names in code, tests, docs, commits or PR text. Use placeholders:

- node ids: `a` / `b`
- names: `spark-a` / `spark-b`
- tailnet: `100.64.0.0/24`
- documentation ranges: `192.0.2.0/24`, `198.51.100.0/24`, `198.18.0.0/15`
- home directories: write `~` or `$HOME` in commands; config examples use the user name `USER`

The identity guard fails on:

- `/home/<user>/` paths
- RFC 1918 addresses
- tailnet addresses outside `100.64.0.0/24`
- `*.bak` files
- `unlock.token`
- files named `sparksi.toml` or `catalog.yaml` (only the `.example` copies belong in the repo)
- a hashed denylist; CI can add words from a secret

Never commit `sparksi.toml`, `catalog.yaml`, the unlock token or backups.

## Commits and pull requests

- One topic per pull request.
- Sign off every commit: `git commit -s` adds a `Signed-off-by` line.
- No force-push to shared branches. No bypassing branch rules.
- The maintainer merges; do not merge your own PR. After a PR is
  open, push only fixes a review or a check asked for.
- Describe what changed and how it was tested.

## Docs

Update `README.md`, `docs/` and `llms.txt` when behavior a user can see changes. Document only what the code does. `CHANGELOG.md` has an Unreleased section.

## Security issues

Report privately to the Peelsa maintainers. See [SECURITY.md](SECURITY.md).
