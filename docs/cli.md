# Command line

`sparksi` is the command-line side of the dashboard. Every command below exists
as written; `sparksi <command> --help` lists its flags. `sparksi --version`
prints the version. `dgx-mem` is a deprecated alias of `sparksi`, kept for one
release.

## Observe

These only read.

- `sparksi status [--json]`: live snapshot of what is resident on each Spark.
- `sparksi inventory [--json]`: the catalog compared with what is live.
- `sparksi dry-run <set> [--json]`: memory check for a set of catalog models. It fails closed: a reading it cannot confirm is `UNCERTAIN`, never `OK`.
- `sparksi wanted [set <set>]`: show the desired set, or save one. Loads nothing.
- `sparksi tokens [--json]`: per-model lifetime prompt and generated token counts.
- `sparksi library [--json]`: archive releases and the copies on each Spark.

## Check the config

These read `sparksi.toml` and nothing else. They never contact a machine and never write.

- `sparksi config validate [--file PATH]`: check the file. It exits 0 and names the Sparks it found, or exits 1 and prints one `error:` line per problem: not TOML or not UTF-8 (a leading byte order mark is named), nested too deeply to read, not a regular file (a device such as `/dev/zero` or a pipe), larger than 1 MiB, not readable, more than four `[[nodes]]`, a repeated node id, a bad `ssh` value, a top-level table that is not a table, or a config path that cannot be worked out (a `~user` that does not exist, a deleted working directory). A deprecated `[paths]` key, or a `[protect] presets` that is not a list of known preset names, prints a `warning:` line and still exits 0. With no `--file` it checks the file Sparksi would load; if there is none it exits 1.
- `sparksi config show [--file PATH]`: print the settings in force (the built-in defaults merged with the file) as JSON. Anything that is not empty and not a boolean is printed as `<redacted>` when its key ends in `token`, `secret`, `password`, `passwd`, `credential`, `api_key` or `private_key`, with `-` read as `_` and an optional plural `s` (so `x-api-key`, `private-key`, `secrets` and `tokens` count, and `token_count` does not), or when it sits anywhere inside a table whose name ends that way (for example `[credentials]`). That includes numbers and every item of a list. The user part of a URL such as `http://user:pass@host` is dropped everywhere, and so is the value of a URL parameter (after `?`, `&`, `;` or `#`) whose name matches the same rule or is `sig` or ends in `signature` (so `X-Amz-Signature` counts). `?token=abc` becomes `?token=<redacted>`; the host, path and other parameters stay. The name is percent-decoded first, so `tok%65n` and `%2574oken` match, and `%3D` counts as `=`. A secret value runs to the next `&`, `;` or `#`, whatever it holds. A parameter whose decoded value holds a secret (a `next=` redirect that carries a token or user info) is hidden whole, and so is a string that is a URL only once decoded. User info is found after `//` or `\\`, up to the last `@` of the authority. A parameter name over 256 characters, or a value or string that still changes after five rounds of percent-decoding, is hidden rather than inspected, since no real URL has either. A name such as `token[]`, `token[0]` or `auth[token]` is secret when any part of it is. A string with no `//` or `\\` is not treated as a URL, so only its own key name decides. No key Sparksi ships holds a secret; this covers keys a site adds. It exits 1 with the problem when the file is unusable.

If `sparksi.toml` is unusable, every other `sparksi` command (`status`, `serve`, and the rest, also under the `dgx-mem` alias) and `sparksi-monitor` print the problem and exit 1 rather than run on the built-in example. `sparksi config validate` and `sparksi config show` still run, so the file can be checked and fixed.

## Lock

- `sparksi lock [--reason TEXT] [--json]`: turn the lock on.
- `sparksi unlock [--ttl SECONDS] [--json]`: open the lock for that many seconds (default 900). Then it is locked again.

See [lock.md](lock.md).

## Recipes through the controller

These commands talk only to the controller HTTP API. They never call Docker.

- `sparksi start <recipe-id> [--wait] [--timeout SECS] [--ctx N] [--json]`
- `sparksi stop <recipe-id|run-id> [--wait] [--timeout SECS] [--json]`
- `sparksi runs [--json]`
- `sparksi plan <recipe-id> [--ctx N] [--json]`: dry plan only; starts nothing.

`--timeout` defaults to 180 seconds. The controller applies the lock, the
free-memory check, port conflicts, duplicate refusal, two-Spark pairing, and
adoption of a matching container that is already running. A recipe that is
already ready, including an adopted container, makes `start` exit 0 without
starting a second copy. Off-limits ports (8188 with the `video-service` preset) and protected services are refused.

Global flags go before the command:

- `--server URL`: the controller. Default is loopback on the configured UI port. `SPARKSI_SERVER` overrides the default.
- `--unlock`: send the unlock secret in a request header. The secret is read from `SPARKSI_UNLOCK`, the file named by `SPARKSI_UNLOCK_FILE`, or the unlock file beside the lock. It is never printed and is never a command argument.

Without `--unlock`, a locked controller makes `start` and `stop` exit 2 with
`Unlock first.`

### Exit codes

For `start`, `stop`, `runs` and `plan`:

| Code | Meaning |
|---|---|
| 0 | ready, stopped, already in that state, or a plan that is ready |
| 1 | the run failed or timed out while waiting to become ready |
| 2 | refused (lock, memory, port, duplicate, protected id) or the controller could not be reached |
| 3 | `--wait` timed out; the run is left as it is |

### JSON output

`--json` prints one object with sorted keys.

- `runs`: `ok`, `action`, `error`, `runs`, and a `serving` array with a stable `state` and `reason`. Each run has `recipe`, `id`, `state`, `port`, `nodes` (`role`, `state`) and `started_by`.
- `start`: `ok`, `action`, `recipe`, `already`, `error`, `run`.
- `stop`: `ok`, `action`, `target`, `already`, `error`, `run`.
- `plan`: `ok`, `action`, `recipe`, `verdict`, `conflicts`, `port`, `error`.

## Import a pinned recipe

```sh
sparksi recipes import <model-id> [--from container:NAME|compose:FILE|unit:NAME|script:PATH] [--dry-run] [--json]
sparksi recipes import --all --dry-run
```

Reads an existing launch over SSH into a pinned recipe. It starts and stops
nothing. `--dry-run` prints the recipe and any problems without saving. See
[recipes.md](recipes.md#pinned-recipes).

## Copy and delete weights

- `sparksi stage <id> --host <spark> [--host <spark>]`: copy a model from the archive onto one or more Sparks.
- `sparksi drop-local <id> --host <spark> --confirm <local-path>`: delete the copy on one Spark. You type its path to confirm.

## Catalog sets (legacy)

- `sparksi apply [<set>] [--commit] [--json]`: plan the wanted set, or the set named. Nothing runs unless `--commit` is given, the lock is off and the plan is `READY`. A failed commit puts the lock back on and rolls back only what it started.
- `sparksi swap`: alias of `apply`.
- `sparksi run <id>`: start a catalog launcher on this machine. Needs the lock off.

## Serve

```sh
sparksi serve [--port 7500] [--state-dir DIR] [--bind ADDR] [--no-announce]
```

Runs the dashboard and API. `--state-dir` gives this process its own lock, job
and run records. `--bind` replaces the configured listen addresses for this
process. That list is checked only against the catalog's `host.bind_deny` (or `0.0.0.0` and `::` when that list is empty), plus `0.0.0.0`, `::` and `*`. The `bind_deny` lists in `sparksi.toml` are not applied to `--bind`; see [Listen addresses](config.md#listen-addresses). `--no-announce` skips
the announce listener.

## Service units

```sh
sparksi install-unit --print [--unit controller|announce] [--prefix DIR] [--node-id ID] [--controller IP:PORT]
```

Writes a user systemd unit to stdout. It never enables or starts anything. See
[deploy.md](deploy.md).

## Terminal monitor

`sparksi-monitor` is a terminal view of the monitor card. Install the
`monitor` extra (`pip install -e '.[monitor]'`) for psutil.

## Refused verbs

`sparksi load`, `sparksi evict` and `sparksi install` exist only to refuse.
