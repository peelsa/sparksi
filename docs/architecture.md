# Architecture

Sparksi is a local Python CLI plus an HTTP API and dashboard. It is not a
container and not an OOM killer.

```
  Spark 1 (head)    --UDP announce-->  controller (sparksi serve, :7500 UI/API)
  Spark 2 (worker)  --UDP announce-->        |
  Spark 3, Spark 4  --UDP announce-->        |
         ^                                   |
         +---- ssh BatchMode, agent.py on stdin (probe / copy / run / stop) --+
```

- `sparksi.announce` runs on each Spark (standard library only) and sends one
  datagram every 3 s to the controller's announce port (default 7501).
- `sparksi.agent` is sent over SSH for each operation (standard library only,
  never installed). For recipe folders it has four ops. `files_stage` writes a
  copy once under that Spark's recipe root, checks every file by SHA-256, makes
  it read-only and renames it into place. It never rewrites or removes an
  existing copy; it only compares it. `files_check` compares a copy with a
  manifest and reports missing, changed and extra files. `files_read` and
  `file_sha256` read files under the mount roots without changing them.
- `sparksi.monitor` shapes the announce data into the monitor card, which the
  dashboard and the terminal UI (`sparksi-monitor`) both read.

## Sources, most authoritative first

1. An optional production model file named in `[paths]`: read only, compared
   with the catalog by `sparksi inventory`
2. The catalog (`catalog.yaml`): library rows, services, seats
3. Live probe and announce packets: what is actually resident
4. Saved recipes: how each model is started
5. Wanted file: desired set (beside the lock)
6. Apply plan: last plan, not executed unless `--commit`

## Rules

1. Never write the production model file. Catalog plus read-only diff only.
2. Probes report actual residents only. Catalog membership is not residency.
3. With the `video-service` preset on, the video seat on port 8188 stays up
   under `apply`. A plan that would unload it is `REFUSE`. Recipe runs never
   start, stop or probe port 8188. The preset is off by default; see
   [`[protect]`](config.md#protect).
4. Observation does not heal units. A systemd conflict is a status line.
5. `apply --commit` only with the lock OFF and a `READY` plan. Re-probe before
   mutating; fail closed (lock back ON) and roll back only what this apply started.
6. Dry-run is fail-closed: an always-tenant miss is `UNCERTAIN`; never `OK`
   from `MemAvailable` alone. RSS and declared VRAM are labeled.
7. Lock defaults ON. A missing, corrupt or expired lease means locked.
8. Seats (`brain`, `vision`, `counsel`, `video`) are names that map to a catalog id.
9. Bind loopback plus one tailnet address. Refuse wildcard addresses and any
   address in the deny list that applies ([config](config.md#listen-addresses)),
   and refuse a reserved model port as the dashboard port.
10. `load`, `evict`, `install` refuse. `stage`, `drop-local`, `run` are the
    bounded mutators; `run` calls only the argv on the library row.
11. One to four Sparks, in `[[nodes]]` order: head, worker, Spark 3, Spark 4.
    A run uses one Spark or exactly two. The engine builders see only `head`,
    `worker` and `both`; `placement.engine_view` maps any other Spark or pair
    onto them, so two-Spark output is unchanged.
12. The browser sends fields. The controller renders argv lists and never
    joins them for a shell.

## Modules

| Module | Role |
|---|---|
| `config` | the only place site values come from; one to four `[[nodes]]` |
| `placement` | placement names (single Spark, `both`, `role+role`) and what each covers |
| `netconfig`, `options` | Network and Options page edits to `sparksi.toml`, with a backup |
| `cli`, `serve` | entry points (CLI, HTTP :7500, UI in `static/index.html`) |
| `announce` / `announce_rx` | node-side UDP sender / controller-side inbox |
| `dash`, `monitor/` | monitor card and terminal UI from announce data |
| `agent`, `remote` | node ops over SSH (agent shipped on stdin) |
| `library`, `launch`, `jobs` | archive releases, copy/delete/run plans, background jobs |
| `nodes` | live per-Spark memory, ports, and containers over SSH (`GET /api/nodes`) |
| `recipes/` | model cards, engine argv, validation, and remote run/stop/status |
| `catalog`, `probe`, `status`, `dryrun`, `apply`, `wanted`, `lock` | side-view core |
| `tokens`, `token_day`, `llama_day` | token ledgers |
| `history`, `telemetry` | chart cache and the durable per-Spark telemetry files ([telemetry](telemetry.md)) |
| `stage_feed` | loopback staging reader ([stage](stage.md)); not the production server |
