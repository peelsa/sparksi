# Configuration

Sparksi reads one TOML file, `sparksi.toml`, for every site value: machines, addresses, paths and user names. Models and services are listed separately in `catalog.yaml` (see [The catalog](#the-catalog)). Start from `config/sparksi.example.toml`; every value in it is a placeholder.

## Where the file lives

The file is searched for in this order:

1. `$SPARKSI_CONFIG`, if that variable is set
2. `$XDG_CONFIG_HOME/sparksi/sparksi.toml` (by default `~/.config/sparksi/sparksi.toml`)
3. `./config/sparksi.toml`

The first file found is used. When `SPARKSI_CONFIG` is set, that file is used even if a later one in the list also exists.

If no file exists at all, Sparksi falls back to built-in defaults. Those defaults describe the generic two-Spark example in `config/sparksi.example.toml`.

The real configuration file is never committed to version control.

## Checking the file

`sparksi config validate` checks the file Sparksi would load, or the one named by `--file`, and `sparksi config show` prints the settings in force. See [cli.md](cli.md#check-the-config).

A file that cannot be used (not valid TOML, not readable, more than four `[[nodes]]`, a repeated node id, a bad `ssh` value, or a top-level table such as `[controller]` written as a string) does not stop Sparksi from being imported. It logs one warning that names the file and the problem, and every command other than `sparksi config ...` exits 1 with that message. Only a missing file is different: that falls back to the built-in defaults, as above.

## Sparks: one to four

The `[[nodes]]` array lists the Sparks. It takes one to four tables.

Zero tables is refused with the message `needs at least one [[nodes]] table (the head Spark)`. Five or more is refused with `Sparksi manages 1 to 4 Sparks`. Each node must have a unique `id`.

Roles follow file order:

- First `[[nodes]]` table: head
- Second: worker
- Third: Spark 3 (role `node3`)
- Fourth: Spark 4 (role `node4`)

Each node table accepts these keys:

- `id` (required) — short key used in announce packets
- `name` — display name and catalog host key; defaults to `id`
- `ssh` — ssh destination, or `local`; defaults to `name`
- `tailscale` — that Spark's tailnet address. Announce packets are accepted only from this address, and engines bind to it.
- `bind_deny` — addresses a model engine must not listen on, such as that Spark's LAN and WiFi addresses. A recipe plan checks each Spark's `tailscale` address against every Spark's `bind_deny` (plus `0.0.0.0` and `::`) and refuses on a match. It is not applied to the dashboard's own listen addresses
- `uma_gib` — defaults to 121
- `mount_roots` — extra host directories a pinned recipe may bind, besides the model cache. `/` is never a root.
- `[nodes.fabric]` (optional) — used by two-Spark recipes, and only read, never written:
  - `interface` — the ConnectX port that is up
  - `ip` — that port's address
  - `hcas` — RoCE device names for `NCCL_IB_HCA`

One Spark works fully. Every model that fits on one Spark can be loaded, recipes save and load, and the pages show one Spark. The two-Spark placement is listed as not allowed with the reason `needs 2 Sparks; this site has 1 Spark`, and a recipe saved for two Sparks is refused with the same words.

Multi-Spark runs use exactly two Sparks. With three or four Sparks you can still run a model on any single Spark, or on any pair — head and worker, or worker and Spark 3, for example. Within a pair, the first Spark in file order leads and serves the port; the second launches first. There are no three- or four-Spark runs.

Placement names in saved recipes:

- One Spark: `head`, `worker`, `node3`, `node4`
- Head and worker together: `both`
- Any other pair: the two roles joined with `+`, for example `worker+node3`

## Example: one Spark

```toml
[controller]
name = "controller"
bind = ["127.0.0.1", "100.64.0.1"]

[[nodes]]
id = "a"
name = "spark-a"
ssh = "local"
tailscale = "100.64.0.1"
```

When the controller runs on that same Spark, `ssh = "local"` runs node operations without SSH.

## Example: two Sparks

```toml
[[nodes]]
id = "a"
name = "spark-a"
ssh = "spark-a"
tailscale = "100.64.0.11"

  [nodes.fabric]
  interface = "enp1s0f1np1"
  ip = "192.0.2.11"
  hcas = ["roce0", "roce1"]

[[nodes]]
id = "b"
name = "spark-b"
ssh = "spark-b"
tailscale = "100.64.0.12"

  [nodes.fabric]
  interface = "enp1s0f1np1"
  ip = "192.0.2.12"
  hcas = ["roce0", "roce1"]
```

Spark `a` is the head, Spark `b` is the worker. The `[nodes.fabric]` tables are read by two-Spark recipes. A third or fourth Spark is one more `[[nodes]]` table in the same form.

## Monitored hosts

The `[[endpoints]]` array lists up to two monitored inference hosts that are not Sparks. Each entry takes:

- `name`
- `url` — must be of the form `http://host:port`

These hosts appear on the Overview page. Sparksi does not start or stop them; it only watches them.

## Changing machines

The Network page edits only the `[[nodes]]` and `[[endpoints]]` tables. Before writing, it keeps a timestamped backup of the file and checks that the edited file still parses.

Removal rules:

- Only the last Spark can be removed. Removing a middle Spark would renumber the Sparks after it, and recipes saved for them would then land on another machine.
- The head can never be removed. Give the head role to another Spark first.
- To move a Spark to the end so it can be removed, swap roles.

Every change asks first and needs the lock off. Nothing starts or stops. Restart Sparksi after a change so it uses the new setup. Recipes saved for a removed Spark stop working until a Spark takes that role again.
## Other sections

### `[controller]`

Settings for the controller process itself.

| Key | Meaning |
| --- | --- |
| `name` | Name of the machine that runs `sparksi serve`, matched against its hostname (`SPARKSI_CONTROLLER_MODE=<name>` forces the match). |
| `bind` | Dashboard listen addresses: loopback plus one tailnet address. Used when the catalog has no `controller:` block and `--bind` is not given. See [Listen addresses](#listen-addresses). |
| `bind_deny` | Checked against `[controller] bind` only, and only when the catalog has no `controller:` block. Never applied to `--bind`. Empty means `0.0.0.0` and `::`. |
| `ui_port` | UI port. Default `7500`. |
| `announce_port` | UDP port for announce packets. Default `7501`. The listener binds this machine's tailnet address. |
| `reserved_ports` | Ports a recipe may not use. |
| `state_dir` | Runtime state. Default `$XDG_RUNTIME_DIR/sparksi`. `sparksi serve --state-dir` overrides it for that process. |
| `catalog` | Catalog file. Default `~/.config/sparksi/catalog.yaml`. |
| `recipes_dir` | Recipe directory. Default `$XDG_CONFIG_HOME/sparksi/recipes`. `SPARKSI_RECIPES_DIR` wins. |
| `metrics_url` | Optional VictoriaMetrics address for token counts. |
| `access_log` | Request timing in the service log: `slow` (over 1 s, the default), `all`, or `off`. |
| `monitor_upstream` | Optional read-only mirror of another Sparksi's `/api/monitor`. Loopback or tailnet only. |
| `allowed_hosts` | Extra names this controller answers to. |
| `stop_unmanaged` | Default false. True allows the Overview Stop button for models Sparksi did not start. |

### Listen addresses

The dashboard picks its listen addresses in this order:

1. `sparksi serve --bind ADDR` (repeat the flag or pass a comma list), when given.
2. On the controller: when `catalog.yaml` has a non-empty `controller:` block, that block's `bind_allow`, or `127.0.0.1` when the block has no `bind_allow`. An empty `controller:` block counts as no block. With no block, `[controller] bind` from `sparksi.toml`, or `127.0.0.1` when that is empty.
3. On any other machine: the catalog's `host.bind_allow`, or `127.0.0.1`.

Then the checks:

- `0.0.0.0`, `::` and `*` are always refused.
- Every list, including `--bind`, is checked against the catalog's `host.bind_deny`. When that list is empty, `0.0.0.0` and `::` are used instead.
- A list from step 2 or 3 is also checked against the deny list of the same block: the catalog's `controller.bind_deny` or `host.bind_deny`, or `[controller] bind_deny` when the catalog has no `controller:` block.

The `[[nodes]] bind_deny` lists in `sparksi.toml` are not used for the dashboard. They are only used when a recipe is planned. The plan checks only the `tailscale` addresses of the Sparks in that placement (the engine listens there), and checks each one against all the `[[nodes]] bind_deny` lists combined, plus `0.0.0.0` and `::`.

Nothing refuses a LAN address unless it is in one of these deny lists. Put each machine's LAN and WiFi addresses there.

### `[engines]`

| Key | Meaning |
| --- | --- |
| `vllm_image`, `sglang_image`, `trtllm_image` | Pinned container tags. A tag containing `latest` is ignored. |
| `allowed_images` | A recipe may only use images in this list. |
| `vllm_master_port` | Default `29501`. |
| `sglang_dist_port` | Default `50000`. |
| `kv_reserve_gib` | KV cache reserve added to the weights when checking that a model fits. Default `8`. |

Shipped defaults: `nvcr.io/nvidia/vllm:26.05-py3`, `nvcr.io/nvidia/sglang:26.02-py3`, `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6`.

### `[archive]`

| Key | Meaning |
| --- | --- |
| `id` | Archive identifier. |
| `node` | Node id that owns the storage. Empty means head. |
| `root` | Storage root path. |
| `label` | Display label. |
| `extra_blocked` | Extra paths the agent refuses, such as a raw mount behind a bind mount. |

### `[paths]`

Paths on the Sparks.

| Key | Meaning |
| --- | --- |
| `home` | The node user's home directory. |
| `hf_hub` | The model cache. Weight copies land under it. |
| `blocked_prefixes` | The agent refuses any path under these. |
| `too_broad` | The agent refuses these exact paths as copy or delete targets. |
| `direct_prefixes` | Address prefixes of the node-to-node link. |
| `lan_prefixes` | LAN address prefixes. Empty means any private IPv4 address outside the tailnet range. |
| `video_service_root` | Install directory of the protected video service. Its `state/mode` file is read; nothing under it is copied or deleted. |
| `model_inventory` | Optional model inventory file. Read-only; `sparksi inventory` diffs it against the catalog. |

The example uses `USER` as a placeholder user name in these paths. Other `[paths]` keys are described in comments in `config/sparksi.example.toml`.

### `service_commands`

Exact argv lists the node agent may run for a catalog service. Matched verbatim, element by element.

### `[services.<catalog-id>]`

One entry per controllable service.

| Key | Default | Meaning |
| --- | --- | --- |
| `user_controllable` | false | Turns on dashboard Start and Stop for this service. |
| `reserve_gib` | 40 | GiB that must be free on the head. |
| `start` | — | Start argv. |
| `stop` | — | Stop argv. |

The table name is the catalog service id. A protected service gets Start and Stop on the dashboard only when `user_controllable` is true and both commands are set. Both run only on the head and need the lock off. Start is refused when a model run is on the head or free memory there is below `reserve_gib`.

### `[extra_machines.tailscale]`

Rows of the form `"TailscaleHostName" = "shown-id"`. Each row maps a Tailscale host name to the id shown in the UI.

### `[compat]`

Compatibility keys for older API readers: `legacy_archive_id`, `legacy_mount_key`, `legacy_row_keys`. Empty for new installs.

### `[protect]`

Protection presets are named sets of rules for a service Sparksi must never touch. All are off by default.

One preset exists: `video-service`. When on:

- Port 8188 is off limits: never started on, stopped, health-probed, reused by a recipe, or used as the `serve` port.
- Catalog rows of kind `video_service` (old `comfy`) and ids `video-service`, `comfy`, `comfyui`, `h3`, `minimax-h3` (and ids containing `video-service`, `h3` or `comfy`) are never started, stopped or evicted.
- A stop/kill/rm/restart command naming it is refused.
- Its running process is left out of memory and unmanaged-container counts.
- The dashboard shows it as a protected row.

When off, port 8188 is an ordinary port and there is no video row.

Turn it on with the `[protect]` table, key `presets = ["video-service"]`, or env `SPARKSI_PROTECT=video-service` (comma-separated; added to the file's list). Unknown names are ignored and logged.

**Auto-enable.** An existing install that still uses an old name gets the preset turned on by itself, so it is never left unprotected. Markers: `[paths]` keys `comfy_root` or `model_truth`; a `[services.video-h3]` table; env `SPARKSI_MODEL_TRUTH`, `DGX_MEM_MODEL_TRUTH` or `SPARKSI_COMFY_STATE`; in the catalog a row with kind or id `comfy`, a `never_touch.do_not_stop` item containing `comfy`, a `comfy` tenant, or `seats.video: comfy`. `sparksi serve` logs one warning at start naming the markers. Add the explicit line to stop relying on the markers.

The node announcer runs on each node and cannot hear from the controller (it only sends UDP), so it decides on its own. It turns the preset on when its `SPARKSI_PROTECT` names it, or on a legacy marker on that node: the old node-id variable `DGX_ANNOUNCE_HOST`, `SPARKSI_COMFY_STATE`, the legacy install directory `~/services/comfyui`, or running as the legacy `dgx-announce.service` unit. It prints one note to stderr naming the marker. `sparksi install-unit --print --unit announce` writes `Environment=SPARKSI_PROTECT=` with the controller's active presets (auto-enabled ones included), so a unit rendered from a protected controller is protected on every node. A node with no marker and an older unit (for example a worker without the video service) has the preset off until its unit sets `SPARKSI_PROTECT=video-service`; add that line when updating an existing install. The node agent gets the preset values from the controller with each request.

`GET /api/config` returns `protect.presets` and `protect.off_limits_ports`; the page reads them.

```toml
[protect]
presets = ["video-service"]
```

### Deprecated names

Some config keys, env vars, API fields and catalog values were renamed to generic names. Existing installs keep working with no edits: the old names are still read, and any of the old names below turns on the `video-service` preset (see [`[protect]`](#protect)), so the video service keeps the same protection.

| Old name | New name | Where |
| --- | --- | --- |
| `comfy_root` | `video_service_root` | `[paths]` key |
| `model_truth` | `model_inventory` | `[paths]` key |
| `SPARKSI_MODEL_TRUTH`, `DGX_MEM_MODEL_TRUTH` | `SPARKSI_MODEL_INVENTORY` | env var |
| `SPARKSI_COMFY_STATE` | `SPARKSI_VIDEO_STATE` | env var (announcer) |
| `comfy` | `video_service` | catalog `kind` |
| `comfy` | `video-service` | catalog resident id |
| `comfy_live` | `video_service_live` | API JSON field |
| `diff_vs_model_truth` | `diff_vs_model_inventory` | API JSON field (`inventory`) |
| `wrote_model_truth` | `wrote_model_inventory` | API JSON field (`inventory`) |
| `comfy` | `video_service` | `/api/monitor` peer key (both sent) |
| `kind: comfy` (`legacy_kind`) and `comfy: true` | `kind: video_service` and `video_service: true` | `/api/monitor` video row (both sent) |
| `comfy` | `video-service` | probe `identity` on `/system_stats` (old value in `legacy_identity`) |
| `comfy` | `video_service` | node agent `/system_stats` flag (both sent) |
| `comfy` | `video` | announce packet key (both sent) |

When only the old config key or env var is set, Sparksi uses it and `sparksi serve` logs one deprecation warning at start. When both are set, the new one wins. During the rename every changed output carries both names: API JSON fields, the `/api/monitor` peer key and video row, the probe identity, the agent flag, and the announce packet. The controller reads either packet key and announcers send both, so controller and announcers can be updated in either order. A page loaded before the update keeps working: it reads the old names, and it also treats any row on port 8188 as the video service. Old names will be removed in a later release; all are listed in one table in `src/sparksi/compat.py`.

### What the page can read

`GET /api/config` returns only the display subset: node ids, names and roles, the head and worker, controller name, archive id and label. Addresses, SSH targets, paths and bind lists are left out. The page reads node names from this endpoint.

### Environment variables

New name first; the deprecated alias is kept for one release.

| Variable | Alias | Purpose |
| --- | --- | --- |
| `SPARKSI_CONFIG` | — | Config file path. |
| `SPARKSI_CATALOG` | `DGX_MEM_CATALOG` | Catalog file path. |
| `SPARKSI_STATE_DIR` | `DGX_MEM_STATE_DIR` | State directory. |
| `SPARKSI_LOCK` | `DGX_MEM_LOCK` | Lock file. |
| `SPARKSI_CONTROLLER_MODE` | `DGX_MEM_CONTROLLER` | Treat this machine as the controller named `<name>`. |
| `SPARKSI_RECIPES_DIR` | — | Recipe directory; overrides `recipes_dir`. |
| `SPARKSI_SMI_TIMEOUT` | `DGX_MEM_SMI_TIMEOUT` | `nvidia-smi` timeout. |
| `SPARKSI_HTTP_TIMEOUT` | `DGX_MEM_HTTP_TIMEOUT` | HTTP timeout. |
| `SPARKSI_CMD_TIMEOUT` | `DGX_MEM_CMD_TIMEOUT` | Command timeout. |
| `SPARKSI_MODEL_INVENTORY` | `SPARKSI_MODEL_TRUTH`, `DGX_MEM_MODEL_TRUTH` | Model inventory file; overrides `[paths] model_inventory`. |

Announcer only:

| Variable | Purpose |
| --- | --- |
| `SPARKSI_NODE_ID` | A `[[nodes]]` id. |
| `SPARKSI_CONTROLLER` | Controller as `ip[:port]`. |
| `SPARKSI_ADVERTISE` | That Spark's tailnet address. |
| `SPARKSI_NODE_IDS` | Comma-separated node ids, for example `a,b`. |
| `SPARKSI_VIDEO_STATE` | State directory of the video service (old name `SPARKSI_COMFY_STATE`). |

### The catalog

`catalog.yaml` is also never committed. See `config/catalog.example.yaml` for a template. It lists library rows (models), residents, services and seats.

Per-row flags:

| Flag | Meaning |
| --- | --- |
| `identity.served` | Served model names that mark the row live. A shared port is not enough. |
| `identity.served_prefix` | Served-name prefix. |
| `no_stage` | Nothing to copy before start. |
| `port_live` | The row is seen only by its port. |
| `start_note` | Text shown after the start step. |
| `display` | Display name (on a resident). |

A model answering on a Spark that matches no catalog row shows as "Not in catalog" with no Load or Stop.

### Agent and announcer

The node agent and announcer are standard-library only and carry no site values. The controller sends the agent its config with every request. The announcer reads its environment.