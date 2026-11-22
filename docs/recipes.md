# Recipes

The first part of this page is for people using Model Manager. The [reference](#reference) after it covers model cards, engines, validation and the HTTP routes.

## What a recipe is

A recipe is the saved settings that start one model. It records the engine, the placement (which Spark, or which pair of Sparks), the port, and the engine fields: context length, max sequences, KV cache dtype, GPU memory fraction, and speculative decoding.

The browser never sends a command string. It sends fields only. The controller fills in the hosts from `sparksi.toml`, renders the argv, checks it, and only then asks the node agent to run it. The command preview shows the exact argv that will run.

A model can have many named recipes, so you can keep separate setups for different quants, concurrency levels, or context sizes.

## Naming

Recipe names allow letters, digits, `.`, `-` and `_`, up to 80 characters, and must start with a letter or digit.

The suggested format is `model-quant-concurrency-kvcache-notes`, lowercase, hyphen-separated. For example:

```
glm-5.3-flash-exl3-4bpw-c4-128k-fp8-2node
```

This reads as: model `glm-5.3-flash`, quant `exl3-4bpw`, concurrency `c4` (4 sequences), context `128k`, KV cache `fp8`, notes `2node`.

**Save as** opens a name box with a suggested name derived from the recipe's own settings. It is a suggestion only; any valid name saves. If a typed name does not follow the format, a hint appears.

The Default flag is never part of the name.

## One default per model

Exactly one recipe per model is the default. The first recipe saved for a model becomes its default. **Set as default** moves the flag to another recipe.

The default recipe cannot be deleted. To delete it, set another recipe as the default first. A rename cannot overwrite the default. The name `default` is reserved for the original recipe.

## The recipe list

The list is Explorer-style: a table with columns Default, Name, Concurrency, Context / KV, Nodes, Spec, Modified, and a `⋯` menu on each row.

Keyboard and mouse:

- Click a row to open it.
- `Up` / `Down` move the selection; `Enter` opens.
- `F2` renames; `Delete` deletes.
- `⋯` or `Shift+F10` opens the row menu: Open, Rename, Save as copy..., Set as default, Delete...

Sorting: click a column header, or use the Sort list. Sort options are Default first, Name, Newest, Concurrency, Context.

With no recipes yet, the page says "No saved recipes for this model yet" and offers Save as...; the form starts from the catalog settings.

## Saving and editing

The toolbar has: New, Save, Save as..., the open recipe name with a "unsaved changes" indicator, Dry run, and Load.

- Switching away from unsaved edits asks "Discard unsaved changes?".
- Deleting asks "Delete recipe \<name\>? This cannot be undone."
- Saving, renaming, setting default, and deleting do not need the lock.

When values are filled in, the precedence is: saved fields win, then the catalog card or launch config, then the engine defaults sized for the placement and free memory. A note under "Load default recipe" says where the loaded values came from. The API (`GET /api/models/<id>/recipe`) reports a source per field: saved recipe, launch config, catalog, or engine defaults.

## Pinned recipes

Pinned recipes (`kind = "pinned"`) keep a launch command that already worked, one per role. They are imported with:

```
sparksi recipes import <model-id> --from container:NAME|compose:FILE|unit:NAME|script:PATH
```

`--dry-run` prints the result without saving. `--all` surveys every catalog row that has a launcher.

Save never rewrites an imported pinned recipe: on one, Save is greyed out with "Pinned recipe: use Save As". Save as keeps your changes as a new recipe. That copy records `copied_from = "<source name>"`, is not locked, and Save updates it in place. The list API reports this as `locked` per recipe. The only pinned recipe of a model cannot be deleted.

The only rewrite at render time is the listen address, which becomes that Spark's tailnet address.

## Where recipes are stored

Recipes live in `$XDG_CONFIG_HOME/sparksi/recipes` (default `~/.config/sparksi/recipes`). The `[controller] recipes_dir` setting in `sparksi.toml`, or the `SPARKSI_RECIPES_DIR` environment variable, changes it.

Layout:

- Recipe folder: `<model-id>/<name>/recipe.toml`. The folder also holds the files the recipe mounts: `files/` for files every Spark uses, and `roles/<role>/` (for example `roles/head/start.sh` and `roles/worker/start.sh`) for files one Spark uses.
- Older layout: the default recipe at `<model-id>.toml` and other recipes at `named/<model-id>/<name>.toml`.
- `named/<model-id>/index.json` records the default's name in both layouts.

A folder wins over an older-layout file with the same name. Save writes `recipe.toml` inside the folder. Save as from a folder recipe copies the whole folder, files included, under the new name. Save as and Rename never replace an existing folder. Set as default only changes the name in `index.json`; a folder is never moved by it. Rename moves the folder. Delete moves the folder into `~/.config/sparksi/backups/<date>/deleted/<model-id>/<name>/` and removes nothing.

`sparksi.toml` and `catalog.yaml` are never written by a recipe save.

### Files a folder recipe mounts

In a folder recipe, a pinned role's mount source (or a native role's `binary`) can name a file in the folder:

- `@files/<path>` is `files/<path>`, the same file for every Spark.
- `@role/<path>` is `roles/<role>/<path>` for the Spark that runs that role, for example `@role/start.sh:/start.sh:ro` mounts `roles/head/start.sh` on the head and `roles/worker/start.sh` on the worker.

The saved recipe keeps these sources as written. A save, dry run or load checks them as the paths they become on each Spark: `<recipe_root>/<model-id>/<name>/<version>/...`, where the version is the hash of the folder's file list and of `recipe.toml` (see `[[nodes]] recipe_root` in [config](config.md)). A file the recipe names but the folder lacks, a path with `..`, a link, or a file under `roles/` for an unknown role refuses the plan. A start script mounted from the folder is checked for wildcard binds from the folder itself. The plan answer has a `files` block: the folder, its version, the file count, each Spark's copy path and the manifest. `recipe.toml` is never copied, but its hash is part of the version, so a folder with no files still has a version and a changed recorded hash makes a new one.

### Copy and check before every start

Before every start of a recipe that uses folder files, Sparksi does three things:

1. It reads the folder again. If any file changed since the check, the start is refused: "The recipe folder changed after it was checked, so nothing was started; load it again."
2. It copies the folder's files to each Spark the recipe uses, under `<recipe_root>/<model-id>/<name>/<version>/`. A folder file may be up to 16 MiB and all of them together up to 64 MiB; weights and caches stay outside the folder. The node agent writes the copy in a temporary folder, checks every file's SHA-256, makes the copy read-only and renames it into place.
3. It checks the result on every Spark before any container starts.

A copy that is already there is only compared, never rewritten. Because the folder name is the version, a saved change makes a new copy and never touches a copy a running container has mounted. If a copy cannot be made, or an existing copy differs, nothing starts and the answer is one sentence, for example "spark-b's copy of this recipe's files does not match the recipe folder, so nothing was started."

A dry run is read-only. Its `files.sparks` block shows each Spark's copy as `match`, `missing` (it will be copied at start), `differs` (this refuses the plan) or `unknown`. The run record keeps the version it mounts as `files_version`. A run started before its recipe was moved into a folder keeps the mounts it has.

### What the recipe page shows

For a folder recipe, the recipe page shows the folder, its version (the first 16 characters of the hash of its files and `recipe.toml`) and the number of files, which may be 0. It also shows one line for each Spark the recipe uses: the copy matches, it is not copied yet (Load copies it first), it does not match (Load is refused), or it could not be checked. The recipe list (`GET /api/models/<id>/recipes`) gives each recipe's `folder` and `version`. A plan gives the same under `files`, with each Spark's state in `files.sparks`.

Once a model has a recipe in a folder, its catalog launch command is no longer a way to start it. The model's `custom` engine is hidden, a plan with it is refused, and a catalog service Start is refused with "This model starts from its recipe folder now, not from its catalog launch command. Load its recipe instead." Stop still works, so a copy started the old way can be stopped. The catalog block itself stays in the catalog until you delete it.

### Recorded builds and memory

A role can record the exact build it runs, so a re-tagged image or a rebuilt binary is caught before a start:

```toml
[roles.head]
image = "example.invalid/vllm:1"
image_id = "sha256:<64 hex>"   # from: docker image inspect --format '{{.Id}}' <image>

[roles.worker]
runtime = "native"
binary = "/opt/server/bin/server"
binary_sha256 = "<64 hex>"     # from: sha256sum <binary>

[roles.worker.library_sha256]   # libraries the binary loads from its own build tree
"/opt/server/bin/libexample.so.0" = "<64 hex>"
```

For a native role, `sparksi recipes bundle` records `binary_sha256` and `library_sha256`. An older recipe that records only `binary_sha256` gets its libraries too, once its binary still matches that hash; a binary that no longer matches refuses the bundle. The libraries are found by reading the binary's needed libraries (and theirs), its RPATH or RUNPATH (with `$ORIGIN`, and the RPATH of the files that load it, as the dynamic loader does) and the role's `LD_LIBRARY_PATH`, without running anything. A file that is not a 64-bit little-endian ELF, or a damaged one, is read as loading nothing, and at most 256 libraries are recorded. Only libraries inside the binary's own tree (the folder above its `bin/`) and inside the Spark's mount roots are recorded, so system and CUDA libraries are left out. The start check reads the binary and every recorded library in one call, and a changed or missing one refuses with one sentence, for example "/opt/server/bin/libexample.so.0 on spark-a does not match the SHA-256 this recipe records, so it will not start." A library added to the tree later is not seen until the binary is bundled again.

Every plan and every start checks the Spark that runs the role. A different build refuses with one sentence, for example "spark-a has a different build of image example.invalid/vllm:1 than this recipe records, so it will not start." The image and the binary are never copied into the recipe folder. They stay where they are, and only their identity is recorded.

A recipe can set its own KV reserve, which the memory check uses instead of the model card's:

```toml
[memory]
kv_reserve_gib = 15
```

Without `[memory]`, the model card's `kv_reserve_gib` is used as before.

### Moving an older recipe into a folder

`sparksi recipes bundle <model-id> [--recipe NAME] [--keep SOURCE] [--dry-run]` moves an older pinned recipe into its own folder. It reads each file the recipe mounts from the Spark that mounts it (read only) and puts it in the folder. It also rewrites each mount to an `@files/` or `@role/` source, and records each role's `image_id` (or `binary_sha256` and `library_sha256` for a native role) and the model card's KV reserve as `[memory] kv_reserve_gib`.

- A file mounted as the role's start script (a `.sh` the command runs) becomes `roles/<role>/start.sh`. The head and worker scripts stay separate.
- Other files go under `files/<container path>`. When the head and the worker mount the same container path with different bytes, each copy goes under `roles/<role>/` instead.
- Weights and the Hugging Face cache are never copied, and nor is a source given with `--keep`. Those mounts stay as they are.
- A link, a missing file or a file over the size limit refuses the bundle, and nothing is written.

`--dry-run` lists every file, its hash and where it was read from, and writes nothing. A write builds the folder in a temporary sibling, checks every hash, renames it into place and moves the older recipe file to `backups/<date>/bundled/<model-id>/<name>.toml`. Nothing is deleted. To undo it, move that file back and move the new folder aside.

`sparksi recipes tidy-backups [--dry-run]` moves `recipes.bak-*` folders and `*.bak*` files from the config folder and the recipes folder into `backups/<date>/`. It only moves. A file whose target already exists is left where it is.

## Loading a recipe

To load a recipe:

1. Pick the recipe.
2. Check the placement and port. Ports are 8010-8099; an off-limits port is never used.
3. Press **Dry run** to see the plan: `READY`, or `REFUSE` with every reason.
4. Unlock.
5. Press **Load** and confirm.

Load progress phases: `stopping`, `copying`, `starting`, `waiting_health`, `ready`, `failed`, `timed_out`. The load timeout is 30 minutes. On a pair, if one side dies after it was seen up, the other side is stopped.

Placements:

- A single Spark (`head`, `worker`, `node3`, `node4`) is allowed when the weights plus the KV reserve (default 8 GiB) fit that Spark's budget and the files are there.
- A pair is for a model that does not fit one Spark; each side is charged half the weights plus the KV reserve.
- llama.cpp never runs on a pair.
- Multi-Spark runs use exactly two Sparks. With three or four Sparks any pair can be chosen; see [config.md](config.md#sparks-one-to-four).

## Reference

A recipe renders to argv lists. Nothing is joined for a shell. Sparksi fills hosts from `sparksi.toml` at render time, checks the result, and only then asks the node agent to exec it. `/api/switch` is a separate legacy path and does not use recipes.

### Cards

`GET /api/models` returns one card per library row:

| Field | Meaning |
|---|---|
| `format` | `nvfp4`, `mxfp4`, `fp8`, `bf16`, `gguf`, or `exl3` |
| `engines` | Preference order. `custom` is the fallback that runs a saved script |
| `files` | Weight path on each node that has one. A missing node is absent |
| `presence` | `test` via one probe per Spark: true, false, or null when the node could not be checked |
| `hints` | `moe`, `mtp`, `vision` |
| `flags` | `forbid` and `need`. `prefix_cache` in `forbid` keeps prefix caching off |
| `placements` | every single Spark (`head`, `worker`, `node3`, `node4`) and every pair this site has (`both` for head and worker, otherwise roles joined with `+`), each `allowed` with a reason when it is not |

An optional `~/.config/sparksi/cards/<id>.toml` overlays a row (repo, revision, format, weights, engines, hints, flags, ctx, timeout). That file is local and is not the catalog.

Placement:

- A single Spark is allowed where the weights plus the KV reserve (default 8 GiB) fit that node's budget and the file is there.
- The budget is UMA minus floor minus margin (121 − 6 − 4 = 111 GiB on the example) minus what is already listening.
- A pair is the placement when the model does not fit one Spark. Each side is charged half the weights plus the KV reserve. The first Spark of the pair in `[[nodes]]` order is rank 0 and serves the port; the second is rank 1 and starts first.
- llama.cpp never runs on a pair.
- A service that only has the `custom` engine (the image tools) stays on one Spark.

### Engines

Templates live in `sparksi.recipes.engines`. They return argv lists. Nothing is joined for a shell. Container names are `sparksi-<id>-<port>`.

| Engine | Single node | Two Sparks |
|---|---|---|
| vLLM | `vllm serve` in `nvcr.io/nvidia/vllm:26.05-py3`, util 0.80, fp8 KV, chunked prefill | `--tensor-parallel-size 2 --nnodes 2 --node-rank 0\|1 --master-addr <rank 0 fabric ip> --master-port 29501`, rank 1 `--headless`, no Ray. Rank 1 starts first |
| SGLang | `python -m sglang.launch_server` in `nvcr.io/nvidia/sglang:26.02-py3` | `--tp 2 --nnodes 2 --node-rank --dist-init-addr <rank 0 fabric ip>:50000`, same NCCL env |
| TensorRT-LLM | `trtllm-serve --backend pytorch` in `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6` | experimental, see below |
| llama.cpp | `llama-server` with `-ngl 99`, q8 KV, `--jinja`, and `--spec-type draft-mtp` when the card says mtp | not offered |
| custom | The catalog start script. Env is only `PORT` and `HOST` | The catalog start script, when that is all the row has |

Two-Spark vLLM and SGLang set `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, and `UCX_NET_DEVICES` to the fabric interface, `NCCL_IB_HCA` to the RoCE HCAs, `NCCL_IB_DISABLE=0`, and `NCCL_IGNORE_CPU_AFFINITY=1`.

Bind is that Spark's tailnet address from `[[nodes]] tailscale`. It is never `0.0.0.0`.

#### TensorRT-LLM on two Sparks

The single-node command is the documented `trtllm-serve` line. The two-node command follows the DGX Spark playbook: `mpirun trtllm-llmapi-launch trtllm-serve --tp_size 2 --backend pytorch` over the fabric, image `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6`.

The playbook reaches the second Spark only after a shell entrypoint installs OpenSSH and starts sshd. That entrypoint is not an argv list, so the two-node template is marked `experimental` and the plan repeats the reason. The recipe does not install packages, curl a script, or start sshd. The rank 1 container is started with the playbook's fabric environment and left running; rank 0 runs the verified `mpirun` line.

### Validation

Rejected before anything is started:

- unknown fields, unknown extra flags, values outside range (`gpu_mem_util` 0.3–0.95, `max_num_seqs` 1–16, `kv_cache_dtype` in the allowed set)
- a context length below 256, above 262144, or above the model's maximum when that maximum is known
- an image tag that is not in `[engines] allowed_images`, or any tag that says `latest`
- a port a protection preset puts off limits (8188 with `video-service`), or any port outside 8010–8099
- memory over the budget, a missing weight file, or `docker image inspect` failing on a target
- argv on the built-in refusal list, such as `tmux`, or anything that mentions `dgx-announce` (the legacy announcer)
- a port in `reserved_ports`, or a port saved on another model's recipe, or a port already listening on a Spark the run uses
- `docker run` with `--privileged`, `--pid=host`, `--cap-add`, `--security-opt`, `--network` other than `host`, `--device` other than a GPU node (`/dev/nvidia`, `/dev/dri`) or `/dev/infiniband` (the two-node TensorRT-LLM fabric mount), a bind whose source is `/`, `/etc`, `/root`, `/var/run/docker.sock`, or any path outside the configured cache root, or `--entrypoint` unless the argv is the two-node TensorRT-LLM recipe

The default port is the saved recipe port when that port is free, otherwise the catalog row's `port` when it is inside 8010–8099 and free, otherwise the next free port in 8010–8099. An off-limits port is never selected. Model Manager sends these fields and shows the argv the agent would exec. It does not send a command string.

Context is capped at the model's maximum when the card has `max_ctx` or `max_model_len`. Otherwise the cap is 2097152. `speculative` is a method, a draft-model path, and a token count (`mtp`, `dflash`, `dflash2`, `nextn`, `draft-mtp`). Tool parsers include `qwen3_coder` and `glm47`. Reasoning parsers include `glm45` and `qwen3`.

### Pinned recipes

`kind = "pinned"` stores the launch that was already working, per role (for example `head` and `worker`):

- `runtime = "docker"`: any image name except a `latest` tag, env, mounts, `network = "host"`, gpus, ipc, shm, container name, and the full argument list
- `runtime = "native"`: an absolute binary under a mount root, or `llama-server` on `PATH`, env (including `LD_LIBRARY_PATH`), and the argument list

The engine label is `vllm`, `sglang`, `trtllm`, `llamacpp`, `tensorfold`, or `other`. `imported_from` records the source and a UTC timestamp. Rendering runs those arguments. The only rewrite is `--host` / `--bind` when it is a listen address: it becomes that node's tailnet address from `sparksi.toml`. An mpirun host list is left as imported.

The form parses context, memory fraction or KV bytes, KV dtype, max sequences, parsers, speculative method / draft model / tokens, image, and port out of the arguments. Editing one of those rewrites the matching argument. Every other token is one row under Extra arguments. Env is one row per variable.

Mount sources must sit under the model cache or under that node's `mount_roots` (see `config/sparksi.example.toml`). `/`, `/etc`, `/root`, and the docker socket are refused. So are `--privileged`, `--pid=host`, `--cap-add`, `--security-opt`, a device other than a GPU node or `/dev/infiniband`, a network other than `host`, a bind of `0.0.0.0`, `::`, or `*`, port 8188, and anything that mentions `dgx-announce`.

An imported port outside 8010–8099, other than an off-limits port, is moved to the next free port in range. The recipe keeps a `port_note` and the page shows it. An off-limits port (8188 with the `video-service` preset) is refused, not remapped.

### Import

```
sparksi recipes import <model-id> [--from container:<name>|compose:<file>|unit:<name>|script:<path>] [--dry-run]
sparksi recipes import --all --dry-run
```

The controller reads over SSH and does not start or stop anything. `docker inspect` supplies image, command, env, mounts, and host config. `docker compose -f <file> config --format json`, `systemctl cat <unit>`, and a start script or `ps` listing are the other readers. With no `--from`, the catalog row's `launch` block and `identity.container` pick the source. `--dry-run` prints the pinned recipe and any validation problems. Without it, a valid recipe is written with the recipe store only.

### Services

A library row that is not a model server (an image tool, a loopback web app, a batch script) is listed under Services on Model Manager. Start and Stop run `launch.start` and `launch.stop`. There is no recipe step. The node agent accepts `service` only when the argv list is identical to one entry in `service_commands` in `sparksi.toml`. A row on port 8188, or named in the catalog's `never_touch`, is listed as Protected and has no Start or Stop unless `[services.<id>]` turns it on (see [config.md](config.md)).

### Defaults

Saved recipe (pinned or generated), then the catalog row, then the engine defaults. llama.cpp accepts `binary` and `ld_library_path` on the model.

`extra_flags` is an allowlist of whole tokens (`--trust-remote-code` and a few others). A flag is one argv element.

### Apply

The node agent accepts `run`, `stop`, and `status` with a JSON body (`sparksi-agent run '<json>'`, or the same object as `op` over SSH). It execs the argv list and returns at once. State for that node is `~/.local/state/sparksi/runs/<run>.json`.

`docker run -d` is waited only until the client returns. A foreground server such as `llama-server` is started in its own session. Stop runs the template stop argv (`docker rm -f` for a container) and signals the recorded pid.

The controller keeps the fleet record under `recipe-runs/` in the state dir and polls health with the same paths and phases as a model load (`starting`, `waiting_health`, `ready`, `failed`, `timed_out`). On a pair, if one side dies after it was seen up, the other side is stopped and `failed_side` says which one died.

### HTTP

Plan and every GET work with the lock on. Start and stop need the lock off (or the unlock secret). Recipe saves, renames, default changes and deletes do not take the lock; they accept same-origin JSON only.

| Method | Path | What it does |
|---|---|---|
| GET | `/api/models` | cards, presence, placements |
| GET | `/api/models/<id>/recipe?placement=&port=&engine=` | argv preview. `defaults` is the filled recipe: `values`, a `source` per field (`saved recipe`, `launch config`, `catalog`, or `engine defaults`), and a one-line `summary`. Query fields replace those values in the preview |
| POST | `/api/models/<id>/plan` | dry run. `conflicts` lists every problem. `verdict` is `READY` or `REFUSE` |
| PUT | `/api/models/<id>/recipe` | older route: writes the default recipe file only. Refuses (409) when the default is a pinned recipe; use Save as instead |
| GET | `/api/models/<id>/recipes` | the saved-recipe list and the default's name |
| POST | `/api/models/<id>/recipes` | Save as (body `name`, optional `overwrite`) |
| GET, PUT, DELETE | `/api/models/<id>/recipes/<name>` | read, save, or delete one named recipe |
| POST | `/api/models/<id>/recipes/<name>/default` | make it the default |
| POST | `/api/models/<id>/recipes/<name>/rename` | rename (body `new_name`) |
| POST | `/api/runs` | start. Returns a run id. Needs the lock off |
| GET | `/api/runs` and `/api/runs/<id>` | phase, health, per-node log tail |
| POST | `/api/runs/<id>/stop` | stop. Refuses off-limits ports and `never_touch` |
