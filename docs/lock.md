# The lock and confirmations

Sparksi starts locked. While the lock is on you can look at everything and plan anything, but nothing that starts or stops a model, or changes which machines are in the setup, will run. This page lists what the lock covers, how to unlock, and the confirmations the dashboard shows before it acts.

## What the lock covers

The lock is a lease file in the runtime state dir (`$XDG_RUNTIME_DIR/sparksi` by default). It is ON by default: a missing, corrupt or expired lease reads as locked.

Actions that need the lock OFF:

- Load, Unload and Restart on Model Manager (`POST /api/runs`, `POST /api/runs/<id>/stop`)
- Catalog service Start/Stop
- Legacy library run/stop/switch
- Stopping a model started outside Sparksi
- Network page changes (add, remove, role change)

Actions that do not need the lock:

- Looking at anything
- Dry run / plan
- Saving, renaming, setting default or deleting recipes
- Copying weights between the archive and a Spark, and deleting a local copy (a transfer queue)
- Options settings

## Unlocking

Press Unlock on the lock bar of Model Manager or Network. The lock opens for 15 minutes and then turns itself back on; the lock bar shows the time left. Press Lock to turn it back on at once. Lock and unlock are recorded on the Activity page.

When a page action is refused because the lock is on, the page says "Unlock first." and sends nothing.

## From the CLI and API

- `sparksi lock` turns the lock on.
- `sparksi unlock [--ttl SECONDS]` opens it for that many seconds (default 900), then it is locked again.

API callers other than the page need the lock off or the unlock secret. `sparksi start` and `sparksi stop` accept the global flag `--unlock`, which sends the unlock secret in a request header. The secret is read from `SPARKSI_UNLOCK`, or the file named by `SPARKSI_UNLOCK_FILE`, or the unlock file beside the lock. It is never printed, never a command argument, and never returned by a GET.

Without it a locked controller answers "Unlock first." and start/stop exit 2.

## Confirmations

Load, Unload and Restart on Model Manager each open the same dialog before anything is sent. It shows:

| Field | Value |
|---|---|
| Title | "Load <model>?", "Unload <model>?" or "Restart <model>?" |
| Model | the model's name |
| Machines | the Spark, or the two Sparks, the action runs on |
| Port | the port the model serves on |

Cancel has the focus and sends nothing. The lock and port checks run before the dialog opens and again when you confirm.

- **Load** uses the machines of the chosen placement. After you confirm, the plan runs first. A refused plan shows "Not loaded" with every reason and "Nothing was started."
- **Unload** uses the machines the run is on. For a model that is serving but was not started by a Sparksi run, the note says "Started outside a Sparksi run."
- Unload and Restart with changes are greyed out, with the tooltip "Not running", when the model has no live run and none of its containers is up. A failed or timed-out run still counts as live until every side is stopped, and a pair counts as up while either side is. The server agrees: `POST /api/library/stop` for a model that is not running answers 409 with `not_running: true` and runs no stop command, so stopped containers and other harnesses on the same port are left alone. When the controller cannot list containers, it lets the stop through and the page keeps Unload enabled rather than guess. If a stop errors on one side, the run is left failed with that side not stopped, and Unload again retries only the sides still up. Two clicks at once never send two stops.
- **Restart with changes** says "Stops this run, then loads it with the changes."

Other prompts:

- Deleting a recipe: "Delete recipe <name>? This cannot be undone."
- Opening another recipe with unsaved edits: "Discard unsaved changes?"
- Load default recipe after editing fields: "Reset these fields to the default recipe?"
- Network add, remove and role changes. Each says a backup is kept and nothing starts or stops. Removing a Spark also warns that recipes saved for it stop working until a Spark takes its role again.
- Stopping a model started outside Sparksi (when that is turned on): the model, machine, port and container
- Saving a GPU max-clock cap on Options while a model is running or decoding on that Spark. The page also warns not to change the clock during an active decode. The cap is applied with `sudo -n nvidia-smi` only (non-interactive, never a password prompt, and no other sudo use) and is not kept across a reboot. A sudo or nvidia-smi error is shown on the row. This action does not need Unlock.

## Things Sparksi never does

- Start, stop or probe anything on a port a protection preset puts off limits (8188 with the `video-service` preset; see [`[protect]`](config.md#protect))
- Listen on `0.0.0.0`, `::` or `*`, or on an address in the deny list that applies (see [Listen addresses](config.md#listen-addresses))
- Run `sparksi load`, `evict` or `install`; those verbs exist only to refuse
- Write the catalog or `sparksi.toml` from a run
- Send a shell command string from the browser (the page sends fields; the controller renders and checks the argv)