# Dashboard pages

Every page has the same header: the Sparksi wordmark, a Menu button that opens the sidebar, a strip of small cards with one card per configured Spark (GPU and memory gauges, memory used of total GiB, temperature in F), today's token totals, and how long ago the last report arrived.

The sidebar links are Overview, Model Manager, Library, Archive, Network, Options, Token totals, and Activity. The URLs are `/overview` (also `/`), `/models`, `/library`, `/archive`, `/network`, `/options`, `/tokens`, and `/activity`.

## Overview (`/overview`)

The serving boxes at the top show what each Spark is running: the model name, the state (for example Ready), the port, and the endpoint URL with a Copy button. When two Sparks share one run, a link between their boxes says so.

Each Spark's box also has a GPU max-clock field in MHz and a Save button. The current clock and the cap already applied are shown as read-only. An empty field means no cap. Save needs the lock off. If a model is running on that Spark, Save asks first. The cap is applied immediately with sudo -n nvidia-smi (non-interactive, and no other sudo use) and does not return after a reboot. A failed command, including a sudo or permissions error, is shown on the box. The allowed range comes from that GPU's own report.

"Models started outside Sparksi" lists models answering on a Spark that Sparksi did not start. The Stop button for them is off unless `sparksi.toml` sets `[controller] stop_unmanaged = true`. When it is on, Stop asks first and needs the lock off.

The fleet chart shows GPU, CPU, memory, and temperature per Spark. Now is on the left and the window is 20 minutes wide. Older and Newer buttons pan back through 24 hours; Live returns to the present. Temperature uses the same 0-100% axis as the other lines, spanning 0 F to 200 F.

Monitored inference hosts (up to two extra machines added on the Network page) appear as cards.

Tiles show Fleet decode, Memory in use, Fleet power (estimate), and Archive available.

Temps and JSON sections fold open at the bottom of the page.

## Model Manager (`/models`)

The lock word sits at the top with a Lock / Unlock button.

The model list has a search box and a filter: All, Can run now, Needs copy, Archive only. A Services block lists catalog services.

Selecting a model shows a status card per Spark and the recipe bar: New, Save, Save as..., the open recipe name with "unsaved changes", Dry run, and Load. Below the recipe bar is the saved-recipe list (see [recipes.md](recipes.md)), then Unload and Restart with changes.

"Load default recipe" resets the form to the default recipe. It asks first if you edited fields. It does not start anything.

Setup steps run in order: 1 Model (size, memory estimate, files, Engine), 2 Placement, 3 Port (8010-8099), 4 Recipe fields, then Command preview (the exact argv) and Load progress (timeout 30 min).

Load, Unload, and Restart open a confirmation (see [lock.md](lock.md)).

## Library (`/library`)

A table of catalog models with columns: name, GiB, archive, and one column per Spark showing whether the weights are there. Click a row to select it; Ctrl/Cmd-click to select several. A job line shows the current copy with Pause and Cancel.

## Archive (`/archive`)

The archive is the storage attached to one Spark (`[archive]` in `sparksi.toml`, or Storage machine on Options).

- The top lines show the storage name, its free and used space, and the copy that is running, if any.
- The list shows each archived model with its size. A complete model has a "Copy to <Spark>" button for the head and the worker; an incomplete one is marked "incomplete".
- "Sync with storage" has a Check sync button that compares each Spark with storage. It only reads. Where a Spark is missing a model or some shards, the cell has a Copy (or "Copy missing shards") button.

Copies run as a transfer queue, one at a time, and do not need the lock.

## Network (`/network`)

Discover looks at this machine's tailnet, its LAN neighbors, and mDNS, then asks common inference ports what they serve. It only reads: one GET per port, no logins.

"This setup" shows the lock bar, then rows for Controller, Head Spark, and Worker Spark, plus Spark 3 / Spark 4 when configured. Each row has a name, an address, and a role selector. A line counts the Sparks, for example "2 of 4 machines." Only the last Spark has a Remove button; the head never does. Add a machine by address (Address, Port default 8000, Name optional, "Add to monitor") to watch an inference host that is not a Spark, up to two.

Every change asks first, edits only the `[[nodes]]` and `[[endpoints]]` tables in `sparksi.toml`, keeps a timestamped backup, needs the lock off, and starts or stops nothing. Sparksi says "Restart Sparksi to use the new setup."

Discover lists each machine it found with its addresses, where it was seen, and any inference endpoint that answered (engine, URL and models), or "No inference port answered." A found machine that is not in the setup has an "Add to Sparksi" button with a role choice; it asks first like any other change.

## Options (`/options`)

Optional settings. Each is saved one line at a time in `sparksi.toml` with a backup; nothing starts or stops and the lock is not needed.

- Storage path (`archive.root`)
- Storage name (`archive.label`)
- Storage machine (`archive.node`, empty means the head)
- Monitor mirror (`controller.monitor_upstream`: read the monitor from another Sparksi on loopback or the tailnet)
- Request log (`controller.access_log`: slow, all, off)
- Metrics database (`controller.metrics_url`, VictoriaMetrics; empty uses `http://127.0.0.1:8428`)

Settings that need it say "Restart Sparksi to use the new setting."

## Token totals (`/tokens`)

Per model: today out, today in, lifetime out, lifetime in. Today is the controller's local calendar day.

## Activity (`/activity`)

A bounded log of the last 500 events of what the controller did: lock and unlock, loads, stops, copies, failures, announce loss, and a GPU temperature crossing 80 C (cleared below 75 C).