# State machines

Sparksi keeps most lifecycles as one field written by one module. That is enough when the field is already the state and a second representation would only drift. The serving display is the exception: the monitor, the run record, and the page each implied a different answer, and the page showed an empty state for a live process.

The table below is the inventory. Only the serving display has a named state, one transition function, and a rejection for an illegal combination.

## Serving display

Owner: `sparksi.serving_state.reconcile`. The page renders these rows when its own run list has nothing to show. `GET /api/runs` adds a `serving` array. `sparksi runs --json` copies that array. Older clients that only read `runs` are unchanged.

There is one serving row per configured Spark: head, then worker, then Spark 3 and Spark 4 when present.

States: `idle`, `loading`, `ready`, `answering`, `split`, `observed`, `stale`.

Events: a monitor row arrives, a run record changes phase, or a host report becomes stale.

Transitions, in order:

- No live row for the role → `idle`.
- The announcing host is missing, or its `fresh` flag is not exactly true → `stale`. An empty host list is stale.
- A span has one side up and the other not confirmed → `split` on both roles, with an empty `run_id`. This is decided before any run is considered. The down side says it is not confirmed. The up side says it is up.
- A role is confirmed up, the served name was observed, the port matches, the run owns the role, and the parent phase is `starting`, `waiting_health`, `copying`, `ready`, or `stopping` → `ready` or `loading`, and `run_id` is that run (the newest start when several qualify; an equal start goes to the larger run id). The run owns the role when the served name is its model id or title. Two catalog rows or a recipe copy can serve one id, so the served name alone cannot tell their runs apart. Then the run also owns the role when the served name is one its model declares (the catalog row's served names, title, or served prefix, or a served name in any of that model's saved recipes) and the engine reports that role's container under the name the run recorded for that role, on a side that is not stopped. A served name the model never declares vetoes the container rule. Runs do not record the served name they started, so the declared names are the check.
- A role is still up and no active run owns it → `answering` when the served name was observed, otherwise `observed`. `run_id` stays empty. A failed, timed-out, cancelled, or stopped parent, including one whose nodes are only partly stopped, is not an owner.

On a span, a side is up only when its own flag is exactly true (`head_up` for the head, `worker_up` for the worker). A single-host row has no span. That role is up when the row's host is the role's node, the state is not `DOWN`, and the row's own `head_up` flag is true. On a worker row that flag means the worker answered, not that the head role is up.

A served name is observed only for a role that is up, and not when it is a static catalog hint. The production model file copies `served_name` onto a span whose head did not answer HTTP (`head_up` false, `container_source` `truth`). That string is not an identity. Container names are reused, so a name alone does not prove the instance either: it counts only when the engine reported it (not a static hint), for an active run on the same port and a served name that run's model declares. A stopped side never takes it, and a newer run that recorded the same host and name supersedes the older one, the same rule `run_live` uses. The page applies the same active-phase gate and veto when it picks the catalog row for a live process. It reads declared names from the catalog row and treats a run the server named on a serving row as declared, because recipe names are read on the server.

Invariants:

- `run_id` is empty for `split`, `answering`, `observed`, `idle`, and `stale`.
- Only an active parent phase can supply `run_id`. `failed`, `timed_out`, `cancelled`, and `stopped` never do, even when a node phase is still `started` or the container name matches.
- `head_up` is true only when the value is exactly true. A missing or false head does not confirm the head, and it does not make a copied `served_name` into an observation.

Previous encoding: the page computed a standing from booleans (`head_up`, `worker_up`, host age, parent phase, node phase) and only painted rows that matched a run. A span with `served` null, `head_up` false, and `worker_up` true matched nothing, so both roles rendered as nothing running. A stopped run on the same port was correctly hidden, and the live containers disappeared with it.

That shape is the regression fixture. The live row stays `split`. The stopped run is not the label.

Left in this machine on purpose: the page still computes the shared-run chain and the endpoint link from the monitor row. A split row with an empty bind and route has no link. Bind class and an advertised route still depend on the announcer process that produced the row. Restarting that process is outside this change.

## Parent run phase

Owner: `sparksi.recipes.runs`.

States: `starting`, `waiting_health`, `copying` is a node concern during launch, then parent `waiting_health`, `ready`, `stopping`, `stopped`, `failed`, `timed_out`, `cancelled`.

Events: start, health success, health failure, timeout, stop request, stop finished.

Invariant: one `phase` string on the run record. The watch loop is the only writer.

Left implicit. The phase string is already the state, and the watch loop is the transition. A second enum beside it would have two writers.

Known ambiguity: a parent can be `ready` while a node phase is still `started`. Callers that treat node `started` as "not ready" disagree with the parent. The serving display treats parent `ready` as `ready`.

## Node phase

Owner: the same run record, `nodes[].phase`.

States seen in practice: `started`, `stopped`, `failed`, `cancelled`.

Events: the agent reports the container, or stop finishes.

Invariant: a node phase does not outrank the parent. A parent that is `failed`, `timed_out`, `cancelled`, or `stopped` is not a current run, even when a node is still `started`.

Left implicit for the same reason as the parent. The serving display reads the parent phase, and it does not treat a node phase as a second run.

## Adoption

Owner: `sparksi.recipes.adopt`.

States: no record, adopted run at `ready`, or left alone because a run already covers the containers.

Events: a container is already up when the controller looks, or a later pass finds the same containers.

Invariant: adoption does not start or stop a process. It records what is already up.

Left implicit. The outcome is a normal run record, and the tests cover the adopt-or-skip choice.

## Jobs and switch

Owner: `sparksi.jobs`.

States: `idle`, `queued`, `paused`, `cancelled`, `ready`, `failed`, plus copy phases named by host.

Events: queue, pause, cancel, finish.

Invariant: one current job. A restarted process does not invent success for an in-flight job.

Left implicit. The job dict's `phase` is the state, and switch still goes through the same job gate. The serving bug was not a job-phase bug.

## Lock and unlock TTL

Owner: `sparksi.lock`.

States: locked, unlocked until a timestamp, expired back to locked.

Events: acquire, release with a TTL, clock passes the timestamp.

Invariant: control routes stay refused while the lock is held. Unlock is a lease, not a permanent open.

Left implicit. The lock file is the state, and a parallel state machine would be a second source of truth on a safety check. Behavior is unchanged.

## Announce freshness

Owner: `sparksi.dash` and the monitor snapshot.

States: fresh (a packet within the stale window), no announce, unreachable when both hosts are silent.

Events: a packet arrives, or the age crosses the window.

Invariant: a stale host does not confirm a role. The last packet can still be kept for display.

Left implicit. Age against one window is the whole machine. The serving display consumes the host `fresh` flag and moves to `stale` instead of inventing a second timer.

Announcer bind detection and an advertised route are properties of the packet. They appear only after each announcer process is running a build that fills them. This change does not restart those processes and does not rewrite unit files.

## Telemetry archive

Owner: `sparksi.history` and `sparksi.telemetry`.

States: empty file, appending, serving a bounded read.

Events: a monitor sample is kept, a reader asks for a window.

Invariant: writes append, and a read is bounded. A missing file starts empty rather than failing the page.

Left implicit. There is no phase to mis-label. The bug in front of operators was serving display, not the archive.

## Stage read-only

Owner: `sparksi.serve` `STATE.readonly`.

States: read-only, or normal.

Events: the stage feed is the process, or it is not.

Invariant: read-only refuses plans, runs, and writes. It does not attach a live serving reconciliation, so a stage view cannot grow a control signal from the inbox.

Left implicit. One boolean is the machine.

## CLI wait

Owner: `sparksi.cli_api` start and stop commands.

States: the same run phases, polled until `ready`, a failure phase, or a quiet phase.

Events: each `GET` of the run.

Invariant: the CLI reports the controller's phase. It does not invent a local phase.

Left implicit. It is a client of the parent run phase. `--json` for `runs` now also includes `serving` when the controller sends it.

## Recipe draft, defaults, and preview

Owner: the overview page, in memory only.

States: clean, edited, draft held for the selected model, preview in flight (a serial), defaults applied.

Events: a field edit, a model change, Load default recipe, a preview response.

Invariants: a preview response for an older serial is dropped. Defaults replace the form only when the payload is for the selected model. A draft is cleared only after that succeeds. Values typed for one model are not reused as another model's saved port. Nothing is written to browser storage.

Left implicit. The serial and the per-model draft are the state, and the recent fixes are tests of those rules. Folding them into the serving machine would mix two lifecycles.

## Load and unload progress

Owner: the models page.

States: progress section closed, opened because the selected card's run just entered an active or error phase.

Events: a phase change on the same card. A historical failure, or switching cards, does not open it.

Left implicit. It is a disclosure rule on top of the run phase, already covered with the models page.

## Collapsible sections

Owner: the models page details elements.

States: open or closed. Extra-argument groups start closed.

Left implicit. There is no transition beyond the element the browser already stores for the visit.

## Chart focus

Owner: the overview chart.

States: no pin, or one role pinned.

Events: click a legend button, click it again to clear. Hover dims only while nothing is pinned.

Left implicit. One string is the state, and the button's pressed state mirrors it.

## Chart time

Owner: the overview chart.

States: live ring, a hole after the contiguous cursor, a reseed of one machine, a panned window, hidden.

Events: the page loads and seeds 20 minutes, a monitor response carries the newest sample, that sample skips seconds or arrives into an empty ring, a history response covers a hole, the reader pans or returns to live, the document becomes hidden or visible.

Invariant: a contiguous live ring does not request history. A hole keeps its cursor on the last timestamp with no hole after it, and the page requests `since` that cursor until a history response covers the machine. The jumped tip does not move the cursor. A fresh sample into an empty ring, or a jump wider than 20 minutes, reseeds that machine. A sample older than the live window does not. Returning to live or unhiding does not request `since` when the cursor already matches the tip. A hidden document does not tick. One history fetch runs at a time. The ring, the cursors, and the window cache stay in RAM. The server keeps its one-hour hot ring.

Left implicit. The follow flag, the per-machine cursor, and the window cache are the state. Legend focus stays a separate flag.

## Scroll position

Owner: the overview refresh.

This is not a lifecycle. Refresh replaces gauge nodes without moving the window. It stays a layout invariant, not a state machine.
