# Stage feed

A second process can record the live dashboard without attaching to its
state or its announce port. It is for a staging checkout on the same
machine. It does not unlock, start a job, or send announce packets.

```text
python -m sparksi.stage_feed --upstream http://127.0.0.1:7500 --state-dir /tmp/sparksi-stage
```

* `--upstream` is loopback HTTP. The feed GETs `/api/monitor` only.
  Redirects and proxies are ignored, and the response must still be that same
  loopback path. The monitor card is the announce reading the dashboard already
  uses. It does not adopt a container or write a run file.
* `--runs-snapshot` is an optional local JSON file with a `runs` list. The
  stage does not call the live runs endpoint, because that endpoint adopts
  containers and stores run records. Without the file, the stage shows no runs
  and says they were not checked. Rows from the file are marked stale. A link
  into the live state directory is refused. Host names, logs, and paths in the
  file are dropped.
* `--state-dir` is required. It must be a new empty directory. A directory that
  already holds files is refused. So is the live runtime directory, the durable
  state directory, and a link to either of them. History, telemetry, the lock,
  the token ledger, the job file, and recipe runs are created there and nowhere
  else. State overrides other than the config paths are dropped first.
* The listener is `127.0.0.1:7600`. There is no announce listener.
* Production config is read. It is not written.
* POST, PUT, and DELETE are rejected before any lock or model call. A job file
  copied into the directory is not resumed. Token polling and model warm-up stay off.

`STATE.monitor_fn` alone does not feed the collector. This module sets
`observe_fn`, which is what the history loop records. `GET /api/runs` on the
stage port returns the local snapshot, not a live adopt. The process is
read-only: every POST, PUT,
and DELETE, including unlock, is rejected before it runs. Job restore, token
polling, and model warm-up stay off, so a copied job file cannot start work.
A missed poll drops the last card. Stale readings are not archived as fresh.
