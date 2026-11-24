# PR lessons

Read this before you start a change. When a review finds a miss, add
one line: what was missed, and the check that would have caught it.

- Apply a rule on every route that does the same job, not only the path
  in the report: the older API route, rename and Save as next to Save,
  and every writer of the same field.
- Keep the page and the Python copies of a rule in agreement.
- A failed, timed_out, cancelled, or half-up run can still have live
  sides; Unload, Restart, and stop must handle those, not only a clean
  running state.
- A retry path and the first-try path each need their own test.
- When a field is renamed, mirror the old name in every changed output.
- A new browser test is not in CI until it is added to the WebKit list
  in `.github/workflows/ci.yml`.
- A docs claim needs a line of code (or a test) that supports it.
- Do not open a pull request until the local acceptance commands have
  passed and, when the branch name is in the workflow `push` filter, CI
  on the branch is green.
- A rerun after a failure is not a pass. If any check fails, even once,
  find the cause and fix it before you report. Re-running a failed job
  does not count as passing.
- Problems outside the stated scope get one line here and are
  named in the report. They are not started in the same job.

## Backlog

- `tests/test_ui.py`: `httpd.shutdown()` can hang while the server thread is blocked in `accept()`; fix the race, no retry.
- Under systemd every log line reaches the journal twice (stderr and `SysLogHandler`); use one sink when `JOURNAL_STREAM` is set, with a test.
- Hash-pin the Playwright npm install: `package.json`, `package-lock.json` and `npm ci`.
- Add job timeouts to `release.yml`, `codeql.yml` and `scorecard.yml`, and hash-pin the `pip install build` in `release.yml`.
- Add a Dependabot entry for the `/.github` pip requirements.
- Pin runner images (for example `ubuntu-24.04`) across all workflows before `ubuntu-latest` moves to Ubuntu 26.
- Break equal-start ties on run id in the other newest-run pickers too (`library.newest_run`, page `newestRun` and `displayedRuns`), as `serving_state` does.
- Cache `serving_state._recipe_served` by recipe file mtime instead of reading every saved recipe on each `/api/runs` that needs a container to decide.
- `tests/test_refresh_after_write.py`: wait on the refresh thread's done event instead of the fixed 4 s `_wait_for` budget, which a scheduler stall can exhaust.
- With a bad `sparksi.toml` the problem prints twice (the import-time WARNING and the exit message); keep one. Also `python -m sparksi.stage_feed` and `evict` do not go through `cli.main`, so a bad file is not refused there (untested).
- A run record can stay `ready` after both of its containers exit outside Sparksi; mark it gone (or stale) when every side's container has left the fresh container lists.
- `heartbeat.answer_probe` answers a loopback probe from loopback and wildcard listeners only; give it the agent's own-tailnet fallback so a beat and an SSH probe agree on a server bound to the tailnet address.
- `monitor.models._engine_port_tp` and the `monitor/peer.py` shell collector still read TP only from `TP=` / `TENSOR_PARALLEL_SIZE=`; give them the announcer's `--tp`, `--tensor-parallel-size` and `WORLD_SIZE` forms.
- The memory estimate for a two-Spark pinned run can read about 20 GiB higher than the live peak (97 vs 117 GiB in one case); find which term (KV reserve, util or margin) is counted twice, with a test. Out of scope for the recipe-folder PRs.
- A service opted in through `controllable_service` in `sparksi.toml` can still be started from the services list after its model gets a recipe folder; decide whether D4 should cover that private opt-in as well. Out of scope for the recipe-folder review.
- A Spark copy of a recipe folder version that differs is refused forever and never repaired; add a command that moves that one version folder aside on the Spark so the next Load copies it again. Out of scope for the recipe-folder review.
- `sparksi apply` starts resident systemd units, which are not library models and have no recipe folders today; if a resident ever gets one, give `apply.build_plan` the same D4 refusal. Out of scope for the recipe-folder review.
- `library_sha256` pins only libraries the binary names (and theirs) today; a library loaded with dlopen, or one added to the build tree later, is not seen until the binary is bundled again. Out of scope for the native-libraries PR.
- `library_sha256` does not record a DT_NEEDED entry written as a path (it holds a `/`), and a native binary under `/usr/bin` inside a mount root would treat all of `/usr` as its build tree; decide whether either needs handling. Out of scope for the native-libraries review.
- Outside the weights root, `files_read` still checks only the last path component for a link, so a file reached through a linked folder is read from wherever the link points, even outside the mount roots; decide whether those reads should use the same no-follow walk as weights-root files. Out of scope for the bundle-carry-files review.
- GB10 on driver 580.x prints no locked-clocks section in `nvidia-smi -q -d CLOCK`, so the applied GPU cap cannot be read back from the GPU after a dashboard restart or a Spark reboot; find a read-only source for the live lock (NVML or a future driver) and drop the in-memory memo. Out of scope for the GPU clock cap review.
- "Head and worker share this run" was reported on the Overview while an outside server ran on only one Spark, but the link carries `hidden` and is not shown; the report came from a page text dump that included hidden elements. Check visibility (a screenshot, or `hidden`/computed style) before logging a display bug. No change needed (emergency stop PR).
- A server started outside Sparksi whose launcher wraps the listener (a script or supervisor that is its parent, not its group leader) survives the emergency stop and may start it again; decide whether to offer stopping the launcher as well. Out of scope for the emergency stop PR.
- The emergency stop protects a web front end only by a built-in name list (Open WebUI) and the catalog's `never_touch.do_not_stop`; a front end that answers like a model under another name would be offered. Out of scope for the emergency stop PR.
