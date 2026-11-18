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
- `library.newest_run`, the page `newestRun` and `displayedRuns` let the later record win on equal start times; break ties on id, like `serving_state` does.
- `serving_state._recipe_served` reads every saved recipe on each reconcile when the container rule is needed; cache it by recipe mtime if recipe counts grow.
