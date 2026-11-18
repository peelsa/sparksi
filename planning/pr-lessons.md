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
- Anything that comes up in the work — a failing or flaky test, a
  warning, or a broken path you touched or saw — gets fixed in this
  job before the final report, on its own branch or pull request if it
  is out of scope. Do not leave it as an offer or a question. The
  final report has no "came up", "want me to", or follow-up items.
  Ask the owner only for spending money, model or live-runtime
  operations beyond the brief, the five-file or public-API limits, or
  a change of plan.

## Backlog

- `tests/test_ui.py`: `httpd.shutdown()` can hang while the server thread is blocked in `accept()`; fix the race, no retry.
- Under systemd every log line reaches the journal twice (stderr and `SysLogHandler`); use one sink when `JOURNAL_STREAM` is set, with a test.
- Hash-pin the Playwright npm install: `package.json`, `package-lock.json` and `npm ci`.
- Add job timeouts to `release.yml`, `codeql.yml` and `scorecard.yml`, and hash-pin the `pip install build` in `release.yml`.
- Add a Dependabot entry for the `/.github` pip requirements.
