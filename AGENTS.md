# Agents

Sparksi is the Peelsa model manager for NVIDIA DGX Spark clusters of one
to four Sparks. Setup, the CI checks, code rules and public-safe rules
are in [CONTRIBUTING.md](CONTRIBUTING.md). Follow them; this file adds
how an agent finishes a change.

Tests use fakes and fixtures. Do not SSH to a machine, do not start,
stop or swap a model, and do not call a live `/api/runs`.

## Definition of done (before you open a pull request)

A pull request is the finished change, not a checkpoint. Open it only
when every step below is done. Push nothing new to it afterwards except
fixes that a reviewer or CI asks for.

### 1. Write the contract first

Put this at the top of your notes and, later, in the PR body:

- **Goal:** the bug or behavior, in one or two sentences.
- **Allowed files:** the files you expect to change. If you need a file
  that is not on the list, stop and say why before you touch it.
- **Acceptance commands:** the exact commands that must pass (see step 6),
  plus the new test that proves the fix.
- **Edge cases:** every case a reviewer would try. For Sparksi that
  usually means one, two, three and four Sparks; head, worker and pair
  runs; failed, timed_out, cancelled and half-up runs; imported or
  adopted containers; pinned recipes, copies and older files written
  before this change; lock on and off; an unreachable machine; a second
  click or request while the first is still running.
- **Stop and ask when:** more than five files change, a public API or
  config key changes, or the fix needs a change of plan.

### 2. Reproduce

Show the bug with a failing test (or, for docs, the line of code the
docs contradict). Keep the failing output. A fix without a test that
failed before it is not done.

### 3. Fix the cause, not the symptom

Explain the root cause in one sentence. Fix it where it starts, server
and page both when both carry the rule.

### 4. Sweep for siblings

Search for the same pattern everywhere before you call it fixed:

```bash
rg -n '<function, field or flag you changed>' src tests scripts docs
```

Check every caller and every other route that does the same job: the
older API route, rename and Save as next to Save, the page and the
Python copy of the same rule, the retry path next to the first try,
every output that carries a renamed field. Fix them in this PR, or list
them under "Not in this PR" with a reason. Record what you searched.

### 5. Edge-case pass

Write one test, or one line in the PR body saying why none is needed,
for each edge case from step 1.

### 6. Run everything locally, raw output only

From the repo root:

```bash
python3 scripts/check_identifiers.py .
python3 scripts/check_stdlib.py .
ruff check src tests scripts
python3 scripts/check_ui_colors.py
python3 -m pytest -q
```

Also run `python3 scripts/check_design.py src/sparksi/static/index.html`
when you touch the page and `IMPECCABLE_BIN` is set, and
`python3 scripts/check_wheel.py` when packaging changes.

Run the WebKit tests from `.github/workflows/ci.yml` when you touch the
page. Add any new browser test to that list in the same commit.

A step passes only when you have its real output and exit code 0. Paste
the last lines and the exit codes into the PR body. "Should pass",
"passes on my side" and "no new failures" are not a pass. A rerun after
a failure is not a pass either: if any check fails, even once, find the
cause and fix it before you report. Re-running a failed job does not
count as passing. Never weaken, skip or delete a test or an assertion
to get green; if an old assertion encoded the bug, say so in the PR
body.

### 7. Fresh-eyes review of the whole diff

Read `git diff origin/<base>...HEAD` from top to bottom as if someone
else wrote it, with the contract and edge-case list beside you. Look for:
files outside the allowed list, debug code, leftover TODOs, missing
tests, changed behavior for two-Spark sites, a guard checked in the page
but not on the server, and identity leaks. If a second agent can
review, give it only the contract and the raw diff, not your summary.

### 8. Zero known issues

List everything you found in steps 4 to 7. Fix each one now, or stop and
report it without opening the PR. Do not open a PR whose body says
"known issue", "follow-up", "WIP" or "part 1 of 2" unless the maintainer
asked for that split in writing.

Anything that comes up in the work — a failing or flaky test, a
warning, or a broken path you touched or saw — gets fixed in this job
before the final report. If it is out of scope, fix it on its own
branch or pull request first. Never leave it as an offer or a
question. The final report has no "came up", "want me to", or
follow-up items. Ask the owner only when the decision is theirs:
spending money, model or live-runtime operations beyond the brief, the
five-file or public-API limits, or a change of plan.

### 9. Open it ready

Finish steps 1-8 and run the local acceptance commands first. Then push
the branch. Where the `push` filter in `.github/workflows/ci.yml` lists
branch patterns, name the branch to match one, so CI runs before the PR
exists, and wait for it to go green. Where the filter lists only `main`,
CI runs on the pull request (and again on main after merge): open it
and wait for those checks. If a check fails, fix the cause; a rerun is
not a pass.

Open the PR as ready for review with the PR template filled in:
contract, root cause, sweep results, edge cases and tests, and raw
check output. One topic per PR. Do not merge.

### After the PR is open

- Push only fixes for review or CI findings. New scope goes in a new PR
  after this one merges.
- A failed check, even one that later passes on a rerun, is not green.
  Find the cause and fix it; do not rerun a failed job to get a pass.
  Findings that come up after the PR opens are fixed in this job
  before the final report, not offered as a follow-up.
- When a review finds something you missed, add one line to the lessons
  file (`planning/pr-lessons.md`; read it before step 1 if it exists):
  what was missed, and the check that would have caught it.
