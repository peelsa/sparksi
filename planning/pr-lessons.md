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
