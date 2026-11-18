# Agents

Sparksi is the Peelsa model manager for NVIDIA DGX Spark clusters of one
to four Sparks. Setup, the CI checks, code rules and public-safe rules
are in [CONTRIBUTING.md](CONTRIBUTING.md). Follow them; this file adds
how an agent finishes a change.

Tests use fakes and fixtures. Do not SSH to a machine, do not start,
stop or swap a model, and do not call a live `/api/runs`.

## Guardrails

These rules apply to every task.

- **Scope.** Before you start, write the goal and the files or area
  you may change. Ask before going past that list.
- **Done.** Write the success criteria first. The work is done only
  when those criteria are met and you have the real output that
  proves it. A claim without output is not a pass. An independent
  review happens before anything is shared or merged.
- **Retries.** A failure that later passes only on retry is not a
  pass. Find the cause and fix it. Do not weaken, skip, or delete a
  check to make it pass.
- **Findings.** Problems inside the stated scope are fixed before
  you report. Problems outside the scope get one line in the backlog
  or lessons file and are named at the end of the report. Do not
  start a new branch or a new task for them in the same job.
- **When to ask.** Stop and ask a human before spending money,
  touching a live system, doing anything irreversible or
  destructive, or changing the scope or the plan.
- **Reports.** Reports state facts and evidence. Do not offer
  "want me to…" for work that is already in scope; do that work.
  Out-of-scope items are listed, not started.
- **History.** Do not force-push or rewrite history. Identity and
  secrets rules are unchanged and live elsewhere in this
  repository.
