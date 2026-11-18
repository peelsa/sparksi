## Summary

<!-- What changed, and why. -->

## Contract

- **Goal:**
- **Root cause:**
- **Allowed files:**
- **Edge cases and tests:**
- **Sibling sweep:** commands run and what they hit
- **Out of scope:**

## Tests and guards

Paste the last lines and the exit code for each command.

- [ ] Tests cover this change and use fakes or fixtures
- [ ] `python3 -m pytest -q` — exit code:
- [ ] `python3 scripts/check_identifiers.py` is clean — exit code:
- [ ] `python3 scripts/check_stdlib.py` is clean
- [ ] `python3 scripts/check_ui_colors.py` is clean when the UI changed
- [ ] `ruff check src tests scripts` is clean
- [ ] Sibling sweep done
- [ ] Whole diff re-read independently
- [ ] No check passed only on retry; any failure was traced and fixed
- [ ] In-scope issues are fixed
- [ ] Out-of-scope findings (if any) are one line in `planning/pr-lessons.md` and listed below, not started in this PR

## Screenshots

<!-- Placeholder data only: spark-a, spark-b, 100.64.0.x, 192.0.2.x, /home/USER.
     Leave this section empty when there is nothing to show. -->

## Sign-off

- [ ] Every commit has a Developer Certificate of Origin sign-off (`git commit -s`). The text is https://developercertificate.org/
