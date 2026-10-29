# Contract: maintenance scripts

## `scripts/update_golden.py`

```text
python scripts/update_golden.py --reason "<text>" [--series ID ...] [--allow-non-reference]
```

| Argument | Behavior |
|---|---|
| `--reason` | Required, non-empty after trimming. Missing or empty → exit code 2, nothing written (US2 scenario 1). |
| `--series` | Optional list of series IDs; default: all real and synthetic series. Unknown ID → exit 2. |
| `--allow-non-reference` | Allows running outside the reference environment; writes to a scratch directory given by `--out`, never to `tests/golden/`, and adds no changelog entry. |

- Outside the reference environment and without `--allow-non-reference` → exit 3, nothing
  written.
- On success: rewrites the selected golden files, appends one entry to
  `tests/golden/CHANGELOG.md` with the changed field paths per series (US2 scenario 2), prints
  a summary, exit 0. If no golden file changed, it says so and adds no changelog entry.

## `scripts/measure_env_diff.py`

```text
python scripts/measure_env_diff.py [--series ID ...] [--out report.md]
```

- Runs `main.main` for each selected series in the current environment and compares with the
  golden files **without tolerances**.
- Reports, per field group of `tolerances.json` and per series: max absolute and max relative
  difference, number of differing leaves, and any non-numeric difference (which is a failure,
  not a tolerance matter).
- Writes nothing under `tests/golden/`. Its output is pasted into `research.md` (Measurement
  results) and used to fill `tolerances.json`.

## `scripts/make_synthetic_fixtures.py`

```text
python scripts/make_synthetic_fixtures.py --out tests/fixtures/synthetic
```

- Deterministic (fixed seed). Writes the 8 CSVs in Hidroweb format (ISO-8859-1, `;`, `,`
  decimal mark, 12 header lines). Run once; the CSVs are committed and treated as golden inputs
  afterwards.

## Workflow `characterization.yml`

| Trigger | Job | Effect |
|---|---|---|
| `push` (any branch), `pull_request` → `main` | `characterization` | `pytest` with `CALCULOIDF_COMPARE_MODE=exact`; required status check on `main` (manual step M-1) |
| `push` whose head commit message contains `[regenerate-golden]` | `generate-golden` | reason = commit message body (marker removed, trimmed), passed via environment variable; runs `update_golden.py`, uploads `tests/golden/` as artifact `golden-<run id>`; never commits. Works on the feature branch before the merge |
| `workflow_dispatch` (inputs `reason`, `series`) — only once the workflow is on `main` | `generate-golden` | same as above, reason from the input; `gh workflow run characterization.yml --ref <branch> -f reason="..."` |
