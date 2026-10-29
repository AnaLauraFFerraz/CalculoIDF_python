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

- The reference environment is `REFERENCE_FINGERPRINT` in `tests/characterization/compare.py`
  (`Linux`, `x86_64`, Python `3.10`, and the numpy, pandas, scipy, scikit-learn versions read
  from `requirements.txt`). When the current fingerprint does not match it and
  `--allow-non-reference` is absent → exit 3, nothing written.
- A golden file is rewritten only when `result_kind`, `result`, `metadata.input_sha256` or
  `metadata.marker` change; `generated_at` alone does not count, and unchanged files are left
  byte-for-byte (contract golden-file.schema.md).
- On success: writes the changed golden files and appends one entry to
  `tests/golden/CHANGELOG.md` (US2 scenario 2) containing a Markdown table with one row per
  changed series — series id, number of changed leaves, max absolute delta, max relative delta —
  and the changed field paths per series. It prints the same table to stdout, ready to paste
  into the PR description (constitution principle I). Exit 0. If no golden file changed, it says
  so and adds no changelog entry.
- Testability: the script's entry function takes the golden root and the reference fingerprint
  as internal parameters, used only by its unit tests (`tmp_path`, `monkeypatch`). They are not
  CLI flags; the CLI always uses `tests/golden/` and `REFERENCE_FINGERPRINT`.

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
| `push` (any branch) | `characterization` | `pytest` with `CALCULOIDF_COMPARE_MODE=exact`; required status check on `main` (manual step M-1, with "require branches to be up to date"). Not run on `pull_request` nor on `workflow_dispatch`; a PR's required check is satisfied by this check on its head SHA |
| `push` to any branch whose head commit message contains `[regenerate-golden]` | `generate-golden` | reason = commit message body (marker removed, trimmed), passed via environment variable; runs `update_golden.py` for every series (no `--series` on push), uploads `tests/golden/` as artifact `golden-<run id>`; never commits. Works on the feature branch before the merge |
| `workflow_dispatch` (inputs `reason`, `series`) — only once the workflow is on `main` | `generate-golden` | same as above, reason from the input; `gh workflow run characterization.yml --ref <branch> -f reason="..."` |
