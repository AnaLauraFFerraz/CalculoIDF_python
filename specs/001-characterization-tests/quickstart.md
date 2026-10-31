# Quickstart: Characterization tests (001)

Validation guide for the finished feature. Formats: [contracts/](contracts/); entities:
[data-model.md](data-model.md); decisions: [research.md](research.md).

## Prerequisites

- Windows development machine: `uv`, Python 3.10.
- Reference environment: the GitHub Actions job (Linux, Python 3.10, pins of
  `requirements.txt`).
- No GCP credentials and no network are needed for the suite.

## Setup (development machine)

```powershell
uv venv --python 3.10 .venv
uv pip install -r requirements.txt -r requirements-test.txt
```

## Scenario 1 — Suite passes on a clean checkout (US1, FR-001, SC-002, SC-004)

```powershell
.venv\Scripts\python -m pytest
```

Expected: all 21 series pass (13 real + 8 synthetic) in `tolerant` mode, under 60 s; no
network access attempted. In CI the same command runs in `exact` mode.

## Scenario 2 — A numeric change is detected (US1 scenario 2, SC-003)

Change one disaggregation coefficient in `src/disaggregation_coef.py` by 0.001 (local, not
committed) and rerun in tolerant mode. Expected: failures listing series, field path, expected
and actual values. Revert. Repeat for the theoretical Gumbel coefficient `0.7797`
(`src/distributions.py:99`, series `chuvas_C_01944009_PL`) in tolerant mode, and for the BFGS
initial guess in `src/ventechow.py` in **exact** mode in a reference-equivalent Linux
environment, because it reaches post-fit fields only, which tolerant mode compares by structure
(research R-05, ADR 0003). Record the outcome of the three perturbations once, in `research.md`.

## Scenario 3 — Regeneration requires a reason (US2)

```powershell
.venv\Scripts\python scripts/update_golden.py
```

Expected: refused (exit 2), nothing written. With `--reason` outside the reference environment:
refused (exit 3).

In the reference environment, **before the merge** (feature branch): push a commit whose
message contains `[regenerate-golden]` and whose body is the reason, e.g.

```text
test: regenerate golden files [regenerate-golden]

initial characterization
```

The `generate-golden` job runs and uploads the artifact `golden-<run id>`. With an empty body
the job fails and uploads nothing.

**After the merge**, regenerations use the manual dispatch:

```powershell
gh workflow run characterization.yml --ref <branch> -f reason="<text>"
gh run list --workflow characterization.yml --limit 1
gh run download <run-id> -n golden-<run-id>
```

In both cases the downloaded diff shows the changed golden files and a new
`tests/golden/CHANGELOG.md` entry; the maintainer reviews and commits it.

## Scenario 4 — Edge cases frozen (US3, FR-007, FR-008)

Inspect `tests/golden/synthetic/S7.json`: `result_kind` is `string` with the insufficient-data
message and `metadata.marker` points to D-02 / C-02. The other synthetic golden files match the
expectations in research R-06 (differences from the hypothesis are reported, not fixed).

## Scenario 5 — Measurement and tolerances (FR-002)

```powershell
.venv\Scripts\python scripts/measure_env_diff.py --out measure.md
```

Expected: a per-field-group, per-station table of differences between Windows and the reference
golden files. For the pre-fit groups its maxima, times the safety factor, are the numeric
tolerances in `tests/golden/tolerances.json`; the post-fit groups are `structure` rules (ADR
0003); Scenario 2 still fails with those rules.

## Scenario 6 — Automatic protection (US4, FR-011)

Push a branch with a numeric change and open a PR to `main`. Expected: the `characterization`
check fails and, with the ruleset of manual step M-1 active, the merge is blocked.

## Scenario 7 — Determinism and no production change (FR-005, FR-010)

Run the suite twice in a row: identical outcome. `git diff main -- src/` is empty.
