# Implementation Plan: Characterization tests for the IDF computation (golden files)

**Branch**: `001-characterization-tests` | **Date**: 2026-10-29 | **Spec**: [spec.md](spec.md)

**Status**: Approved (2026-10-29)

**Amended**: 2026-10-29 — CI trigger aligned after /speckit-analyze (A1): `characterization` runs on push only; the required check is satisfied on the head SHA; CPU kernels pinned from the start (R-03).

**Input**: Feature specification from `specs/001-characterization-tests/spec.md`; technical notes
from `docs/knowledge/07-technical-notes-by-feature.md`, section "001 — Characterization tests".

## Summary

Freeze the current output of `main.main(csv_path)` for the 13 real Hidroweb CSVs and for
synthetic edge-case series, as golden files generated in a **reference environment equal to
production** (Linux, Python 3.10, the pins in `requirements.txt`). A pytest suite reruns the
pipeline and compares the full output recursively: **bit-for-bit** in the reference environment
(the CI job), **per-field tolerances** anywhere else (e.g., the Windows development machine).
The tolerance values come from an explicit measurement of the Windows-vs-reference differences
on the 13 CSVs. A script regenerates golden files only with a recorded reason and appends to a
versioned changelog. A GitHub Actions workflow runs the suite on every push; a
repository ruleset (manual step) makes it a required status check on `main`. Production code is
not touched: the import-time `storage.Client()` is neutralized by a test-side stub of
`google.cloud.storage`.

## Technical Context

**Language/Version**: Python 3.10 (production runtime and reference environment); the same
version on the Windows development machine via `uv`.

**Primary Dependencies**: production pins unchanged (`requirements.txt`: pandas 2.0.1,
numpy 1.24.3, scipy 1.10.1, scikit-learn 1.3.0, Flask, google-cloud-storage). Test-only:
`pytest`, pinned in a separate `requirements-test.txt` (research R-07).

**Storage**: files in the repository — golden JSON files, tolerance file, changelog, synthetic
CSV fixtures.

**Testing**: pytest; the characterization suite is the feature itself.

**Target Platform**: reference environment = GitHub Actions `ubuntu-24.04` runner with Python
3.10 and the pins (research R-03); development = Windows 11 with Python 3.10 via `uv`.

**Project Type**: single project — test suite and maintenance scripts around an existing
computation library.

**Performance Goals**: full suite under 60 s on the development machine (SC-002).

**Constraints**: no network and no cloud credentials during the run (FR-004, SC-004); no
behavior change in `src/` (FR-010); deterministic (FR-005); `src/csv/` never modified;
`requirements.txt` unchanged.

**Scale/Scope**: 13 real series + 8 synthetic series covering the 7 edge-case categories of
FR-007 (research R-06); one output object per series (~a few hundred numeric leaves each).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Assessment | Status |
|---|---|---|
| I. Numerical correctness | The feature exists to enforce it: exact equality in the reference environment, per-field tolerances elsewhere (wording of principle I v1.0.0), deliberate regeneration with recorded reason. | PASS |
| II. Characterize before restructuring | Golden files freeze current behavior, including D-02 (C-02) and the D-13 path, marked as known/suspected defects. No restructuring in this feature. | PASS |
| III. Methodological transparency | Not affected: no change to user-facing output. | N/A |
| IV. AI interprets, never computes | No AI feature. | N/A |
| V. Simplicity and justified boundaries | No new service, storage or network boundary. One new test-only dependency (pytest) in a separate file; CI is GitHub Actions, already the host of the repo. No ADR needed. | PASS |
| VI. Spec before code, approval | Spec approved 2026-10-29 (`255ac0e`). This plan requires its own approval before `/speckit-tasks` → implementation. | PASS (pending plan approval) |
| VII. Official sources and provenance | Golden metadata records environment, date, input file and its SHA-256; real fixtures are the unchanged ANA CSVs. | PASS |
| VIII. Accessible to the non-specialist | Not affected; v0 user-facing strings are frozen as they are, typo included. | N/A |
| Additional constraints | Pinned versions; no unused dependency; library upgrades stay out of scope. English artifacts. | PASS |
| Workflow step 6 | Implemented by FR-011 (workflow + ruleset). | PASS |

**Post-design re-check (after Phase 1)**: unchanged — PASS. The design adds no production
change; the only repository-level additions are test files, scripts, a CI workflow and
`requirements-test.txt`.

## Project Structure

### Documentation (this feature)

```text
specs/001-characterization-tests/
├── plan.md              # This file
├── research.md          # Phase 0: decisions R-01..R-11
├── data-model.md        # Phase 1: golden file, tolerance file, changelog, fixtures
├── quickstart.md        # Phase 1: how to run, regenerate, measure, verify
├── contracts/
│   ├── golden-file.schema.md      # Golden JSON format
│   ├── tolerances.schema.md       # Per-field tolerance file format
│   └── update-golden-cli.md       # Regeneration and measurement script interfaces
├── checklists/requirements.md
└── tasks.md             # Phase 2 (/speckit-tasks — not created here)
```

### Source Code (repository root)

```text
src/                              # production code — NOT modified
└── csv/                          # the 13 real Hidroweb CSVs — read in place, never modified

tests/
├── conftest.py                   # sys.path (src/), google.cloud.storage stub, BLAS thread
│                                 # env vars, network guard, environment fingerprint
├── characterization/
│   ├── test_golden.py            # one parametrized case per series → compare with golden
│   └── compare.py                # normalization + recursive comparison + path-based diff
├── fixtures/
│   └── synthetic/                # 8 Hidroweb-format CSVs (ISO-8859-1, `;`, `,`, 12 header lines)
│       └── README.md             # what each fixture covers; linked backlog item
└── golden/
    ├── real/<station-id>.json    # 13 files
    ├── synthetic/<case-id>.json  # 8 files
    ├── tolerances.json           # per-field tolerances for non-reference environments
    └── CHANGELOG.md              # reason log for every (re)generation

scripts/
├── update_golden.py              # regenerate golden files; requires --reason
├── measure_env_diff.py           # compare current environment vs golden; per-field report
└── make_synthetic_fixtures.py    # deterministic generator of the synthetic CSVs (run once)

.github/workflows/characterization.yml   # jobs: characterization (push (any branch));
                                         # generate-golden (push with [regenerate-golden],
                                         # workflow_dispatch after the merge)
requirements-test.txt             # pytest (pinned); installed with -r requirements.txt
pytest.ini                        # testpaths, markers, no warnings-as-errors
```

**Structure Decision**: single project. Tests live under `tests/`, maintenance scripts under
`scripts/`, both outside `src/`. The real CSVs are
referenced from `src/csv/` in place rather than moved (CLAUDE.md rule; avoids touching
golden-file inputs).

## Implementation Phases (input for `/speckit-tasks`)

1. **Harness** — `requirements-test.txt`, `pytest.ini`, `tests/conftest.py` (path, storage stub,
   thread env vars, socket guard), `compare.py` with unit tests of the comparator itself.
2. **Reference environment and initial golden files** — CI workflow with the
   `generate-golden` job, triggered by `push` on the feature branch when the head commit
   message contains `[regenerate-golden]`; the reason is the commit message body (e.g.
   "initial characterization"). It runs `update_golden.py` on `ubuntu-24.04` and uploads
   `tests/golden/` as an artifact; the maintainer reviews and commits it (manual step M-2,
   research R-04). `workflow_dispatch` cannot be used here: the workflow is not on `main` yet.
3. **Measurement task (FR-002)** — on the Windows development machine (Python 3.10 via `uv`),
   run `scripts/measure_env_diff.py` against the golden files of the 13 CSVs; record the
   per-field maximum absolute and relative differences in `research.md` (table per field group
   and per station); set `tests/golden/tolerances.json` from that measurement (rule in R-05);
   confirm that the tolerances still detect the SC-003 perturbations.
4. **Synthetic fixtures (US3)** — generate the 8 CSVs, generate their golden files in the
   reference environment, mark known/suspected defects (D-02/C-02, D-13).
5. **Deliberate update path (US2)** — `update_golden.py` refusal without reason, changelog
   entry with changed fields, `--station` filter.
6. **Automatic protection (US4)** — `characterization` job on push (any branch) in the
   reference environment, exact mode enforced; `generate-golden` job also reachable by
   `workflow_dispatch` once the workflow is on `main`, for later regenerations
   (`gh workflow run characterization.yml --ref <branch> -f reason="..."`); **manual step
   M-1**: repository ruleset on `main` requiring the `characterization` status check
   (research R-09).
7. **Verification** — SC-002 timing, SC-003 one-time sensitivity test (recorded), SC-004
   network guard, determinism (two consecutive runs).

## Manual Steps (maintainer)

- **M-1 — Repository ruleset (FR-011)**: GitHub → Settings → Rules → Rulesets → New branch
  ruleset, target `main`, enforcement *Active*, rule *Require status checks to pass* with the
  check `characterization` (the job name of the workflow). It can only be selected after the
  workflow has run at least once. Not automated: changing repository settings is outside the
  code and needs explicit approval.
- **M-2 — Commit generated golden files**: download the `golden-<run id>` artifact of the
  `generate-golden` job (`gh run download <run-id> -n golden-<run-id>`), review the diff and the
  new `CHANGELOG.md` entry, and commit. Initial generation (phase 2): push a commit whose
  message contains `[regenerate-golden]` and whose body is the reason. Later regenerations
  (after the merge): `gh workflow run characterization.yml --ref <branch> -f reason="<text>"`.

## Risks

- **Bit-exactness across CI runners**: OpenBLAS selects kernels by CPU; GitHub runners vary in
  CPU model. Mitigation in R-03 (single-thread BLAS, CPU kernels pinned from the start with
  `OPENBLAS_CORETYPE` and `NPY_DISABLE_CPU_FEATURES`, repeated runs before the ruleset is enabled
  to verify stability).
- **Production parity of the reference environment**: the Cloud Function also runs on Linux
  x86-64 with Python 3.10 and the same pins, but its exact CPU is unknown. The golden files
  characterize the code with the pinned libraries, not a specific machine; the production
  runtime is to be retired by spec 002 anyway.
- **Synthetic fixtures may not reach the fit** (e.g., degenerate series): each fixture's
  expected path is documented, and the golden file freezes whatever the code does today,
  including exceptions (R-08).

## Complexity Tracking

No constitution violations to justify.
