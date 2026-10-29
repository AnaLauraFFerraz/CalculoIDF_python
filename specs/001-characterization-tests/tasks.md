---

description: "Task list for 001 — Characterization tests (golden files)"
---

# Tasks: Characterization tests for the IDF computation (golden files)

**Input**: Design documents from `specs/001-characterization-tests/`

**Prerequisites**: plan.md (approved 2026-10-29), spec.md (approved 2026-10-29), research.md,
data-model.md, contracts/, quickstart.md

**Tests**: This feature *is* the test suite. Separate unit-test tasks cover only the code that
guards the suite (the comparator and the regeneration script); they are listed explicitly.

**Organization**: Tasks are grouped by user story. Production code in `src/` and the CSVs in
`src/csv/` are never modified (FR-010); `requirements.txt` is unchanged.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1–US4 (spec.md)
- **[MANUAL]**: needs the maintainer (push, download, repository settings). Stop and ask; never
  push without explicit approval (CLAUDE.md).

## Path Conventions

Single project. Tests in `tests/`, maintenance scripts in `scripts/`, workflow in
`.github/workflows/`. Commands run from the repository root with `.venv\Scripts\python`.

---

## Phase 1: Setup

**Purpose**: Test tooling and local environment (research R-07).

- [X] T001 Create `requirements-test.txt` containing only `pytest==8.3.5`.
- [X] T002 [P] Create `pytest.ini` with `testpaths = tests`, a `golden` marker, and no
  `filterwarnings = error` (library warnings must not break the comparison, research R-08).
- [X] T003 Reuse the existing `.venv` (Python 3.10; do not recreate it) and install only the
  test requirements: `uv pip install -r requirements-test.txt`. Confirm `git status` shows
  `requirements.txt` untouched.

---

## Phase 2: Foundational (blocking prerequisites)

**Purpose**: Harness, comparator, generation script and CI workflow. All user stories need them.

**⚠️ CRITICAL**: No user story starts before this phase is complete.

- [X] T004 Create `tests/conftest.py`: (a) set `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `OPENBLAS_CORETYPE=Haswell` and `NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL"`
  in `os.environ` **before** numpy is imported (R-03; implemented in
  `tests/characterization/bootstrap.py`, which `conftest.py` and the scripts call; a numpy
  `RuntimeWarning` on CPUs without AVX-512 is harmless); (b) put `src/` on `sys.path`;
  (c) register a stub `google.cloud.storage` in `sys.modules` (and as the `storage` attribute of
  `google.cloud`) whose `Client()` returns an object whose methods raise, before `main` is
  imported (R-01); (d) an autouse fixture replacing `socket.socket.connect` and
  `socket.create_connection` with functions that raise (R-10, FR-004).
- [X] T005 Create `tests/characterization/compare.py` with:
  - `fingerprint()` (os, machine, Python `major.minor.micro`, numpy, pandas, scipy,
    scikit-learn, and the effective values of `OPENBLAS_CORETYPE` and
    `NPY_DISABLE_CPU_FEATURES` — contract golden-file.schema.md) and `fingerprints_match()`
    (compares os, machine, Python `major.minor`, the four library versions and the two kernel
    variables);
  - `REFERENCE_FINGERPRINT`, a module constant: `os: "Linux"`, `machine: "x86_64"`,
    `python: "3.10"`, the versions of `numpy`, `pandas`, `scipy`, `scikit-learn` **read from
    the pins in `requirements.txt`** at import time, and the two kernel variables with the
    values of R-03 (research R-03, R-11);
  - `normalize()` (numpy scalars → `.item()`, arrays → lists, JSON round-trip with
    `sort_keys=True`; `NaN`/`Infinity` are kept — contract golden-file.schema.md);
  - `run_series(csv_path)` returning `(result_kind, result)` for `object`/`string`/`exception`
    (R-02, R-08: a JSON-string return is decoded; an exception is `{type, message}`);
  - the series registry: 13 real series from `src/csv/*.csv` (read in place) and synthetic series
    **discovered by glob `tests/fixtures/synthetic/S*.csv`**, each with `id`, `path`, `sha256`
    and `marker`; markers live in a dict keyed by series id and are applied only to series whose
    file exists, so the suite works before the synthetic CSVs exist;
  - `load_golden()`;
  - `compare(expected, actual, mode, tolerances)` recursive per R-05: exact key sets, list length
    and order, **types** (`bool` ≠ number, `false` ≠ list, strings stay strings), `NaN == NaN`,
    `==` in `exact`, `math.isclose` with the first matching rule in `tolerant`; returns diffs as
    `(path, expected, actual, rule)`;
  - mode selection via `CALCULOIDF_COMPARE_MODE` (`exact`|`tolerant`) else fingerprint match; in
    `tolerant` mode refuse to run with a clear error while any rule in `tolerances.json` has a
    `null` tolerance (contract tolerances.schema.md). Path syntax: dot-separated keys, `[i]` for
    list indices; patterns `*`, `[*]`, `**`.
- [X] T006 [P] Create `tests/unit/test_compare.py` testing the comparator itself: identical
  structures pass; one differing leaf is reported with the right path; type mismatches
  (`false` vs list, `1` vs `True`, `"2001"` vs `2001`) fail even in tolerant mode; list
  length/order; extra/missing key; `NaN` equality; pattern matching and first-match-wins; exact
  mode rejects a 1-ulp difference that tolerant mode accepts; the `null`-tolerance refusal;
  `REFERENCE_FINGERPRINT` matches the pins in `requirements.txt`.
- [X] T007 Create `scripts/update_golden.py` per `contracts/update-golden-cli.md`: argparse with
  required `--reason` (empty after trimming → exit 2, nothing written), optional `--series`,
  `--allow-non-reference` with `--out`; when the current fingerprint does not match
  `compare.REFERENCE_FINGERPRINT` and the flag is absent → exit 3. Writes
  `tests/golden/{real|synthetic}/<id>.json` exactly per `contracts/golden-file.schema.md`
  (UTF-8, LF, sorted keys, indent 2, trailing newline; metadata with `series_id`, `input_path`,
  `input_sha256`, `generated_at`, `environment`, `marker`). Expose the logic as a function
  (e.g. `run(argv, golden_root=..., reference_fingerprint=...)`) whose golden root and reference
  fingerprint are **internal parameters for tests, not CLI flags**; the CLI always uses
  `tests/golden/` and `compare.REFERENCE_FINGERPRINT`. Reuses `tests/characterization/compare.py`
  (add `tests/` to `sys.path`) and the same conftest-style storage stub and thread variables so it
  runs without credentials. Prints the series written; the field-path diff is T020 and the
  changelog is T022.
- [X] T008 [P] Create `tests/golden/tolerances.json` with the shape of
  `contracts/tolerances.schema.md`: the seven rules, all tolerances `null` except the default
  `**` rule (`rel_tol: 1e-12`, `abs_tol: 0.0`), `safety_factor: 10`.
- [X] T009 Create `.github/workflows/characterization.yml` (R-03, R-04, R-09): triggers `push`
  (all branches) and `workflow_dispatch` (inputs `reason` required, `series` optional); no
  `pull_request` trigger. Both jobs on `ubuntu-24.04`, `actions/checkout@v7`,
  `actions/setup-python@v7` with Python 3.10, `pip install -r requirements.txt -r
  requirements-test.txt`, workflow-level env `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell` and `NPY_DISABLE_CPU_FEATURES` with the AVX-512
  list of R-03.
  - Job `characterization` (name exactly `characterization`): runs on `push` to any branch only
    (`if: github.event_name == 'push'`); not on `pull_request` and not on `workflow_dispatch`,
    so there are no duplicate runs. A PR's required check is satisfied by this push-triggered
    check on the branch head commit, because GitHub matches required checks by name on the head
    SHA (R-09). Runs `CALCULOIDF_COMPARE_MODE=exact python -m pytest`.
  - Job `generate-golden`: runs on `push` to any branch whose
    `github.event.head_commit.message` contains `[regenerate-golden]`, and on
    `workflow_dispatch`. Reason = commit-message body with the marker removed and trimmed (push)
    or the `reason` input (dispatch), passed through an **environment variable** (never
    `${{ }}` inside the shell command); runs
    `python scripts/update_golden.py --reason "$REASON" [--series ...]`; uploads `tests/golden/`
    with `actions/upload-artifact@v7` as `golden-${{ github.run_id }}`; never commits.
- [X] T010 Run `.venv\Scripts\python -m pytest tests/unit` and fix until green. Run
  `git diff main -- src/ requirements.txt` and confirm it is empty.

**Checkpoint**: harness, comparator, generator and workflow exist; no golden files yet.

---

## Phase 3: User Story 1 — Detect any change in the results (P1) 🎯 MVP

**Goal**: the suite processes each of the 13 real series and compares the full output with a
frozen reference; exact in the reference environment, tolerant elsewhere.

**Independent Test**: on a clean checkout the suite passes; changing a disaggregation
coefficient by 0.001 makes it fail with series id, field path, expected and actual.

- [X] T011 [US1] Create `tests/characterization/test_golden.py`: one parametrized case per
  registered series (ids = series ids, marker `golden`); asserts the input SHA-256 equals
  `metadata.input_sha256` (a mismatch fails: input changed); runs `run_series`; compares
  `result_kind` and `result` with the selected mode; on failure the assertion message lists, per
  diff, **series id, field path, expected value, actual value, rule** (FR-003, US1 scenario 2);
  business-message and exception results are compared too (US1 scenario 3). A missing golden
  file fails with a message pointing to `scripts/update_golden.py`.
- [X] T012 [US1] Add the `marker` definitions for real series in `compare.py`'s registry.
  C-01 (D-01) affects every fitted result, so there is **no per-station marker** for it; state
  that in a comment at the registry and in `tests/fixtures/synthetic/README.md` (T027). C-03
  (D-07): only `chuvas_C_02346117` has outliers removed today (4 years — 1969, 1970, 1972,
  1973 — all through the lower bound); its marker is `{status: "suspected", debt: "D-07", fix:
  "C-03", note: "4 low outliers removed without recomputing the statistics"}`. Every other
  real series: marker `null`.
- [ ] T013 [MANUAL] [US1] Maintainer approves the push of the branch and opens a **draft PR** to
  `main`. The PR is the review surface: it shows the `characterization` check that every push
  to the branch triggers on its head commit (R-09), and the stability re-runs of T020 happen
  from it.
- [ ] T014 [MANUAL] [US1] Maintainer approves and pushes a final **empty** commit
  (`git commit --allow-empty`) whose subject contains `[regenerate-golden]` and whose body is the
  reason `initial characterization` (plan M-2). Expected: `generate-golden` runs and uploads
  `golden-<run id>`; the `characterization` job triggered by the same push fails (no golden
  files exist yet) — expected, not a bug (R-04).
- [ ] T015 [MANUAL] [US1] Maintainer downloads the artifact
  (`gh run download <run-id> -n golden-<run-id>`). Claude reviews the 13
  `tests/golden/real/*.json` files and the changelog entry (fingerprint is Linux / x86_64 /
  Python 3.10 / pinned versions; the two insufficient-data series have `result_kind:
  "string"`), then commits them as `test: add golden files for the 13 real series`.
- [ ] T016 [US1] Create `scripts/measure_env_diff.py` per the contract: runs `main.main` for
  selected series in the current environment, compares with the golden files **without
  tolerances**, reports per tolerance-rule group and per series the max absolute and max
  relative difference, number of differing leaves, and any non-numeric difference (reported as a
  failure, not a tolerance matter); `--series`, `--out report.md`; writes nothing under
  `tests/golden/`.
- [ ] T017 [US1] Run the measurement on the Windows machine (`.venv`, Python 3.10, same pins):
  `.venv\Scripts\python scripts/measure_env_diff.py --out $env:TEMP\measure.md`. Paste the
  table (field group × series: max abs Δ, max rel Δ) into `research.md` under "Measurement
  results", replacing the pending row, with date and both fingerprints.
- [ ] T018 [US1] Fill `tests/golden/tolerances.json` from T017 by the rule in R-05: per group,
  measured maximum × 10 rounded **up** to one significant digit, floor `rel 1e-12`; if the
  measured difference is exactly zero use the floor. Record `measured_at`,
  `reference_environment`, `measured_environment`, `safety_factor` and each rule's
  `measured_max_abs` / `measured_max_rel`. If a group has a non-numeric difference, stop and
  report it.
- [ ] T019 [US1] Run `.venv\Scripts\python -m pytest` on Windows (tolerant mode): all 13 real
  series pass, total under 60 s (SC-002). Run it twice and confirm identical outcome (FR-005).
  Commit tolerances and research table as `test: set per-field tolerances from measurement`.
- [ ] T020 [MANUAL] [US1] Verify stability before relying on exact mode (R-03): the
  `characterization` job must pass in exact mode on at least 3 runs; re-runs on new runners
  count. The runs are re-runs of the push-triggered job on the branch head, started from the
  draft PR of T013. The CPU kernels are already pinned (`OPENBLAS_CORETYPE=Haswell`,
  `NPY_DISABLE_CPU_FEATURES`) from the initial generation, so this task verifies stability; it
  no longer reacts to instability. If a run still differs, stop and report it; do not change
  the golden files or the tolerances. Record the outcome (runs, runner CPU models if visible)
  in `research.md`.

**Checkpoint**: US1 is functional: the 13 real series are frozen and checked in both modes.

---

## Phase 4: User Story 2 — Update references deliberately (P2)

**Goal**: references are regenerated only with a recorded reason, producing a changelog entry
that lists the changed fields and the size of each change.

**Independent Test**: regenerating one series with a reason adds a changelog entry (date,
series id, changed fields, reason); without a reason the script refuses and writes nothing.

- [X] T021 [US2] Extend `scripts/update_golden.py`: for each selected series compare the new
  content with the existing golden file. A file **is rewritten only when `result_kind`,
  `result`, `metadata.input_sha256` or `metadata.marker` change**; a difference in
  `generated_at` (or in `environment` alone) does not count, and an unchanged file is left
  byte-for-byte as it is. For each changed series compute the changed field paths (or "initial
  generation" when the file is absent), the **number of changed leaves** and the **max absolute
  and max relative delta** over numeric leaves (constitution principle I). Unknown `--series`
  ID → exit 2 with the list of valid IDs. If no golden changed, print that and add no changelog
  entry (contract).
- [X] T022 [US2] Implement `--allow-non-reference` with `--out <dir>`: writes only to the given
  scratch directory, never to `tests/golden/`, and adds no changelog entry.
- [X] T023 [US2] Implement the changelog in `scripts/update_golden.py`: append-only
  `tests/golden/CHANGELOG.md`, one entry per run with UTC date, series IDs, reason, environment
  fingerprint, trigger (workflow run URL from `GITHUB_SERVER_URL`/`GITHUB_REPOSITORY`/
  `GITHUB_RUN_ID` when present, else "local (reference environment)"), and a **Markdown table
  per run with one row per changed series: series id, changed leaves, max abs Δ, max rel Δ**,
  followed by the changed field paths (computed, not typed). The script prints the same table to
  stdout so it can be pasted into the PR description. Create the file with a short header if
  absent.
- [X] T024 [P] [US2] Create `tests/unit/test_update_golden.py`, calling the script's function with
  a golden root under `tmp_path` and the reference fingerprint injected through `monkeypatch`
  (never touching `tests/golden/`): missing or whitespace-only `--reason` → exit 2 and no file
  written; unknown series → exit 2; non-reference fingerprint without the flag → exit 3 and
  nothing written; with `--allow-non-reference --out <tmp>` nothing changes under the golden
  root and no changelog entry; an unchanged result leaves the file byte-identical and adds no
  entry; a changed result rewrites the file and appends an entry with the table (changed leaves,
  max abs/rel Δ), the changed paths and the reason.
- [ ] T025 [MANUAL] [US2] After the merge only: demonstrate a later regeneration with
  `gh workflow run characterization.yml --ref <branch> -f reason="..." -f series="<id>"` and
  check the changelog entry (plan § Implementation Phases, item 6). Before the merge, T024 is the
  proof.

**Checkpoint**: US2 done; US1 unaffected.

---

## Phase 5: User Story 3 — Document current behavior in edge cases (P2)

**Goal**: eight synthetic series freeze the current behavior in the seven FR-007 categories,
with defects marked.

**Independent Test**: each synthetic series has a golden file; the cases for D-02 and D-13
carry a marker pointing to the backlog item.

- [ ] T026 [US3] Create `scripts/make_synthetic_fixtures.py` (`--out tests/fixtures/synthetic`,
  fixed seed, deterministic): writes `S1`–`S8` in Hidroweb format (ISO-8859-1, `;` separator, `,`
  decimal mark, 12 header lines, columns including `NivelConsistencia`, `Data`, `Maxima`) so that
  `main.load_data` reads them. Cases (R-06): S1 < 10 complete water years; S2 10–29 years; S3 no
  consisted data (level 2 absent); S4 empty water year in the middle; S5 starts in October; S6
  ends in September; S7 Oct/1990–Sep/2000 (10 complete years, mandatory); S8 a month **inside
  the consisted period with consisted `Maxima` empty or 0 and earlier than the first raw
  record**, so `Maxima` is NaN when `fillna(0, inplace=True)` runs (`process_data.py:67`), with
  ≥ 10 complete water years reachable afterwards. Read `src/main.py` and `src/process_data.py`
  first to build inputs the code actually accepts.
- [ ] T027 [US3] Run the generator, commit the CSVs (check `git check-attr text` reports
  `unset` for `*.csv`, so bytes are preserved), and create
  `tests/fixtures/synthetic/README.md`: what each fixture covers and its backlog item, plus a
  note that C-01 (D-01) affects every fitted result and therefore has no per-station marker.
- [ ] T028 [US3] Add the synthetic markers to the marker dict in `compare.py`: S5, S6, S7 →
  `{status: "known-defect", debt: "D-02", fix: "C-02"}` with a one-line `note` each; S8 →
  `{status: "suspected", debt: "D-13", fix: null, note: "pandas 3 Copy-on-Write turns
  fillna(inplace=True) into a no-op; handled in the phase-2 library upgrade"}` (FR-008).
- [ ] T029 [MANUAL] [US3] Push a `[regenerate-golden]` commit (body: `edge cases: synthetic
  series S1-S8`). The push trigger has no `--series`, so the job generates **every** series;
  thanks to T021 the 13 real golden files stay byte-identical and only S1–S8 are written.
  Download the artifact.
- [ ] T030 [US3] Review the synthetic golden files against the R-06 hypotheses, **report any
  difference to the maintainer instead of changing the code or the fixture to match**. Confirm
  the real golden files are unchanged in the artifact. Mandatory: `S7.json` has `result_kind:
  "string"` (insufficient data, 8 years). Then commit
  `test: add golden files for synthetic edge cases`.
- [ ] T031 [US3] Run the full suite on Windows (21 series, tolerant mode), under 60 s. If a
  synthetic series moves a tolerance group beyond T018, re-run T017–T018 including them and
  record it.

**Checkpoint**: US3 done; SC-001 (13 real + 7 edge-case categories) covered.

---

## Phase 6: User Story 4 — Automatic protection on every change (P3)

**Goal**: the suite runs on every push and blocks integration on failure.

**Independent Test**: a PR with a numeric change fails the `characterization` check and cannot
be merged.

- [ ] T032 [US4] Confirm in `.github/workflows/characterization.yml` that `characterization`
  sets `CALCULOIDF_COMPARE_MODE=exact` explicitly (a fingerprint mismatch in CI must fail, not
  relax), that its job name is exactly `characterization`, and that it runs on `push` to any
  branch only.
- [ ] T033 [MANUAL] [US4] Maintainer marks the draft PR of T013 ready for review and confirms the
  check passes with all 21 series in exact mode (and ≥ 3 stable runs, T020).
- [ ] T034 [MANUAL] [US4] **M-1**: maintainer creates the repository ruleset on `main`
  (Settings → Rules → Rulesets, enforcement Active, *Require status checks to pass* with
  `characterization`, and *Require branches to be up to date before merging*, so the tested head
  commit equals the merge result). Claude does not change repository settings.
- [ ] T035 [MANUAL] [US4] Demonstrate blocking: push a throwaway branch with a 0.001 change in a
  disaggregation coefficient, open a PR, confirm the check fails and the merge is blocked; close
  the PR and delete the branch (maintainer's decision).

**Checkpoint**: all four stories complete.

---

## Phase 7: Polish & verification

- [ ] T036 SC-003 sensitivity test, once, manually, in **tolerant mode only** (justification in
  research R-05: exact mode compares with `==`, so any bit change fails by construction, as the
  1-ulp unit test of T006 proves). Perturb by 0.001 (a) a disaggregation coefficient in
  `src/disaggregation_coef.py`, (b) the BFGS initial guess in `src/ventechow.py`, (c) a value in
  `src/yn_sigman.py` or `src/grubbs_test.py`; confirm the suite fails each time with series id,
  path, expected and actual; **revert each perturbation** and run
  `git diff --exit-code -- src/` (must exit 0) **before starting the next one**. Perturbations
  are never committed. If a tolerance hides a perturbation, report it — do not widen it silently
  (R-05). Record the three outcomes in `research.md`.
- [ ] T037 [P] SC-004 / FR-004: run the suite with the network disabled and no GCP credentials
  (`GOOGLE_APPLICATION_CREDENTIALS` unset); confirm zero network calls (the T004 guard would
  raise). SC-002: record the full-suite time.
- [ ] T038 [P] FR-005: two consecutive runs on Windows and the CI exact-mode runs give the same
  outcome. FR-010: `git diff main -- src/ requirements.txt` is empty.
- [ ] T039 [P] Update `CLAUDE.md` "Commands" (test commands that now exist, `requirements-test.txt`)
  and the "No … tests … no CI yet" sentences; update `docs/knowledge` only where it states the
  absence of tests/CI. Describe only what exists.
- [ ] T040 Run the quickstart scenarios 1–7 in `quickstart.md` and note any deviation in
  `research.md`.
- [ ] T041 After implementation, run `/speckit-converge` to catch unbuilt work.

---

## Dependencies & Execution Order

- **Phase 1 → Phase 2 → Phase 3** strictly. T004 → T005 → (T006, T007) → T010; T008, T009 are
  independent of T005–T007 and can run in parallel with them.
- **US1**: T011 → T012 → T013 [MANUAL] → T014 [MANUAL] → T015 [MANUAL] → T016 → T017 → T018 →
  T019; T020 runs over several re-runs in parallel with US2/US3.
- **US2** depends on Phase 2 (T007). **Do T021–T023 before T014**, so the first CI generation
  already writes the changelog entry (the changelog is never hand-written) while US2 stays
  independently testable (T024).
- **US3** needs T005 (registry), T021 (unchanged real files are not rewritten, for T029) and the
  US1 tolerances (T018) for T031.
- **US4** needs US1 golden files, the workflow (T009) and the draft PR (T013).
- **Polish** after all stories.

### Parallel opportunities

- T002 with T001; T006 with T007; T008 with T005–T007; T009 after T007 contracts are settled.
- T024 with T026 (different files).
- T037, T038, T039 together.

```text
# Phase 2 in parallel after T004:
T005 compare.py   |  T008 tolerances.json  |  T009 characterization.yml
# then: T006 test_compare.py  |  T007 update_golden.py
```

## Implementation Strategy

### MVP (User Story 1)

1. Phases 1 and 2, then T021–T023 (change detection and changelog) so the first CI generation
   records its reason.
2. T011–T015: draft PR and first golden files from CI (three [MANUAL] gates).
3. T016–T019: measure and set tolerances. **Stop and validate**: suite green on Windows
   (tolerant) and in CI (exact).

### Incremental delivery

US1 → US2 tests (T024) → US3 edge cases → US4 ruleset → Polish. Commit after each task or
logical group, Conventional Commits in English, no `Co-Authored-By`.

## Notes

- Manual gates: T013/T014/T015/T020/T025/T029/T033–T035 and M-1/M-2. Pause there and ask.
- Never edit a golden file by hand; never "fix" a synthetic result to match a hypothesis.
- Any behavior surprise in `src/` is reported, not fixed (out of scope; FR-010).
