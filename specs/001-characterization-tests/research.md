# Research: Characterization tests (001)

Decisions taken during planning (R-01 to R-11). Sources: spec 001, constitution v1.0.0,
`docs/knowledge/07-technical-notes-by-feature.md` (section 001), `docs/knowledge/02-current-system.md`,
and the code in `src/` as of `255ac0e`.

## R-01 — Import of `main` without GCP credentials

- **Decision**: `tests/conftest.py` registers a stub module for `google.cloud.storage` in
  `sys.modules` (and as the `storage` attribute of `google.cloud`) **before** `main` is
  imported. The stub's `Client()` returns an object whose methods raise if called, so any
  accidental use of Cloud Storage fails loudly. `src/` is put on `sys.path`, because the modules
  import each other by bare name (`from gcs_utils import ...`).
- **Rationale**: FR-010 — no change to production behavior. `gcs_utils.py` calls
  `storage.Client()` at import time; the stub removes the credential lookup without editing it.
- **Alternatives considered**: lazy client creation inside `gcs_utils` functions (minimal and
  safe, but a production change that FR-010 asks to avoid; left for spec 002, which removes GCS
  entirely); setting fake credentials via environment variables (still attempts network/metadata
  lookups — violates FR-004).

## R-02 — Entry point and what is compared

- **Decision**: the unit under test is `main.main(csv_path)`. Its return value is recorded as
  is: a `dict` on success, a **JSON string** on business errors (D-04: `"Erro ao carregar o
  arquivo"`, `"Dados não são sufientes para completar a análise"`). The golden file stores
  `result_kind` (`"object"` / `"string"` / `"exception"`) and the decoded result.
  `process_request` (HTTP handler) is out of scope: it only adds GCS download/deletion.
- **Rationale**: spec FR-001 ("full output"), US1 scenario 3 (business messages compared too).
  Two real CSVs (`…01848001_UB.csv`, `…02043032_MA.csv`) end in the insufficient-data string.
- **Alternatives considered**: comparing only selected fields (rejected: FR-001 requires the full
  output).

## R-03 — Reference environment and determinism

- **Decision**: reference environment = GitHub Actions `ubuntu-24.04` (pinned label, not
  `ubuntu-latest`), `actions/setup-python` with Python 3.10, `pip install -r requirements.txt
  -r requirements-test.txt`. Environment variables `OMP_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` **and the CPU-kernel variables**
  `OPENBLAS_CORETYPE=Haswell` and
  `NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL"` are
  always set, from the initial generation on, in the workflow (workflow-level `env`) **and**
  locally (`tests/characterization/bootstrap.py`, called by `conftest.py` and the maintenance
  scripts) before numpy is imported. The environment fingerprint
  (`platform.system()`, machine, Python `major.minor`, versions of numpy, pandas, scipy,
  scikit-learn, and the effective values of the two kernel variables) is
  stored in each golden file's metadata; a run without the pinned kernels never matches the
  reference fingerprint. The reference fingerprint itself is a constant in
  `tests/characterization/compare.py` (`REFERENCE_FINGERPRINT`): `Linux`, `x86_64`, Python
  `3.10`, and the numpy, pandas, scipy and scikit-learn versions read from the pins in
  `requirements.txt` and the two kernel variables, so it exists before any golden file does. Before the
  ruleset is enabled, the CI job is run several times (≥ 3 runs; re-runs on new runners count) to
  **verify** bit-exact stability (task T020); the kernels are not a reaction to instability, they
  are pinned from the start (evidence below).
- **Rationale**: FR-005, FR-009, clarification Q1. BFGS with a fixed initial guess is
  deterministic in a fixed environment; multithreaded BLAS and CPU-specific kernels are the
  known sources of last-bit variation.
- **Alternatives considered**: a Docker image (`python:3.10-slim`) as the reference (more
  control, but adds an image to maintain; can be adopted later if runner variation appears);
  `ubuntu-latest` (drifts silently).
- **Evidence for pinning the CPU kernels** (maintainer's review, measured 2026-10-27 and 2026-10-29; same pins,
  Python 3.10, Linux). OpenBLAS and numpy dispatch kernels by CPU (AVX-512 vs AVX2); the
  ill-conditioned fit (D-01) amplifies the last-bit differences into the fitted parameters.

  | Date | Setup | Host | Result |
  |---|---|---|---|
  | 2026-10-27 | Thread variables = 1 only | cloud host | `chuvas_C_01944009_PL` k2 = 855.25 |
  | 2026-10-29 | Thread variables = 1 only | cloud host | `chuvas_C_01944009_PL` k2 = 894.73 |
  | 2026-10-29 | Thread variables = 1 only | Intel i7-9750H | `chuvas_C_01944009_PL` k2 = 906.92 |
  | 2026-10-29 | Thread variables = 1, `OPENBLAS_CORETYPE=Haswell`, `NPY_DISABLE_CPU_FEATURES` as above | AVX-512 Xeon and Intel i7-9750H | all 13 outputs byte-identical (same SHA-256) |

  On CPUs without AVX-512, numpy prints a harmless `RuntimeWarning` that those features cannot be
  disabled; `pytest.ini` does not turn warnings into errors (R-08), so it does not fail the suite.
  The thread variables alone changed nothing. **Decision**: both kernel variables are always set,
  in CI and locally, from the initial generation on.

## R-04 — How golden files are generated in Linux from a Windows machine

- **Constraint**: GitHub accepts `workflow_dispatch` only for workflows present on the default
  branch. `characterization.yml` exists only on `001-characterization-tests` until the merge,
  so the initial golden files cannot be generated by a manual dispatch.
- **Decision**: two triggers for the same generation step.
  - **Before the merge (initial generation, feature branch)**: a `generate-golden` job
    triggered by `push`, guarded by
    `if: contains(github.event.head_commit.message, '[regenerate-golden]')`. The reason is the
    **body** of the head commit message (everything after the subject line, with the
    `[regenerate-golden]` marker removed and whitespace trimmed). The message reaches the
    script through an environment variable, never by interpolating `${{ }}` into the shell
    command (avoids script injection). The job runs `scripts/update_golden.py --reason
    "$REASON"` in the reference environment and uploads `tests/golden/` as an artifact
    (`golden-<run id>`). An empty body → the script refuses (exit 2) and the job fails. Only the
    head commit of a push is inspected; the marker must be in the last commit pushed.
  - **After the merge (later regenerations)**: `workflow_dispatch` with inputs `reason`
    (required) and `series` (optional), run against any branch that contains the workflow:
    `gh workflow run characterization.yml --ref <branch> -f reason="<text>" [-f series="S1 S7"]`,
    then `gh run download <run-id> -n golden-<run-id>`.
  - In both cases the maintainer downloads the artifact, reviews the diff and commits it
    (manual step M-2). The CI never commits or pushes.
- **Rationale**: the maintainer works on Windows; the golden files must be generated on Linux
  (clarification Q1). The push-triggered job works on the feature branch before the workflow
  reaches `main`; keeping the commit manual preserves review and the "no push without approval"
  rule.
- **Side effect**: on the push that carries `[regenerate-golden]` before any golden file exists,
  the `characterization` job also runs and fails (no golden files exist yet). This is expected
  on the feature branch and disappears once the generated files are committed.
- **Alternatives considered**: `workflow_dispatch` only (not executable before the merge);
  WSL or Docker on the development machine (viable fallback; the fingerprint check accepts it if
  it matches); CI committing directly (rejected: bypasses review).

## R-05 — Comparison rules and tolerances

- **Decision**:
  - Normalize the result before serializing: numpy scalars → Python scalars (`.item()`),
    numpy arrays → lists; then JSON round-trip with `sort_keys=True`. Floats keep full
    precision (`repr` round-trips exactly).
  - Recursive comparison: dict keys must match exactly; list length and **order** must match;
    **types must match** (`bool` ≠ number, `false` ≠ list in `empty_years`, strings stay
    strings in `year_range`); strings compared exactly; `NaN` equals `NaN`.
  - **Mode**: `exact` when the current environment fingerprint equals the golden metadata
    fingerprint (the CI case) → numbers compared with `==`. Otherwise `tolerant` → numbers
    compared with `math.isclose(rel_tol, abs_tol)` taken from `tests/golden/tolerances.json`
    by field-path pattern. An environment variable `CALCULOIDF_COMPARE_MODE=exact|tolerant`
    overrides the automatic choice (CI sets `exact` explicitly, so a fingerprint mismatch in
    CI fails instead of silently relaxing).
  - Field groups in the tolerance file and their rule kind (amended 2026-10-29, ADR 0003).
    **Post-fit** groups — `parameters.**` (k, m, c, n — least stable, D-01),
    `intensity_graph_data_*.i_calculated[*]`, `intensity_graph_data_*.regression.*`,
    `mean_relative_errors.*`, `ns.*` — use the `structure` rule: types, keys, list lengths and
    finiteness are compared, finite values are not, with D-01 as the recorded reason. **Pre-fit** groups —
    `graph_data.**` and the default `**` (which includes `i_real`) — use numeric tolerances. In
    `exact` mode the rule kinds are ignored and every number is compared with `==`.
  - **Setting the values (measurement task, FR-002)**: run `scripts/measure_env_diff.py` on the
    Windows machine (Python 3.10 via `uv`, same pins) against the reference golden files of the
    13 CSVs. For each field group, record max |Δ| and max |Δ|/|expected| per station in a
    table appended to this file ("Measurement results", to be filled during implementation).
    Numeric tolerance per **pre-fit** group = measured maximum × 10, rounded up to one
    significant digit, with a floor of `rel 1e-12`; if the measured difference of a group is
    exactly zero, use the floor. The measurement (T017, 2026-10-29, "Measurement results" below)
    showed that the same rule applied to the post-fit groups would give relative tolerances of
    about 2 (`parameters.**`), 0.2 (`i_calculated`), 40 (`regression.*`), 0.3
    (`mean_relative_errors.*`) and 7e-3 (`ns.*`), which cannot detect the SC-003 perturbations;
    those groups use the `structure` rule instead (ADR 0003). The tolerance file records the
    measurement date, both fingerprints, the safety factor and, for every rule, the measured
    maxima.
  - **Check after setting (amended 2026-10-29)**: each SC-003 perturbation (0.001 in a
    disaggregation coefficient, in the BFGS initial guess, in a statistical constant) is verified
    in the mode that can see it, according to the field groups it reaches (read from `src/`). If
    a numeric tolerance would hide one that reaches a pre-fit group, it is too loose — report it,
    do not silently widen. Exact mode compares numbers with `==`, so any bit change fails by
    construction (the 1-ulp unit test of the comparator, `tests/unit/test_compare.py`, proves
    it); a perturbation that reaches post-fit groups only cannot fail in tolerant mode, so it is
    verified in exact mode in a reference-equivalent Linux environment (fingerprint match: Linux
    x86-64, Python 3.10.x, the pins and the pinned kernels, e.g. a VM or WSL), without
    committing the perturbation.
    - (a) a disaggregation coefficient (`disaggregation_coef.py:27-42`) → `ventechow.py:25,30` →
      `i_real` (`ventechow.py:41`, group `**`, pre-fit) and, because `i_real` is the fit target
      (`ventechow.py:116,143`), every post-fit group. `graph_data` is not reached
      (`ventechow.py:236-240` is built from the distribution data only). Verified in **tolerant**
      mode.
    - (b) the BFGS initial guess (`ventechow.py:8`, used at `:102`; the call at `:204` is
      overwritten at `:114` and `:122`) → `parameters`, `i_calculated`, `regression`,
      `mean_relative_errors`, `ns` only; `i_real` and `graph_data` are computed before the fit.
      Verified in **exact** mode, reference-equivalent environment.
    - (c) a statistical constant. Values in `yn_sigman.py` are not reachable: they enter
      `params["yn"]` and `params["sigman"]` (`distributions.py:37-38`) and are used only by the
      finite Gumbel (`distributions.py:112-113`, `k_coefficient.py:57-59`); Pearson's r
      (`distributions.py:115`) is invariant to that affine map and equals the theoretical Gumbel's
      r² (`distributions.py:98-107`), the tie at `distributions.py:143` resolves to
      `gumbel_theoretical` (first key, `:139-140`), and no golden file has `"dist": "Gumbel
      Finita"`. Values in `grubbs_test.py` are threshold-gated (`outlier_test.py:40,72,82`): a
      0.001 change matters only if a sample's statistic lies within 0.001 of the critical value
      of its sample size, which reading cannot decide. T036 (c) therefore perturbs a constant
      that always propagates for the chosen distribution: the theoretical Gumbel coefficient
      `0.7797` (`distributions.py:99`, `k_coefficient.py:50`), on a series whose chosen
      distribution is `Gumbel Teórica` (`chuvas_C_01844000_CV`, `chuvas_C_01844000_CV1`,
      `chuvas_C_01944009_PL`). It reaches `graph_data.P_dist` (`ventechow.py:239`) and `k` →
      `idf_data["1day"]` (`ventechow.py:22-23`) → `i_real` and every post-fit group. Verified in
      **tolerant** mode. SC-003 (amended 2026-10-29, ADR 0003) names this kind of change, a constant
      of a candidate distribution, and leaves out statistical-table values that cannot change the
      output.
- **Rationale**: clarification Q1 and principle I (identical in the same environment; per-field
  tolerance across environments). The factor 10 absorbs run-to-run noise on the development
  machine without approaching the magnitudes SC-003 must detect; it is a planning choice open
  to review. (Amended 2026-10-29: this holds for the pre-fit fields only; for the post-fit
  fields the measured differences make the factor-10 values larger than the magnitudes SC-003
  must detect, see ADR 0003.)
- **Alternatives considered**: a single global tolerance (rejected: D-01 makes `k, m, c, n`
  move far more than intensities); fixed guessed values (rejected by Q1: values come from
  measurement). Rejected after the measurement of 2026-10-29: the literal factor-10 rule for the
  post-fit groups, per-series tolerances, and no cross-environment comparison at all; see
  ADR 0003 (`docs/adr/0003-cross-environment-comparison-of-ill-conditioned-outputs.md`).

## R-06 — Synthetic fixtures (US3, FR-007)

- **Decision**: 8 synthetic CSVs in Hidroweb format (12 header lines, `;`, `,` decimal mark,
  ISO-8859-1, columns including `NivelConsistencia`, `Data`, `Maxima`), generated once by
  `scripts/make_synthetic_fixtures.py` with a fixed seed and committed. `.gitattributes`
  already applies `*.csv -text`, so their bytes are preserved.

  | ID | Covers (FR-007 category) | Expected current behavior | Marker |
  |---|---|---|---|
  | S1 | < 10 complete water years | insufficient-data string | — |
  | S2 | 10–29 years | full output, `sample_size_above_30_years: false` | — |
  | S3 | no consisted data (level 2 absent) | full output, `empty_consistent_data: true` | — |
  | S4 | empty water year in the middle | `empty_years` lists that year | — |
  | S5 | starts in October (ends mid-year) | current trimming, one complete year dropped | known defect D-02 / C-02 |
  | S6 | ends in September (starts mid-year) | current trimming, one complete year dropped | known defect D-02 / C-02 |
  | S7 | Oct/1990–Sep/2000 (mandatory case) | 8 years → insufficient-data string | known defect D-02 / C-02 |
  | S8 | a month missing in both consistency levels (a whole water year, Oct/1966–Sep/1967) | that water year listed in `empty_years`, through the `fillna(0)` path | suspected (D-13; changes under pandas 3) |

  **How S8 must be built.** `resample_data` fills with 0 every month missing *inside each
  level's own date range*, so a gap in the middle of both levels never reaches
  `merge_and_fill_data` as NaN. `merge_and_fill_data` merges `left` on the consisted months and
  takes the raw value for consisted months that are NaN or 0; the raw value is NaN only when the
  raw level has no row for that month. Therefore S8 needs a month that is:
  1. **inside the consisted period** (between the first and last consisted record), with its
     consisted `Maxima` **empty or 0** (or no consisted row for that month), and
  2. **earlier than the first raw (level 1) record**, so the left merge finds no raw row and
     `Maxima` is NaN when `merged_df['Maxima'].fillna(0, inplace=True)` runs
     (`process_data.py:67`).

  Under pandas 2.0.1 that line turns the NaN into 0; under pandas 3 it is a no-op (D-13). The
  rest of S8 (≥ 10 complete water years, both levels overlapping afterwards) must let the
  pipeline reach the fit, so the golden file captures the effect on the full output.

  **A whole water year, not a single month** (amended 2026-10-30, T026). A single month with
  these properties is neutralized: the per-year maximum (`add_water_year`) skips NaN, so the
  output is the same with or without the `fillna` and the golden file could not show D-13
  (verified in memory by replacing `merge_and_fill_data` with a copy without the `fillna`). S8
  therefore leaves the whole water year Oct/1966–Sep/1967 empty in the consisted level (from
  Jan/1965), before the first raw record (Mar/1968). Expected today: through the `fillna(0)`
  path the year's maximum is 0 and the year is listed in `empty_years`. Without the `fillna`
  the pipeline raises `ValueError: array must not contain infs or NaNs` (same in-memory check).
  FR-007 is still covered: every month of that year is missing in both consistency levels.

  The "expected" column is the hypothesis from the spec and the knowledge base; the golden file
  records what the code actually returns. If it differs from the hypothesis, the difference is
  reported to the maintainer, not "fixed". `tests/fixtures/synthetic/README.md` documents each
  fixture and its backlog item (FR-008). C-01 (D-01) has no per-station marker because it affects every fitted result
  (documented in the registry comment and in the synthetic README). Among the real series, only
  `chuvas_C_02346117` carries a marker, for C-03 (D-07).
- **Rationale**: SC-001 counts 7 categories; S5 and S6 isolate each boundary, S7 is the
  mandatory combined case from the spec.
- **Alternatives considered**: deriving fixtures by trimming real CSVs (rejected: modifies or
  copies golden inputs and couples cases to real data quirks).

## R-07 — Test dependencies

- **Decision**: `requirements-test.txt` with `pytest==8.3.5` only (supports Python 3.10).
  Installed together with `requirements.txt`; `requirements.txt` is unchanged. No plugins:
  network blocking, timing and output capture use pytest built-ins and the standard library.
- **Rationale**: user constraint; principle V and "no unused dependency".
- **Alternatives considered**: `pytest-socket` (one more dependency for a 10-line guard);
  `syrupy`/snapshot plugins (do not support per-field numeric tolerance with recorded reasons).

## R-08 — Exceptions and noise during the run

- **Decision**: if `main.main` raises for a series, the golden file records
  `result_kind: "exception"` with the exception type and message, and the test compares them.
  Standard output (`pprint` in `ventechow`, `print` in `main`) is captured by pytest and
  ignored. Library warnings are not turned into errors (`pytest.ini` has no `error`
  filterwarnings); they are not part of the golden file.
- **Rationale**: spec edge cases (warnings must not break the comparison); FR-010 (no removal
  of `pprint`, which is debt D-10 for spec 002).
- **Alternatives considered**: recording warnings in the golden file (rejected: they vary with
  library versions and would make tolerant mode noisy).

## R-09 — CI and required status check (FR-011)

- **Decision**: `.github/workflows/characterization.yml` with triggers `push` (all branches)
  and `workflow_dispatch` (inputs `reason`, `series`); no `pull_request` trigger. Jobs:
  - `characterization` — on `push` to any branch only; not on `pull_request` and not on
    `workflow_dispatch`, so each commit is checked once: runs `pytest` with
    `CALCULOIDF_COMPARE_MODE=exact`. This is the required status check. GitHub matches required
    checks by name on the PR's head SHA, so the push-triggered check on the branch head commit
    satisfies it; this meets FR-011 and US4 ("every push") as written. The ruleset also requires
    branches to be up to date before merging, so the tested head equals the merge result. A
    **draft PR to `main`**, opened by the maintainer before the first golden generation, is the
    review surface; the stability re-runs of R-03 are re-runs of this job started from it.
    PRs from forks would not get the push-triggered check and would stay blocked; acceptable
    for a single-maintainer repository.
  - `generate-golden` — on `push` to any branch when the head commit message contains
    `[regenerate-golden]`, and on `workflow_dispatch`: runs `update_golden.py` and uploads
    `tests/golden/` as an artifact (R-04). Not a required check.

  Both jobs share the reference environment setup (`ubuntu-24.04`, Python 3.10, the pins, the
  BLAS thread variables). Actions pinned by major version tag (`actions/checkout@v7`,
  `actions/setup-python@v7`, `actions/upload-artifact@v7`). The ruleset on `main` is **manual
  step M-1** in the plan.
  - **Deviation (2026-10-29)**: the plan first pinned `checkout@v4`, `setup-python@v5` and
    `upload-artifact@v4`. The first CI run (40913456358) warned that they target the deprecated
    Node.js 20, so the workflow uses the current majors, which run on Node.js 24. Release notes
    from our versions up to v7 were read; nothing affects this workflow:
    - `checkout` (v5 to v7): Node 24 (runner ≥ 2.327.1, met by hosted runners); v6 stores
      credentials in a separate file (the workflow runs no authenticated git command after
      checkout); v5.1/v6.1/v7 block fork-PR checkouts for `pull_request_target` and
      `workflow_run` (not used: triggers are `push` and `workflow_dispatch`). Default fetch
      behavior is not changed in these notes.
    - `setup-python` (v6, v7): Node 24; v7 removes the `pip-install` input (not used); the
      `python-version` input is unchanged.
    - `upload-artifact` (v5 to v7): Node 24; v7 adds `archive` (default `true`, so the
      `name`/`path` behavior is unchanged); hidden files have been excluded by default since
      v4.4 and `tests/golden/` has none.
- **Rationale**: a required status check is a repository setting, not code; listing it as a
  manual step keeps it explicit and reviewable.
- **Alternatives considered**: classic branch protection (rulesets are the current mechanism and
  can be exported as JSON for documentation).

## R-10 — Network guard (FR-004, SC-004)

- **Decision**: an autouse fixture in `conftest.py` replaces `socket.socket.connect` (and
  `socket.create_connection`) with a function that raises; the test run proves zero network
  calls. Combined with the R-01 stub, no credential or metadata lookup can happen.
- **Alternatives considered**: running CI with networking disabled (not available per job on
  hosted runners).

## R-11 — Reference changelog (FR-006, SC-005)

- **Decision**: `tests/golden/CHANGELOG.md`, one entry per run of `update_golden.py` that
  changes at least one golden file: date (UTC), series IDs, reason, environment fingerprint,
  trigger, a Markdown table with one row per changed series (series id, number of changed
  leaves, max absolute delta, max relative delta) and the changed field paths (computed by
  diffing old vs new golden). The script prints the same table, ready to paste into the PR
  description.
  - A golden file is rewritten only when `result_kind`, `result`, `metadata.input_sha256` or
    `metadata.marker` change; `generated_at` (or `environment`) alone does not count, and an
    unchanged file is left byte-for-byte. A regeneration of every series therefore touches only
    the series whose content changed.
  - The script refuses to run without a non-empty `--reason` and refuses to write when the
    current fingerprint does not match `REFERENCE_FINGERPRINT` (R-03) unless
    `--allow-non-reference` is given (that flag is only for local experiments and writes
    nothing to the changelog — it outputs to a scratch directory).
  - For its unit tests, the golden root and the reference fingerprint are internal parameters of
    the script's entry function, injected with `tmp_path` and `monkeypatch`; they are not CLI
    flags.
- **Rationale**: US2 acceptance scenarios; SC-005; constitution principle I (the PR shows the
  table of differences per station).

## Measurement results

Task T017, measured on 2026-10-29 with `scripts/measure_env_diff.py` (the golden files compared
with a run in the current environment, without tolerances). The golden files come from CI run
40916698777.

- **Golden (reference) environment**: `os=Linux`, `machine=x86_64`, `python=3.10.22`,
  `numpy=1.24.3`, `pandas=2.0.1`, `scipy=1.10.1`, `scikit-learn=1.3.0`,
  `OPENBLAS_CORETYPE=Haswell`,
  `NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL`.
- **Measured environment**: `os=Windows`, `machine=AMD64`, `python=3.10.22`, `numpy=1.24.3`,
  `pandas=2.0.1`, `scipy=1.10.1`, `scikit-learn=1.3.0`, the same two kernel variables. The `.venv`
  matches the 43 pins of `requirements.txt`.
- The two series that end in the insufficient-data message (`chuvas_C_01848001_UB`,
  `chuvas_C_02043032_MA`) have no numeric leaves and match. No non-numeric difference was
  found in any series.
- An independent script on another CPU (Intel i7-9750H, Linux VM) with the pinned kernels
  reproduced all 13 golden files exactly (maintainer's cross-check, 2026-10-29).

### Per field group and series

| Field group | Series | Numeric leaves | Differing leaves | Max abs Δ | Max rel Δ |
|---|---|---|---|---|---|
| `parameters.**` | chuvas_C_01844000_CV | 8 | 3 | 2.205e-01 | 9.464e-04 |
| `parameters.**` | chuvas_C_01844000_CV1 | 8 | 4 | 8.799e+00 | 1.946e-02 |
| `parameters.**` | chuvas_C_01943011 | 8 | 6 | 3.356e-01 | 8.003e-04 |
| `parameters.**` | chuvas_C_01944009_PL | 8 | 2 | 7.230e-02 | 6.028e-04 |
| `parameters.**` | chuvas_C_02040003_FFL | 8 | 4 | 2.033e+00 | 5.650e-03 |
| `parameters.**` | chuvas_C_02040022 | 8 | 8 | 4.141e+00 | 1.913e-02 |
| `parameters.**` | chuvas_C_02043011_FAZ_P | 8 | 8 | 5.149e+01 | 1.737e-01 |
| `parameters.**` | chuvas_C_02243077_HF_RIO | 8 | 8 | 7.545e+01 | 1.582e-01 |
| `parameters.**` | chuvas_C_02243155 | 8 | 8 | 1.760e+01 | 4.843e-02 |
| `parameters.**` | chuvas_C_02346117 | 8 | 6 | 5.840e-02 | 8.015e-03 |
| `parameters.**` | chuvas_C_02346328 | 8 | 6 | 2.554e+01 | 1.795e-01 |
| `intensity_graph_data_*.i_calculated[*]` | chuvas_C_01844000_CV | 120 | 64 | 1.099e-02 | 2.958e-04 |
| `intensity_graph_data_*.i_calculated[*]` | chuvas_C_01844000_CV1 | 120 | 64 | 1.331e-01 | 4.089e-03 |
| `intensity_graph_data_*.i_calculated[*]` | chuvas_C_01943011 | 120 | 120 | 5.251e-02 | 6.782e-04 |
| `intensity_graph_data_*.i_calculated[*]` | chuvas_C_01944009_PL | 120 | 56 | 7.210e-02 | 3.163e-04 |
| `intensity_graph_data_*.i_calculated[*]` | chuvas_C_02040003_FFL | 120 | 64 | 6.790e-02 | 1.040e-03 |
| `intensity_graph_data_*.i_calculated[*]` | chuvas_C_02040022 | 120 | 120 | 3.243e-01 | 2.909e-03 |
| `intensity_graph_data_*.i_calculated[*]` | chuvas_C_02043011_FAZ_P | 120 | 120 | 1.375e+00 | 1.169e-02 |
| `intensity_graph_data_*.i_calculated[*]` | chuvas_C_02243077_HF_RIO | 120 | 120 | 9.583e-01 | 1.698e-02 |
| `intensity_graph_data_*.i_calculated[*]` | chuvas_C_02243155 | 120 | 120 | 3.883e-01 | 4.993e-03 |
| `intensity_graph_data_*.i_calculated[*]` | chuvas_C_02346117 | 120 | 120 | 2.433e-01 | 5.561e-04 |
| `intensity_graph_data_*.i_calculated[*]` | chuvas_C_02346328 | 120 | 120 | 5.576e-01 | 1.175e-02 |
| `intensity_graph_data_*.regression.*` | chuvas_C_01844000_CV | 4 | 2 | 5.817e-05 | 1.164e-03 |
| `intensity_graph_data_*.regression.*` | chuvas_C_01844000_CV1 | 4 | 2 | 1.316e-02 | 6.397e-02 |
| `intensity_graph_data_*.regression.*` | chuvas_C_01943011 | 4 | 4 | 3.322e-02 | 3.447e-02 |
| `intensity_graph_data_*.regression.*` | chuvas_C_01944009_PL | 4 | 4 | 1.910e-02 | 1.146e-02 |
| `intensity_graph_data_*.regression.*` | chuvas_C_02040003_FFL | 4 | 2 | 1.589e-03 | 6.214e-03 |
| `intensity_graph_data_*.regression.*` | chuvas_C_02040022 | 4 | 4 | 1.561e-01 | 5.385e-02 |
| `intensity_graph_data_*.regression.*` | chuvas_C_02043011_FAZ_P | 4 | 4 | 8.788e-02 | 3.252e-01 |
| `intensity_graph_data_*.regression.*` | chuvas_C_02243077_HF_RIO | 4 | 4 | 4.213e-01 | 4.289e-01 |
| `intensity_graph_data_*.regression.*` | chuvas_C_02243155 | 4 | 4 | 8.992e-02 | 2.411e-01 |
| `intensity_graph_data_*.regression.*` | chuvas_C_02346117 | 4 | 4 | 2.340e-02 | 3.059e-02 |
| `intensity_graph_data_*.regression.*` | chuvas_C_02346328 | 4 | 4 | 1.160e-01 | 3.482e+00 |
| `mean_relative_errors.*` | chuvas_C_01844000_CV | 2 | 1 | 4.780e-05 | 1.978e-05 |
| `mean_relative_errors.*` | chuvas_C_01844000_CV1 | 2 | 1 | 2.830e-03 | 1.130e-03 |
| `mean_relative_errors.*` | chuvas_C_01943011 | 2 | 2 | 1.510e-04 | 9.054e-05 |
| `mean_relative_errors.*` | chuvas_C_01944009_PL | 2 | 1 | 4.588e-04 | 1.740e-04 |
| `mean_relative_errors.*` | chuvas_C_02040003_FFL | 2 | 1 | 1.075e-03 | 2.070e-04 |
| `mean_relative_errors.*` | chuvas_C_02040022 | 2 | 2 | 3.015e-03 | 9.410e-04 |
| `mean_relative_errors.*` | chuvas_C_02043011_FAZ_P | 2 | 2 | 8.688e-02 | 2.919e-02 |
| `mean_relative_errors.*` | chuvas_C_02243077_HF_RIO | 2 | 2 | 1.415e-02 | 4.942e-03 |
| `mean_relative_errors.*` | chuvas_C_02243155 | 2 | 2 | 2.187e-03 | 5.180e-04 |
| `mean_relative_errors.*` | chuvas_C_02346117 | 2 | 2 | 5.600e-04 | 1.781e-04 |
| `mean_relative_errors.*` | chuvas_C_02346328 | 2 | 2 | 1.400e-03 | 5.782e-04 |
| `ns.*` | chuvas_C_01844000_CV | 2 | 1 | 1.532e-06 | 1.535e-06 |
| `ns.*` | chuvas_C_01844000_CV1 | 2 | 1 | 5.422e-05 | 5.432e-05 |
| `ns.*` | chuvas_C_01943011 | 2 | 2 | 1.984e-05 | 1.990e-05 |
| `ns.*` | chuvas_C_01944009_PL | 2 | 1 | 8.929e-06 | 8.984e-06 |
| `ns.*` | chuvas_C_02040003_FFL | 2 | 1 | 9.763e-06 | 9.823e-06 |
| `ns.*` | chuvas_C_02040022 | 2 | 2 | 1.778e-05 | 1.798e-05 |
| `ns.*` | chuvas_C_02043011_FAZ_P | 2 | 2 | 6.826e-04 | 6.896e-04 |
| `ns.*` | chuvas_C_02243077_HF_RIO | 2 | 2 | 1.170e-04 | 1.177e-04 |
| `ns.*` | chuvas_C_02243155 | 2 | 2 | 7.901e-05 | 7.930e-05 |
| `ns.*` | chuvas_C_02346117 | 2 | 2 | 2.655e-05 | 2.661e-05 |
| `ns.*` | chuvas_C_02346328 | 2 | 2 | 1.196e-04 | 1.198e-04 |
| `graph_data.**` | chuvas_C_01844000_CV | 99 | 0 | 0 | 0 |
| `graph_data.**` | chuvas_C_01844000_CV1 | 93 | 0 | 0 | 0 |
| `graph_data.**` | chuvas_C_01943011 | 90 | 13 | 9.948e-14 | 9.729e-16 |
| `graph_data.**` | chuvas_C_01944009_PL | 192 | 0 | 0 | 0 |
| `graph_data.**` | chuvas_C_02040003_FFL | 108 | 12 | 1.705e-13 | 1.109e-15 |
| `graph_data.**` | chuvas_C_02040022 | 96 | 0 | 0 | 0 |
| `graph_data.**` | chuvas_C_02043011_FAZ_P | 192 | 3 | 2.842e-14 | 6.495e-16 |
| `graph_data.**` | chuvas_C_02243077_HF_RIO | 39 | 0 | 0 | 0 |
| `graph_data.**` | chuvas_C_02243155 | 66 | 1 | 2.842e-14 | 5.304e-16 |
| `graph_data.**` | chuvas_C_02346117 | 159 | 43 | 9.948e-13 | 6.131e-15 |
| `graph_data.**` | chuvas_C_02346328 | 57 | 16 | 1.847e-13 | 2.743e-15 |
| `**` | chuvas_C_01844000_CV | 120 | 0 | 0 | 0 |
| `**` | chuvas_C_01844000_CV1 | 120 | 0 | 0 | 0 |
| `**` | chuvas_C_01943011 | 120 | 45 | 1.990e-13 | 1.199e-15 |
| `**` | chuvas_C_01944009_PL | 120 | 0 | 0 | 0 |
| `**` | chuvas_C_02040003_FFL | 121 | 0 | 0 | 0 |
| `**` | chuvas_C_02040022 | 120 | 0 | 0 | 0 |
| `**` | chuvas_C_02043011_FAZ_P | 120 | 0 | 0 | 0 |
| `**` | chuvas_C_02243077_HF_RIO | 120 | 0 | 0 | 0 |
| `**` | chuvas_C_02243155 | 126 | 15 | 2.842e-13 | 1.140e-15 |
| `**` | chuvas_C_02346117 | 120 | 120 | 6.821e-13 | 2.169e-15 |
| `**` | chuvas_C_02346328 | 127 | 0 | 0 | 0 |

### Maximum per field group (all series)

| Field group | Numeric leaves | Differing leaves | Max abs Δ | Max rel Δ |
|---|---|---|---|---|
| `parameters.**` | 88 | 63 | 7.545e+01 | 1.795e-01 |
| `intensity_graph_data_*.i_calculated[*]` | 1320 | 1088 | 1.375e+00 | 1.698e-02 |
| `intensity_graph_data_*.regression.*` | 44 | 38 | 4.213e-01 | 3.482e+00 |
| `mean_relative_errors.*` | 22 | 18 | 8.688e-02 | 2.919e-02 |
| `ns.*` | 22 | 18 | 6.826e-04 | 6.896e-04 |
| `graph_data.**` | 1191 | 88 | 9.948e-13 | 6.131e-15 |
| `**` | 1334 | 180 | 6.821e-13 | 2.169e-15 |

Reading: the pre-fit groups differ by rounding noise only; every non-trivial difference is in
the post-fit groups. Decision and consequences: ADR 0003.

The tables above (T017) cover the 13 real series and are kept as history; the measurement over
all 21 series follows.

### Measurement over all 21 series (T042)

Task T042, measured on 2026-10-31 with `scripts/measure_env_diff.py` over the 13 real and the 8
synthetic series, against the golden files of CI runs 40916698777 (real) and 41159855775
(synthetic). Same golden (reference) and measured environments as T017. The rows of the 13 real
series are identical to the T017 table above. S1 and S7 end in the insufficient-data message,
have no numeric leaves and match. No non-numeric difference was found in any series.

**Synthetic series**

| Field group | Series | Numeric leaves | Differing leaves | Max abs Δ | Max rel Δ |
|---|---|---|---|---|---|
| `parameters.**` | S2 | 8 | 5 | 9.299e-01 | 1.980e-03 |
| `parameters.**` | S3 | 8 | 3 | 1.548e+00 | 9.657e-03 |
| `parameters.**` | S4 | 8 | 4 | 1.505e+02 | 5.867e-01 |
| `parameters.**` | S5 | 8 | 2 | 4.000e-04 | 9.675e-06 |
| `parameters.**` | S6 | 8 | 3 | 1.218e+00 | 5.740e-03 |
| `parameters.**` | S8 | 8 | 0 | 0 | 0 |
| `intensity_graph_data_*.i_calculated[*]` | S2 | 120 | 120 | 1.414e-02 | 2.687e-04 |
| `intensity_graph_data_*.i_calculated[*]` | S3 | 120 | 64 | 3.268e-02 | 7.514e-04 |
| `intensity_graph_data_*.i_calculated[*]` | S4 | 120 | 64 | 1.218e+00 | 4.071e-02 |
| `intensity_graph_data_*.i_calculated[*]` | S5 | 120 | 64 | 2.135e-05 | 5.433e-07 |
| `intensity_graph_data_*.i_calculated[*]` | S6 | 120 | 64 | 9.358e-03 | 2.892e-04 |
| `intensity_graph_data_*.i_calculated[*]` | S8 | 120 | 0 | 0 | 0 |
| `intensity_graph_data_*.regression.*` | S2 | 4 | 4 | 8.698e-03 | 4.443e-01 |
| `intensity_graph_data_*.regression.*` | S3 | 4 | 2 | 1.813e-03 | 2.085e-02 |
| `intensity_graph_data_*.regression.*` | S4 | 4 | 4 | 2.439e-01 | 1.437e+00 |
| `intensity_graph_data_*.regression.*` | S5 | 4 | 2 | 8.256e-06 | 1.605e-04 |
| `intensity_graph_data_*.regression.*` | S6 | 4 | 2 | 1.353e-03 | 1.091e+00 |
| `intensity_graph_data_*.regression.*` | S8 | 4 | 0 | 0 | 0 |
| `mean_relative_errors.*` | S2 | 2 | 2 | 1.087e-03 | 5.675e-04 |
| `mean_relative_errors.*` | S3 | 2 | 1 | 3.244e-04 | 1.749e-04 |
| `mean_relative_errors.*` | S4 | 2 | 1 | 3.697e-01 | 1.506e-01 |
| `mean_relative_errors.*` | S5 | 2 | 1 | 1.364e-06 | 6.368e-07 |
| `mean_relative_errors.*` | S6 | 2 | 1 | 1.951e-03 | 1.010e-03 |
| `mean_relative_errors.*` | S8 | 2 | 0 | 0 | 0 |
| `ns.*` | S2 | 2 | 2 | 4.566e-06 | 4.571e-06 |
| `ns.*` | S3 | 2 | 1 | 3.451e-06 | 3.454e-06 |
| `ns.*` | S4 | 2 | 1 | 1.971e-05 | 1.975e-05 |
| `ns.*` | S5 | 2 | 1 | 2.649e-09 | 2.653e-09 |
| `ns.*` | S6 | 2 | 1 | 7.070e-06 | 7.079e-06 |
| `ns.*` | S8 | 2 | 0 | 0 | 0 |
| `graph_data.**` | S2 | 57 | 3 | 2.842e-14 | 4.968e-16 |
| `graph_data.**` | S3 | 99 | 0 | 0 | 0 |
| `graph_data.**` | S4 | 99 | 30 | 2.416e-13 | 4.138e-15 |
| `graph_data.**` | S5 | 57 | 0 | 0 | 0 |
| `graph_data.**` | S6 | 60 | 0 | 0 | 0 |
| `graph_data.**` | S8 | 99 | 0 | 0 | 0 |
| `**` | S2 | 120 | 15 | 7.105e-14 | 6.152e-16 |
| `**` | S3 | 120 | 0 | 0 | 0 |
| `**` | S4 | 121 | 0 | 0 | 0 |
| `**` | S5 | 120 | 0 | 0 | 0 |
| `**` | S6 | 120 | 0 | 0 | 0 |
| `**` | S8 | 121 | 0 | 0 | 0 |

**Maximum per field group (all 21 series)**

| Field group | Numeric leaves | Differing leaves | Max abs Δ | Max rel Δ |
|---|---|---|---|---|
| `parameters.**` | 136 | 80 | 1.505e+02 | 5.867e-01 |
| `intensity_graph_data_*.i_calculated[*]` | 2040 | 1464 | 1.375e+00 | 4.071e-02 |
| `intensity_graph_data_*.regression.*` | 68 | 52 | 4.213e-01 | 3.482e+00 |
| `mean_relative_errors.*` | 34 | 24 | 3.697e-01 | 1.506e-01 |
| `ns.*` | 34 | 24 | 6.826e-04 | 6.896e-04 |
| `graph_data.**` | 1662 | 121 | 9.948e-13 | 6.131e-15 |
| `**` | 2056 | 195 | 6.821e-13 | 2.169e-15 |

Reading: the pre-fit maxima are those of T017, so the numeric tolerances of
`tests/golden/tolerances.json` are unchanged. S4 raises the post-fit maxima of `parameters.**`
(5.867e-01 against 1.795e-01), `i_calculated` (4.071e-02 against 1.698e-02) and
`mean_relative_errors.*` (1.506e-01 against 2.919e-02); these groups are `structure` rules, so
only their recorded `measured_max_*` values and reasons were updated (`measured_at` 2026-10-31).
No `kind`, `rel_tol` or `abs_tol` changed.

## Stability verification (T020)

The `characterization` job (exact mode, `CALCULOIDF_COMPARE_MODE=exact`, pinned kernels of R-03)
passed on three runs, each with a different runner name. Data from the GitHub API and the job
logs.

| Run | Run id | Job id | Head | Runner | CPU model | Conclusion | pytest summary |
|---|---|---|---|---|---|---|---|
| 1 | 40947614206 | 123707004679 | `8bdbf60` | GitHub Actions 1000000279 | not logged; no AVX-512 (numpy warning) | success | 61 passed |
| 2 | 41124734166 | 124300132129 | `2410d69` | GitHub Actions 1000000280 | not logged; no AVX-512 (numpy warning) | success | 113 passed |
| 3 | 41127421864 | 124309444025 | `7ee0592` | GitHub Actions 1000000281 | AMD EPYC 9V74 80-Core Processor (AVX-512) | success | 113 passed |

- The CPU model is known only for run 3: the step that logs it was added in `7ee0592`. For runs
  1 and 2 the CPU class comes from numpy, which warns at import when the CPU lacks the features
  listed in `NPY_DISABLE_CPU_FEATURES`: `RuntimeWarning: During parsing environment variable
  'NPY_DISABLE_CPU_FEATURES': You cannot disable CPU features (AVX512F AVX512CD AVX512_SKX
  AVX512_CLX AVX512_CNL AVX512_ICL), since they are not supported by your machine.` The warning
  appears once in each of jobs 123707004679 and 124300132129 and is absent from job 124309444025.
- The golden files were generated on a CPU without AVX-512: the same warning appears in the
  `generate-golden` job (123598723308, runner GitHub Actions 1000000278) of run 40916698777.
- Run 3's CPU has AVX-512 (`avx512f` present). With `OPENBLAS_CORETYPE=Haswell` and the
  AVX-512 features disabled in numpy, it runs the same pinned Haswell/AVX2 kernels as the CPUs
  without AVX-512. Its match with the golden files shows that the pinning works on a CPU where it
  is needed.
- Run 1 has fewer tests because its head precedes `b77aaaf` and `2410d69`, which added the
  unit tests of the `structure` rule and of the tolerance file; the 13 real series
  (`test_golden.py`) passed in all three runs.
- Deviation: the three runs are new pushes, each on a new runner, not re-runs from the draft PR
  as T020 states; the evidence is equivalent.
- Additional evidence (T029): run 41159855775, job 124414788296, runner GitHub Actions
  1000000282, AMD EPYC 7763 (no AVX-512); the 13 real series passed in exact mode (the job
  failed only on the 8 synthetic series, which had no golden files yet).

## Ruleset and blocking demonstration (T034, T035)

**Ruleset (T034, M-1)**, verified through the GitHub API on 2026-10-31: id 26272232,
"main protection", enforcement `active`, target `branch`, condition `~DEFAULT_BRANCH` (`main`),
no bypass actors (`current_user_can_bypass: never`). Effective rules on `main`:

- `deletion` and `non_fast_forward`;
- `required_status_checks`: `characterization` from GitHub Actions (integration 15368), strict
  policy (branch must be up to date before merging);
- `pull_request`: 0 required approvals, allowed merge methods `merge` only (M-1 as amended on
  2026-10-31).

**Blocking demonstration (T035)**: on a throwaway branch `t035-blocking-demo` from `ab486b5`,
the `"1h"` coefficient in `src/disaggregation_coef.py` was changed from 0.42 to 0.421 (commit
`eda2941`) and PR #2 was opened to `main`.

- Check `characterization` on the PR head: run 41207406904, job 124555203942, runner GitHub
  Actions 1000000286, AMD EPYC 7763 (no AVX-512), exact mode: failure, `17 failed, 105 passed`.
- The 17 failures are the 11 real series with numeric output and S2–S6 and S8. The 4 series
  that end in the insufficient-data message (`chuvas_C_01848001_UB`, `chuvas_C_02043032_MA`,
  S1, S7) have no numeric output and passed.
- Fields reached (the CI message lists the first 50 differences per series; the groups were
  read with `scripts/measure_env_diff.py` on the branch, for `chuvas_C_02040022` and S2): the
  pre-fit `i_real` (group `**`, max relative change 2.381e-03 = 0.001/0.42) and every post-fit
  group (`parameters`, `i_calculated`, `regression`, `mean_relative_errors`, `ns`) changed;
  `graph_data` did not, as R-05 predicted.
- `mergeStateStatus: BLOCKED` (`mergeable: MERGEABLE`, i.e. no conflict; the block came from
  the ruleset). PR #2 was closed without merge and the branch was deleted.

## Sensitivity, network and determinism checks (T036–T038)

Run on 2026-10-31 from `4268d97`. Each perturbation was exactly 0.001, never committed to
`001-characterization-tests`, and reverted before the next one (`git diff --exit-code -- src/`
exited 0 after each).

**SC-003 sensitivity (T036)**

- (a) Disaggregation coefficient `"2h"` 0.52 → 0.521 (`src/disaggregation_coef.py`), tolerant
  mode on Windows, `pytest tests/characterization`: `17 failed, 4 passed`. The failures are the
  11 real series with numeric output and S2–S6, S8; the 4 insufficient-data series passed. Every
  reported difference is `intensity_graph_data_2.i_real[*]` under the pre-fit `**` rule (8 or 9
  per series), e.g. `intensity_graph_data_2.i_real[45]` 22.557793849006924 → 22.601174221793475
  (`rule=tolerant ** rel_tol=1e-12 abs_tol=7e-12`). The post-fit groups are structure-only in
  tolerant mode (ADR 0003), so they report nothing.
- (b) BFGS initial guess, component `m` 0.1 → 0.101 (`INITIAL_GUESS` in `src/ventechow.py`),
  exact mode in the reference environment. Deviation from the task text: instead of a VM or
  WSL, the CI reference environment itself was used, through the throwaway branch
  `t036-initial-guess` (commit `3a6820e`, pushed without a PR, then deleted); this is equivalent
  or stronger, since it is the environment that generated the golden files. Check
  `characterization`: run 41210406613, job 124564022887, runner GitHub Actions 1000000288, AMD
  EPYC 7763 (no AVX-512): failure, `17 failed, 105 passed`, the same 17 series as (a). Each
  failing series has 135 or 136 differences, against 136 post-fit numeric leaves per series
  (`parameters` 8, `i_calculated` 120, `regression` 4, `mean_relative_errors` 2, `ns` 2), e.g.
  `intensity_graph_data_1.i_calculated[*]`. `scripts/measure_env_diff.py` on that branch (Windows,
  all 21 series) gave the pre-fit maxima of T017 unchanged: `graph_data.**` 9.948e-13 abs /
  6.131e-15 rel, `**` 6.821e-13 abs / 2.169e-15 rel, no non-numeric difference. The initial guess
  therefore reaches the post-fit fields only, as R-05 predicted.
- (c) Theoretical Gumbel coefficient 0.7797 → 0.7807 in both `src/distributions.py:99` and
  `src/k_coefficient.py:50`, tolerant mode on Windows: `6 failed, 15 passed`. The failures are
  exactly the 6 series whose chosen distribution is `Gumbel Teórica` (`chuvas_C_01844000_CV`,
  `chuvas_C_01844000_CV1`, `chuvas_C_01944009_PL`, S3, S6, S8). For `chuvas_C_01944009_PL`, 184
  differences: 120 `intensity_graph_data_*.i_real[*]` (rule `**`) and 64 `graph_data.P_dist[*]`
  (rule `graph_data.**`), e.g. `graph_data.P_dist[0]` 45.22143997305043 → 45.195526713815966.

In every case the failure reported series id, field path, expected value, actual value and rule
(FR-003). No numeric tolerance hid a perturbation that reaches a pre-fit group.

**SC-004 / FR-004 network and credentials (T037)**

- `GOOGLE_APPLICATION_CREDENTIALS` unset and no application-default credentials file on the
  machine.
- The network adapter was not disabled. Instead, an ad-hoc pytest plugin outside the repository
  replaced `socket.socket.connect`, `socket.socket.connect_ex`, `socket.create_connection` and
  `socket.getaddrinfo` for the whole session (collection included) with functions that record
  the attempt and raise. Full suite: `122 passed`, `0 network attempt(s)`. Positive control: a
  test calling `urllib.request.urlopen` and `socket.create_connection` under the same plugin
  recorded 2 attempts.
- SC-002: full suite (21 series and the unit tests) in 37.92 s (pytest) and 37.0 s wall time,
  under 60 s.

**FR-005 / FR-010 determinism and production code (T038)**

- Two consecutive full runs on Windows (tolerant mode): 122 tests passed in both, with
  identical per-test outcomes (`pytest -rA` lists compared).
- CI exact mode on unchanged `src/`: runs 41160365316 (`25d4cbb`), 41205456211 (`ab486b5`) and
  41209553808 (`4268d97`), `122 passed` each.
- `git diff main -- src/ requirements.txt` is empty (`main` at `7fedeac`).
- No run differed from the golden files; neither the golden files nor the tolerances were
  changed.

## Quickstart validation (T040)

Run on 2026-10-31 from `e3deba1` on the Windows development machine (`.venv`, Python 3.10.22,
the pins of `requirements.txt`), with `CALCULOIDF_COMPARE_MODE` unset (tolerant mode by
fingerprint). `src/` was not modified.

| Scenario | Outcome | Deviation |
|---|---|---|
| 1 — suite passes | `122 passed` (unit tests and 21 series) in 36.66 s (pytest), 37.8 s wall time; no network error from the guard | none |
| 2 — numeric change detected | not re-run | Deviation: the three perturbations require editing `src/`; quickstart asks to record them once, which T036 did (section above, from `4268d97`) |
| 3 — regeneration requires a reason | no `--reason` → exit 2 (argparse); `--reason "   "` → exit 2 (`--reason must not be empty`); `--reason "…"` on Windows → exit 3 (not the reference environment); `git status` unchanged | The CI paths were not re-run here: `[regenerate-golden]` push was exercised by T014 and T029; the dispatch after the merge is T025, still open |
| 4 — edge cases frozen | `S7.json`: `result_kind` `string`, the insufficient-data message, marker `known-defect` D-02 / C-02. S1 string; S2 `sample_size_above_30_years: false`; S3 `empty_consistent_data: true`; S4 `empty_years: [1985]`; S5 and S6 marker D-02 / C-02; S8 `empty_years: [1966]`, marker `suspected` D-13 | none against R-06 |
| 5 — measurement and tolerances | `scripts/measure_env_diff.py --out <scratch>/measure.md`, all 21 series, exit 0, no non-numeric difference, nothing written under `tests/golden/`. Pre-fit maxima equal to T017: `graph_data.**` 9.948e-13 abs / 6.131e-15 rel, `**` 6.821e-13 abs / 2.169e-15 rel, within the numeric tolerances of `tests/golden/tolerances.json` | See below: post-fit maxima over 21 series exceed the T017 values |
| 6 — automatic protection | not re-run | Deviation: requires a push and a PR; demonstrated by T035 (PR #2, `mergeStateStatus: BLOCKED`) |
| 7 — determinism and no production change | second run `122 passed` in 37.27 s, per-test outcomes (`pytest -rA`) identical to scenario 1; `git diff main -- src/ requirements.txt` empty (`main` at `7fedeac`); `git diff tcc-2023 -- src/ requirements.txt` empty | none |

**Post-fit maxima over the 21 series (scenario 5).** T017 measured the 13 real series only. With
the 8 synthetic series included, S4 has the largest Windows × Linux differences in three
post-fit groups:

| Field group | Max abs Δ (21 series) | Max rel Δ (21 series) | Series | Max rel Δ in T017 (13 real) |
|---|---|---|---|---|
| `parameters.**` | 1.505e+02 | 5.867e-01 | S4 | 1.795e-01 |
| `intensity_graph_data_*.i_calculated[*]` | 1.218e+00 (S4); 1.375e+00 overall | 4.071e-02 | S4 | 1.698e-02 |
| `mean_relative_errors.*` | 3.697e-01 | 1.506e-01 | S4 | 2.919e-02 |

`regression.*` and `ns.*` keep the T017 maxima. S8 has no post-fit difference. T031's condition
(a synthetic series moving a tolerance group beyond T018) was therefore met in the structure
groups: S4 exceeds the T017 maxima. These groups are `structure` rules (ADR 0003), so the suite
outcome does not change; the pre-fit groups, which carry the numeric tolerances, are not moved.
The measurement record (`measured_max_*` in `tests/golden/tolerances.json`) covered the 13 real
series only; it is updated over all 21 series by T042.

## Convergence (T041)

`/speckit-converge` run on 2026-10-31 from `e3deba1` (T040 staged). No hooks registered, no
constitution violation, no `missing` or `contradicts` finding. Three findings; the maintainer
decided each one.

| ID | Gap type | Severity | Source | Finding | Resolution |
|---|---|---|---|---|---|
| F1 | partial | MEDIUM | FR-002, T018, T031 | The measurement record in `tests/golden/tolerances.json` covered the 13 real series only; with the 8 synthetic series, S4 exceeds the T017 maxima of three post-fit (`structure`) groups | Fixed by T042: re-measured over all 21 series; measurement fields and the 21-series table updated, no `kind`, `rel_tol` or `abs_tol` changed ("Measurement over all 21 series (T042)") |
| F2 | partial | LOW | T028, FR-008 | The S8 marker note in `tests/characterization/compare.py` differs from the text of T028 | The note was adjusted on the maintainer's instruction; T043 aligned the T028 text with it. The marker and `S8.json` are unchanged |
| F3 | unrequested | LOW | plan: Project Structure | `tests/unit/test_tolerances_file.py`, `tests/unit/test_bootstrap.py` and the "Log runner CPU" workflow step are not in the plan's structure | Accepted, no plan change: `test_bootstrap.py` covers the kernel and thread variables pinned by R-03 (`bootstrap.py`, T004); `test_tolerances_file.py` checks the tolerance file filled by T018 (rule kinds, mandatory reasons, recorded measurement); the CPU log step provides the runner CPU evidence recorded by T020 |

T025 (manual, after the merge) remains open by design and is not a convergence finding.
