# Data Model: Characterization tests (001)

Entities from the spec (Key Entities), with fields and validation rules. File formats are in
[contracts/](contracts/).

## Reference series

An input CSV in Hidroweb format.

| Field | Description | Rule |
|---|---|---|
| `id` | Stable identifier: the CSV file name without extension (real) or `S1`…`S8` (synthetic) | Unique across real and synthetic |
| `kind` | `real` or `synthetic` | — |
| `path` | `src/csv/<file>.csv` (real, read in place) or `tests/fixtures/synthetic/<file>.csv` | File exists; never modified by the suite |
| `sha256` | Hash of the input bytes | Stored in the golden metadata; a mismatch fails the test (input changed) |
| `covers` | What the series exercises (FR-007 category) | Required for synthetic series |
| `marker` | `known-defect` / `suspected` / none, with backlog item (`C-01`, `C-02`, `C-03`) and debt ID | Required when the case freezes a defect (FR-008) |

Real series (13): the files in `src/csv/`. Synthetic series (8): see research R-06.

## Frozen reference (golden file)

One JSON file per reference series: `tests/golden/real/<id>.json` or
`tests/golden/synthetic/<id>.json`.

| Field | Description | Rule |
|---|---|---|
| `metadata.series_id`, `metadata.input_path`, `metadata.input_sha256` | Link to the series | Must match the series |
| `metadata.generated_at` | UTC timestamp of generation | ISO 8601 |
| `metadata.environment` | Fingerprint: OS, Python `major.minor.micro`, numpy, pandas, scipy, scikit-learn versions, machine architecture | Must be the reference environment for committed files (FR-009) |
| `metadata.marker` | Copied from the series | Present when applicable |
| `result_kind` | `object` / `string` / `exception` | — |
| `result` | Normalized output of `main.main` (object), the decoded string, or `{type, message}` | Full output, no field omitted (FR-001) |

**State transitions**: *absent* → *generated* (initial run, changelog entry) → *regenerated*
(only via `update_golden.py --reason`, new changelog entry). A golden file is never edited by
hand.

## Tolerance set

`tests/golden/tolerances.json`: per-field tolerances used only in `tolerant` mode.

| Field | Description | Rule |
|---|---|---|
| `measured_at`, `reference_environment`, `measured_environment` | Provenance of the measurement | Required |
| `safety_factor` | Multiplier applied to measured maxima (planned: 10) | Required |
| `rules[]` | Ordered list of `{pattern, rel_tol, abs_tol, measured_max_abs, measured_max_rel}` | First matching pattern wins; last rule is the default `**` |

## Reference change log entry

`tests/golden/CHANGELOG.md`, append-only.

| Field | Rule |
|---|---|
| Date (UTC) | Required |
| Series IDs | Required, ≥ 1 |
| Per changed series: number of changed leaves, max absolute delta, max relative delta | Computed by the script; a Markdown table, one row per series, that can be pasted into the PR (constitution principle I) |
| Changed field paths (or "initial generation") | Computed by the script, not typed |
| Reason | Required, non-empty (FR-006); the script refuses otherwise |
| Environment fingerprint | Required |
| Trigger | URL of the `generate-golden` workflow run (push with `[regenerate-golden]` or `workflow_dispatch`), or "local (reference environment)" |

## Comparison result (runtime only)

Produced by the comparator on failure; not stored.

| Field | Description |
|---|---|
| `series_id` | Station / case |
| `path` | Field path, e.g. `parameters.parameters_1.k1`, `graph_data.F[3]` |
| `expected`, `actual` | Values (or types, for a type mismatch) |
| `rule` | Tolerance rule applied (`exact` or the matched pattern) |
