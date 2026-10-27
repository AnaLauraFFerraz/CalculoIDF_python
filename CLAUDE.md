# CalculoIDF_python

Computation core of CalculoIDF: reads a Hidroweb (ANA) rainfall series as CSV and fits the station's IDF equation `i = k · Tr^m / (c + td)^n`. Deployed today as a Google Cloud Function called by a Node API; the web interface lives in a separate repo: [CalculoIDF_front](https://github.com/AnaLauraFFerraz/CalculoIDF_front). The Node API is [CalculoIDF_API](https://github.com/AnaLauraFFerraz/CalculoIDF_API).

This repository is the project's reference repository (ADR 0002): it holds the Spec Kit setup, the specs, the ADRs and the knowledge base for all three repositories.

## Stack

- Python 3.10 in production (1st-gen Cloud Run functions runtime). pandas 2.0.1, numpy 1.24.3, scipy 1.10.1, scikit-learn 1.3.0, Flask (request object of the function), google-cloud-storage — pinned in `requirements.txt`, which also lists unused packages (D-11).
- Local environment with **uv**: `.venv` on Python 3.10. pandas 2.0.1 has wheels only up to Python 3.11.
- No `pyproject.toml`, no tests, no linter and no CI yet.

## Commands

```powershell
uv venv --python 3.10 .venv        # local environment matching production
uv pip install -r requirements.txt
```

There are no test, lint or run scripts yet; spec 001 adds the test suite.

## Architecture

Flow: `main.process_request` (HTTP handler: reads `csv_file_url`, downloads the CSV from Cloud Storage) → `main.main(csv_path)` → `process_data` → `outlier_test` → `distributions` → `k_coefficient` + `disaggregation_coef` → `ventechow` (fit and output JSON).

| Path | Responsibility |
|---|---|
| `src/main.py` | HTTP handler and pipeline orchestration; `load_data` reads the Hidroweb CSV (`;`, `,` decimal mark, ISO-8859-1, 12 header lines) |
| `src/gcs_utils.py` | Download and deletion of the CSV blob |
| `src/process_data.py` | Consisted/raw merge, water-year trimming (October–September), annual maxima, empty years, 10-year minimum |
| `src/outlier_test.py`, `src/grubbs_test.py` | Grubbs and Grubbs & Beck outlier removal, critical-value table |
| `src/distributions.py`, `src/yn_sigman.py` | Five distributions, R² selection, finite Gumbel tables |
| `src/k_coefficient.py`, `src/disaggregation_coef.py` | Frequency factors, disaggregation coefficients |
| `src/ventechow.py` | BFGS fit per duration range, relative error, Nash-Sutcliffe, regression, **assembly of the output JSON** |
| `src/csv/` | 13 real Hidroweb CSVs, the characterization inputs |
| `docs/adr/` | Architecture decision records (extended Nygard format) |
| `docs/knowledge/` | Domain and methodology, current system and debt IDs (`D-xx`), decisions log, roadmap (`C-xx`, `B-xx`), ENSO groundwork, Spec Kit guide, technical notes per feature (input for `/speckit-plan`) |

The JSON returned by `ventechow.main` (the "v0 contract", described in `docs/knowledge/02-current-system.md`) is consumed by the front-end. This repo owns that contract: any change to it must be flagged so the front-end can be updated, and it is a deliberate, specified change.

## Conventions

- **Language:** code, identifiers, comments, log messages, test names, documentation, file and folder names, and commit messages in **English**. User-facing text (front-end, error messages shown to users, reports) stays in **Brazilian Portuguese**. The existing user-facing strings — including the typo in `"Dados não são sufientes para completar a análise"` — are part of the v0 contract; never change them outside a spec. Hidroweb column names (`NivelConsistencia`, `Data`, `Maxima`, …) are the source format and keep their names.
- **Comments: minimal.** Only where the *why* is not obvious from the code. Do not narrate what the code does. Do not rewrite existing comments as a side effect of other work.
- **Numbers:** a change that alters any numeric output is never made "along the way". It is deliberate, justified and comes with updated golden files and a table of differences per station (constitution, principle I). Structural changes must keep the output identical.
- **Library upgrades** are separate changes, verified by the golden files. Upgrading numpy 1.x → 2.x changes the fitted parameters (see Known pitfalls).
- **Commits:** small, Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`), English subject, **no `Co-Authored-By` trailer**.
- **ADRs:** a `Proposed` ADR may be amended; an `Accepted` ADR is never edited — a new ADR supersedes it.

## Rules

- Never commit secrets: GCP or Firebase credentials, `.env`, API keys, tokens. `.env` and `.venv` are gitignored.
- Never modify, re-save or re-encode the CSVs in `src/csv/`: they are golden-file inputs (ISO-8859-1, excluded from line-ending conversion by `.gitattributes`).
- The `tcc-2023` tag and its immutable release preserve the code described in the thesis. Never move or delete them, and never delete or rename this repository, `CalculoIDF_front` or `CalculoIDF_API`.
- Never rewrite published git history.
- Do not change the deployed Cloud Function, the Vercel projects or the Render service without explicit approval.
- Documentation must describe only what exists in the code. Planned work is described as planned; no invented metrics or results.
- New work follows Spec-Driven Development: spec, plan and tasks are committed before the implementation.

## Spec-Driven Development

Spec Kit 1.2.0 (Claude Code integration, PowerShell scripts), initialized at the root of this repository (ADR 0002). The constitution is `.specify/memory/constitution.md`; read it before planning.

- Flow: `/speckit-specify` → `/speckit-clarify` → **approval** → `/speckit-plan` → **approval** → `/speckit-tasks` → `/speckit-analyze` → `/speckit-implement` → `/speckit-converge`.
- Approval is explicit: `**Status**: Approved (YYYY-MM-DD)` in the header of the spec and of the plan, in a separate `docs:` commit. Never implement a spec or plan that is not approved (principle VI).
- Each feature lives in `specs/NNN-short-name/` (`spec.md`, `plan.md`, `tasks.md`, …). Commit those artifacts (`docs:`) before any implementation commit.
- Pass the matching section of `docs/knowledge/07-technical-notes-by-feature.md` to `/speckit-plan`.
- This repo owns the API contract. A feature that changes the front-end is specified here, with a "Front-end impact" section.
- The project existed before SDD was adopted; only features from `specs/001` onward are built this way.

## Known pitfalls

- Importing `src/main.py` fails without GCP credentials: `gcs_utils.py` creates `storage.Client()` at import time.
- pandas 2.0.1 has no wheels for Python 3.12 or later; installing `requirements.txt` there tries to build from source and fails.
- The fit is ill-conditioned: BFGS ignores the declared `OPTIMIZATION_BOUNDS` (D-01). Upgrading numpy 1.x → 2.x alone changes `k, m, c, n` by up to ~18% while the fitted curve moves at most ~1.2%. Python 3.11/3.12, pandas 2.3.3 and scipy 1.15.3 with numpy 1.26.4 reproduce the output bit for bit (spike of 2026-10-27).
- Under pandas 3, `merged_df['Maxima'].fillna(0, inplace=True)` in `process_data.py` is a no-op (`ChainedAssignmentError`, D-13).
- The production runtime (Python 3.10) was deprecated on 2026-10-04 and will be decommissioned on 2027-04-04.
- The development machine has `core.autocrlf=true`; `.gitattributes` keeps the repository on LF regardless.

## How to work in this repo

- Talk to the maintainer in Brazilian Portuguese; write every artifact in English.
- Audit the relevant code and present a plan before changing files; wait for approval.
- One phase at a time; at the end, summarize what changed and what needs review.
- When a decision has a trade-off, present the options instead of deciding alone.
- If a request conflicts with what the code actually does, or with the constitution, say so before proceeding.
- Only commit as part of an approved plan or when asked; never push without explicit approval.
- Edit text files only with the Edit and Write tools, never with shell scripts.
