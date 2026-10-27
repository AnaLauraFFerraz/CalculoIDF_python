# 07 — Technical notes by feature (input for `/speckit-plan`)

> The specs say **what** and **why**. This document gathers the **how** already discussed or discovered while analyzing the code — the material to pass to `/speckit-plan` for each feature. Nothing here is a final decision: the plan is where it is evaluated and approved.

---

## 001 — Characterization tests

- **Import trap:** `gcs_utils.py` creates `storage.Client()` **when the module is imported**, and `main.py` imports `gcs_utils`. Importing `main` without GCP credentials fails. To avoid changing production code, a `conftest.py` can register a stub of `google.cloud.storage` in `sys.modules` before the import. Alternative (minimal, safe change): create the client lazily inside the functions.
- **Entry point under test:** `main.main(csv_path)` — returns a `dict` on success and a **JSON string** on business errors (D-04). The golden files must record both cases as they are. Two of the 13 CSVs (`…01848001_UB.csv`, `…02043032_MA.csv`) return the insufficient-data string.
- **Generation environment:** generate the golden files with the **current pinned versions** (`requirements.txt`: pandas 2.0.1, numpy 1.24.3, scipy 1.10.1) on **Python 3.10**, the runtime of the deployed function, **on Linux** (the CI runner; production also runs on Linux). Spike of 2026-10-27: Python 3.10 and 3.11 give bit-identical results with these pins. pandas 2.0.1 has wheels only up to Python 3.11.
- **Determinism:** BFGS with a fixed initial guess is deterministic in the same environment; set `OMP_NUM_THREADS=1`/`OPENBLAS_NUM_THREADS=1` to avoid BLAS variation. Record the Python, library versions and platform used in the golden-file metadata.
- **Comparison:** serialize with `json.dumps(sort_keys=True)`; compare recursively with `math.isclose` (configurable relative and absolute tolerances, **per field**); list order matters; readable diff by path (`parameters.parameters_1.k1: 512.3 → 512.4`). The `parameters` field is the least stable one: upgrading numpy 1.x → 2.x changes `k, m, c, n` by up to ~18% while `i_calculated` moves at most ~1.2% (D-01).
- **Deliberate update:** a `pytest --update-golden` flag or a `scripts/update_golden.py --reason "..."` script that rewrites the file and appends an entry to `tests/golden/CHANGELOG.md` (date, station, changed fields, reason).
- **Synthetic fixtures** in Hidroweb format (12 header lines, `;`, `,` decimal mark, ISO-8859-1) for: series starting in October and ending in September (D-02), no consisted data, < 10 years, 10–29 years, an empty year in the middle, and a month missing in both consistency levels (exposes the pandas 3 behavior of D-13).
- **CI:** GitHub Actions running `pytest` on every push.
- **Suggested structure:** `tests/fixtures/hidroweb/*.csv` (move or reference the 13 from `src/csv/`), `tests/golden/*.json`, `tests/test_characterization.py`.

## 002 — Backend unification (ADR 0001)

- **Steps, each verified by the golden files:**
  1. **Wrap** the current pipeline in FastAPI **with the 2023 pins**, on Python 3.11 (current Starlette and pydantic-settings require ≥ 3.11) → **exact** parity. Runs locally and in CI only.
  2. **(2a) Upgrade** to Python 3.12, pandas 2.3.3, numpy 1.26.4, scipy 1.15.3 → **exact** parity (shown in the spike of 2026-10-27). **First version deployed to Vercel** (its Python runtime starts at 3.12). Replacing scikit-learn with `numpy.polyfit` is its own commit, with a tolerance declared only for the regression (difference ~1e-14). Also, in their own commits, changes that do not alter numbers: `from scipy import stats` instead of the deprecated `scipy.stats.stats` namespace (`distributions.py`), `pprint` → logging (D-10).
  3. **Decommission** the Node API, Firebase Storage and the Cloud Function; archive `CalculoIDF_API`.
- **Out of scope (phase 2, with C-01):** numpy 2.x / pandas 3.x / current scipy — **changes numbers**. Known code points for that step: `process_data.py:67` (`fillna(inplace=True)` is a no-op under pandas 3, D-13), `process_data.py:66` and `:165` (`inplace` deprecated), `ventechow.py:223–224` (`SettingWithCopyWarning`), and scipy ≥ 1.18 requires Python ≥ 3.12.
- **Endpoint:** keep `POST /upload` returning **exactly the v0 contract** during the migration (the front-end only changes the URL). The versioned v1 contract comes in 004. Model v0 as a Pydantic `response_model` without changing the JSON (decision of 2026-10-27); the v0 error is a string with HTTP 200 (D-04), which may make full typing awkward — evaluate in the plan.
- **Upload without disk:** read the `UploadFile` in memory and pass `io.BytesIO` to `pd.read_csv` (`load_data` already accepts a file object) → eliminates the temporary file (D-08) and the name race (D-20) by construction.
- **2 MB limit:** Starlette does not limit by default → check `Content-Length` and read with a limit. Vercel caps request bodies at 4.5 MB.
- **CPU-bound:** declare the endpoint as `def` (not `async def`) so it runs in the threadpool.
- **Rate limit:** in-memory `slowapi` is per function instance, not global — acceptable for the load. Record the trade-off.
- **CORS:** `CORSMiddleware` with origins from an environment variable (Vercel domain of the front-end + localhost) — closes D-21.
- **Vercel deployment** (limits checked on 2026-10-27; recheck in the plan): Python runtime 3.12 (default), 3.13 or 3.14; entrypoint declared with `[tool.vercel] entrypoint = "module:app"` in `pyproject.toml`; dependencies from `pyproject.toml` + `uv.lock`; bundle up to 500 MB uncompressed (the dependencies measured ~243 MB); exclude `src/csv/`, tests and fixtures from the bundle (`excludeFiles` in `vercel.json`); Hobby: 2 GB / 1 vCPU, maximum duration 300 s; default region `iad1` (check whether a São Paulo region is available on Hobby). Separate Vercel project from the front-end's, linked to this repository. Health endpoint. Structured JSON logs with `request_id`.
- **Front-end switch:** the front-end's Vercel project uses Node.js 18, which blocks new builds; upgrade it to Node 22 (test `npm ci && npm run build` locally first) before changing `REACT_APP_BASE_URL`.
- **Before starting:** measure the current latency (p50/p95) of the 10-step flow to have a baseline (SC of the spec), and the cold start after the first deployment.

## 003 — Front-end modernization

- **Vite + React + TypeScript**; keep styled-components (v6), react-router, chart.js + react-chartjs-2 to limit the scope to parity. `react-katex` is unmaintained → use `katex` directly in a small component.
- `VITE_API_URL` environment variable (replaces `REACT_APP_BASE_URL`).
- **Contract types:** hand-written, mirroring v0, until 002 publishes OpenAPI; then generated with `openapi-typescript`.
- **Tests:** Vitest + Testing Library rendering `Report` with the **golden JSON from 001 as fixtures** (reuse: the same artifact protects back-end and front-end). One E2E smoke test (Playwright) of the upload → report flow.
- Fix along the way: D-32, D-33, D-35, D-36, D-38; remove spurious dependencies (D-34). Check D-14 (independently sorted chart series).
- Vercel: Vite preset.

## 004 — Validation and errors

- **Staged validation** (stops at the first error, with a specific code):
  1. *File:* size; ZIP detected by the `PK\x03\x04` bytes (Hidroweb delivers compressed downloads); encoding (ISO-8859-1, UTF-8 with BOM).
  2. *Structure:* metadata header; separator (`;` × `,` — **a file reopened and saved in Excel** is a common real case); required columns; a **streamflow** file (stage/discharge columns) sent by mistake.
  3. *Content:* `dd/mm/yyyy` dates; numeric `Maxima` ≥ 0; a single `EstacaoCodigo`; plausibility limit.
  4. *Sufficiency and quality:* ≥ 10 years; quality report.
- **Error format:** RFC 9457 *Problem Details* (`application/problem+json`) with `type`, `title`, `status`, `detail`, `code`, `request_id`, `errors[]`. Status: 413 (size), 415 (type), 422 (invalid content and insufficient data), 429 (limit), 500 (internal), 503 (dependency unavailable). `title`/`detail` are user-facing → Brazilian Portuguese.
- **Stable codes** (example): `ZIP_FILE`, `NOT_HIDROWEB_FORMAT`, `STREAMFLOW_FILE`, `INVALID_SEPARATOR`, `MISSING_COLUMNS`, `INVALID_DATES`, `MULTIPLE_STATIONS`, `INSUFFICIENT_DATA`, `SIZE_LIMIT`, `RATE_LIMIT`, `INTERNAL_ERROR`, `SERVICE_UNAVAILABLE`.
- **v1 contract** at `POST /v1/analyses`: response with `result` + `quality` + `provenance`; `dist` as a **code** (`gumbel_finite`) with a separate label; `empty_years` always a list; years as integers. `/upload` (v0) is removed once the front-end migrates.
- **Quality report:** years analyzed; years discarded and why; missing months per year (highlight October–March); **removed outliers** (year, value, test); consistency level used.
- **Front-end:** error component per code, with a suggested action and a help link; retry for 5xx/503; no `alert()`.
- Internal exceptions mapped to `INTERNAL_ERROR` without exposing a *stack trace*; fix D-05 (do not swallow exceptions).

## 005 — Integrated Hidroweb search

- **Sources to evaluate** (check documentation, terms of use and limits before the plan):
  - **Legacy SOAP WebService** `ServiceANA` at `telemetriaws1.ana.gov.br/serviceana.asmx` — historical series methods (station code, dates, data type 2 = rainfall, consistency level) and station inventory; XML. A public project that uses it reports **redundant/inconsistent** responses (overlapping series for the same station and month) → deduplication is mandatory.
  - ANA's **HidroWebService (REST)**, with registration/credentials — check current availability and rules.
- **Adapter pattern:** a `DataSource` interface with `CsvUpload` and `AnaApi`, both producing the **same normalized DataFrame** (`NivelConsistencia`, `Data`, `Maxima`) → pipeline untouched.
- **Equivalence test:** for the 13 fixture stations, the series obtained from the API over the **same period** as the CSV must produce the same result.
- **Station inventory** in a local cache, refreshed periodically (search by state/municipality/name; latitude/longitude for the B-13 map).
- **Resilience:** *timeouts*, *retries* with *backoff*, *circuit breaker*; with ANA down, the UI offers the upload.
- **Series cache** per station with an expiry (e.g., 7 days). This **reintroduces storage** — now with a real justification (cache persistence, not transport). Record it in an ADR: same component, different justification — a good architecture lesson.
- **Provenance** in the response: source, station code and name, municipality/state, query date/time, period.
- The Grubbs (n ≤ 147) and finite Gumbel (n ≤ 150) tables may now be exceeded (D-06) → handle it.

## 006 — IDF table and export

- **Table:** i for each Tr (2, 5, 10, 20, 30, 50, 75, 100) × each duration (5 min … 24 h), from the fitted equations (60 min exists in both ranges — decide which one to use or show both).
- **PDF:** server (HTML → PDF with WeasyPrint; charts with matplotlib for reproducibility) **×** client (print CSS / react-pdf). Quick MVP: print CSS; robust "calculation record" version: server. Decide in the plan.
- **Spreadsheet/CSV:** CSV on the client is trivial; `.xlsx` via openpyxl on the server or SheetJS on the client.
- **Copyable equation** in three formats: text, LaTeX and spreadsheet formula.
- **Reproducibility stamp** in the report: app version and method version (the same one referenced by the golden files), analysis date, data provenance.

## 007 — AI interpretation

- **Provider:** Anthropic API (Claude). Pick the model by cost/quality at implementation time (a smaller model tends to suffice for structured interpretation); check current names and prices.
- **Server-side call only** (the key never reaches the front-end). Input: the **structured JSON of the results** (no raw data); a *system prompt* with the domain glossary and the interpretation ranges (NS, relative error, warnings). Structured output in sections: fit quality · series reliability · warnings · how to use it in the project · limitations. The generated text is user-facing → Brazilian Portuguese.
- **Numerical guard (principle IV):** a post-generation validator that extracts the numbers from the text and checks that each one exists in the input; if not, reject and regenerate (or fall back to a standard text).
- **Cost:** on demand only (button), cache by the hash of the result, *rate limit*, token limit, *prompt caching* of the *system prompt*.
- **Evaluation:** an evaluation set with the 13 golden files and the expected points per case; a manual review rubric.
- No personal data is sent.

## 008 — ENSO analysis

- **Data:** `data/enso/roni.csv` and `oni.csv` copied from the CPC, with the retrieval date; an update script; a test ensuring coverage of every year in the fixtures. Years before 1950 → "unclassified".
- ***Off-by-one* trap:** `AnoHidrologico = Y` covers **Oct/Y – Sep/Y+1**. That year's DJF is **Dec/Y – Feb/Y+1**, which appears **in the Y+1 row** of the CPC table. E.g., water year 1997 → DJF 1997-98 (very strong El Niño). Write a test with this example.
- **Function** `classify_enso_years(annual_df, index="RONI")` → adds `phase`, `strength`, `index_value`.
- **Interaction with outliers (important):** the removed outliers may be precisely the extremes of El Niño years — the outlier test may be **erasing the signal**. Compute the ENSO analysis on the series **before** the removal (with and without outliers) and report the phase of the removed years.
- **Sensitivity:** ratio `mean(Pmax | El Niño) / mean(Pmax)` (and the same for La Niña); `scipy.stats.mannwhitneyu` El Niño × Neutral, two-sided; report p and n per group; below a minimum n per group → "insufficient sample".
- **Representativeness:** count per phase/strength in the series × climatological proportion since 1950; warning rule to be defined during clarification.
- **Front-end:** color by phase on the distribution chart points, legend, count per phase, sensitivity card, warning in the existing warnings block.
- **Contract:** an `enso` block in the v1 response.
