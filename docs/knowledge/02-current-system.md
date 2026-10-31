# 02 — Current system (as-is, 2023 code)

> Snapshot of the code as it stood in October 2026 (last real commit: 2023-08-29; tag `tcc-2023`). It serves as the "spec of the existing system" — the brownfield starting point for every feature. The `D-xx` IDs are referenced by the specs and the roadmap.

## 1. Repositories and deployment

Local folder: `C:\CalculoIDF\`. Repositories on GitHub (`AnaLauraFFerraz/CalculoIDF_front`, `CalculoIDF_API`, `CalculoIDF_python`). The state described in the thesis is preserved in each repository by the `tcc-2023` tag and an immutable release.

| Repo | Stack | Hosting | Configuration |
|---|---|---|---|
| `CalculoIDF_front` | React 18.2 + Create React App (react-scripts 5.0.1), styled-components 5, react-router-dom 6, axios, chart.js 4 + react-chartjs-2 5, katex + react-katex, react-spinners | Vercel (Hobby plan), Git integration: a push to `main` deploys to production; project configured with Node.js 18, which Vercel has discontinued and which blocks new builds | `REACT_APP_BASE_URL` |
| `CalculoIDF_API` | Node + Express 4 (CommonJS `src/app.cjs`), multer, cors, express-rate-limit 6, firebase-admin 11, axios, dotenv | Render | `PORT` (default 5000), `FUNCTION_URL`, `FIREBASE_STORAGE_BUCKET_URL`, `FIREBASE_CREDENTIALS` (service account JSON in base64) |
| `CalculoIDF_python` | Python, Flask (HTTP handler), pandas 2.0.1, numpy 1.24.3, scipy 1.10.1, scikit-learn 1.3.0, google-cloud-storage | Cloud Run functions, 1st gen (entry point `process_request`), Python 3.10 runtime — deprecated on 2026-10-04, decommissioned on 2027-04-04; deployed manually (no deployment pipeline; CI only runs the characterization suite of spec 001) | default GCP credentials |

**Deployed code × repository (checked on 2026-10-27).** The deployed function (version 48, deployed on 2023-09-01) differs from the `tcc-2023` tag only in a line break (`main.py`), two comments (`process_data.py`), a commented-out `# bounds=OPTIMIZATION_BOUNDS` and the final `pprint` (`ventechow.py`), and three commented-out lines in its own `requirements.txt`. **No difference affects the numbers.**

## 2. Request flow (10 steps — Figure 8 of the thesis)

1. The user picks the CSV in the front-end.
2. Front-end → `POST {BASE_URL}/upload` (`multipart/form-data`, field `file`).
3. The API writes the file to Firebase Storage **under the original file name**.
4. The API builds `gs://…/<name>`.
5. API → `POST FUNCTION_URL` with `{ "csv_file_url": "<gs://…>" }`.
6. The Cloud Function downloads the blob to `/tmp/<name>`.
7. The Python pipeline runs and returns JSON.
8. The function responds to the API.
9. The API relays the JSON to the front-end.
10. The API deletes the blob (`finally` block). The function also tries to delete the local file and the blob — **on the success path only**.

Limits at the edge (API): body and file up to **2 MB**; *rate limit* of **100 requests / 15 min per IP**; open CORS.

## 3. Python backend — modules (`CalculoIDF_python/src/`)

| File | Responsibility |
|---|---|
| `main.py` | `process_request` (HTTP handler: reads `csv_file_url`, downloads, calls `main`, cleans up) and `main(csv_path)` (orchestrates the pipeline). `load_data`: `sep=";"`, `encoding="ISO 8859-1"`, `skiprows=12`, `decimal=","`, `usecols=[NivelConsistencia, Data, Maxima]` |
| `gcs_utils.py` | download and deletion of the blob in GCS; creates `storage.Client()` at import time |
| `process_data.py` | dates, consisted/raw split, monthly *resample* (`MS`, `.first()`), merge and fill, trimming to the water-year cycle, annual maximum, empty years, 10-year rule |
| `grubbs_test.py` | Grubbs critical values at 10% (n = 3…147) |
| `outlier_test.py` | statistics, critical values (Grubbs and Grubbs & Beck), outlier removal |
| `yn_sigman.py` | yn and σn tables for the finite Gumbel (n = 1…150) |
| `distributions.py` | Weibull, parameters, the 5 distributions, R² and choice of the best one |
| `k_coefficient.py` | K factor for Tr = 2, 5, 10, 20, 30, 50, 75, 100 according to the chosen distribution |
| `disaggregation_coef.py` | durations (h) and disaggregation coefficients |
| `ventechow.py` | intensities, reshaping to long format, condition (range 1 or 2), BFGS optimization, recomputation, relative errors, NS, regression (sklearn), **assembly of the output JSON** |

**Existing fixtures:** 13 real CSVs in `CalculoIDF_python/src/csv/`:
`chuvas_C_01844000_CV.csv`, `chuvas_C_01844000_CV1.csv`, `chuvas_C_01848001_UB.csv`, `chuvas_C_01943011.csv`, `chuvas_C_01944009_PL.csv`, `chuvas_C_02040003_FFL.csv`, `chuvas_C_02040022.csv`, `chuvas_C_02043011_FAZ_P.csv`, `chuvas_C_02043032_MA.csv`, `chuvas_C_02243077_HF_RIO.csv`, `chuvas_C_02243155.csv`, `chuvas_C_02346117.csv`, `chuvas_C_02346328.csv`.
Two of them (`…01848001_UB.csv` and `…02043032_MA.csv`) end in the insufficient-data path (< 10 years), as shown by a run of the `tcc-2023` code on 2026-10-27; the other 11 produce a full result.

## 4. Current response contract (implicit, "v0")

Success — object assembled by hand at the end of `ventechow.main`:

```json
{
  "graph_data": { "F": [..%..], "P_max": [...], "P_dist": [...] },
  "intensity_graph_data_1": { "i_real": [...], "i_calculated": [...], "regression": { "slope": 0.0, "intercept": 0.0 } },
  "intensity_graph_data_2": { "...": "same, 60–1440 min range" },
  "parameters": {
    "parameters_1": { "k1": 0, "m1": 0, "c1": 0, "n1": 0 },
    "parameters_2": { "k2": 0, "m2": 0, "c2": 0, "n2": 0 }
  },
  "mean_relative_errors": { "interval_1": 0.0, "interval_2": 0.0 },
  "sample_size_above_30_years": true,
  "empty_consistent_data": false,
  "year_range": { "first_year": "1975", "last_year": "2020" },
  "dist": "Gumbel Finita",
  "ns": { "parameter_1": 0.0, "parameter_2": 0.0 },
  "empty_years": false
}
```

- `dist` comes as a Portuguese **display name** ("log-normal", "Pearson tipo III", "log-Pearson tipo III", "Gumbel Teórica", "Gumbel Finita") — presentation mixed with data.
- `empty_years` is `false` **or** a list of years (variant type).
- `year_range` holds years as strings.

"Business" error — a **JSON string with HTTP 200**: `"Erro ao carregar o arquivo"` (*error loading the file*) or `"Dados não são sufientes para completar a análise"` (*insufficient data*, with a typo). The front-end tells them apart with `typeof data === "string"`. These strings are part of the v0 contract.

Technical errors — `{"error": "..."}` with 400/500 from the function; the API turns them into `{"message": "..."}` 500.

## 5. Front-end (`CalculoIDF_front/src/`)

- Routes: `/` (`pages/Main.js`) and `/sobre` (`pages/About.js`); `components/Header.js` with an "About" link.
- `Main.js`: instructions (link to Hidroweb; October–September water-year notice), `Input` (upload + spinner) and `Report`.
- `Report.js`: no data → "Nenhum dado disponível"; string → shows the message; object → warnings (< 30 years; empty years), distribution chart (`IdfGraph`), "Considerations" (period, distribution used, use of raw data), two observed × calculated charts (`IntensityGraphs`), equation in KaTeX, table of k, m, c, n parameters per range, table of mean relative error and NS, legend of the NS criteria.
- Errors shown with `alert()`.
- All user-facing text is in Brazilian Portuguese.

## 6. Technical debt and issues found

Severity legend: **H** = may produce a wrong result or a failure for the user; **M** = robustness, security or maintenance; **L** = hygiene. IDs are grouped by area (0x Python, 2x API, 3x front-end, 4x cross-cutting) and are intentionally not contiguous.

### Python backend
| ID | Sev. | Where | Issue |
|---|---|---|---|
| D-01 | H | `ventechow.py` | `OPTIMIZATION_BOUNDS` is declared but **BFGS does not accept bounds** — the parameters are unconstrained (the deployed code has the `bounds=` argument commented out). Fixing it (e.g., L-BFGS-B) **changes the numbers** → deliberate golden-file update. The unconstrained fit is also ill-conditioned: upgrading numpy 1.x → 2.x alone changes `k, m, c, n` by up to ~18% (see `03-architecture-and-decisions.md`). |
| D-02 | H | `process_data.remove_out_of_cycle_data` | **Confirmed bug (synthetic-data test, 2026-10-20):** if the series starts exactly in **October**, the code drops everything up to the first September — one **complete** water year is lost; if it ends exactly in **September**, it drops from the last October to the end — another year lost. Example: a series with exactly **10 complete water years** (Oct/1990–Sep/2000) ends up with **8** and falls below the 10-year minimum → "insufficient data". Jan/1990–Dec/2000 correctly keeps the 10. Since consisted series may close in water-year blocks, the end case may be frequent. |
| D-03 | H | `process_data` | Gaps become 0 (`fillna(0)`): a year missing rainy months looks complete, with an underestimated maximum. There is no per-year completeness check. |
| D-04 | M | `main.py` | Business errors returned as a **string with HTTP 200**; message with a typo ("sufientes"). |
| D-05 | M | `outlier_test.main` | Catches **any** exception and returns an empty DataFrame → fails later with a `KeyError` → generic 500. |
| D-06 | M | tables | Grubbs covers n = 3…147 and finite Gumbel n = 1…150; outside that → `KeyError` → 500. Rare with a CSV, possible with an automated search on any station. |
| D-07 | H? | `outlier_test.remove_outliers` | Mean, standard deviation and bounds are computed **once** and not recomputed after each removal — differs from the classic iterative Grubbs test (which recomputes with the new n). Validate against the literature/advisors before changing. |
| D-08 | M | `main.process_request` | On exception, the file in `/tmp` and the blob are **not deleted**; the API also deletes in `finally` → double deletion, one of them fails. |
| D-09 | L | `distributions.params_calculation` | `g = 0` → `α = 4/g²` = division by zero (edge case). |
| D-10 | L | `ventechow.py` | `pprint` of the whole output (log noise); **scikit-learn used for a single linear regression** (`numpy.polyfit` does it). |
| D-11 | L | `requirements.txt` | Unused/legacy dependencies (`beautifulsoup4`, `google`, `google-cloud`…) and 2023 versions. |
| D-13 | M | `ventechow.main`, `process_data` | Chained-assignment patterns (`df_slice[...] = ...`, `fillna(inplace=True)` on a column) that change behavior under **pandas 3 Copy-on-Write**. Confirmed on 2026-10-27: under pandas 3, `merged_df['Maxima'].fillna(0, inplace=True)` (`process_data.py:67`) does nothing (`ChainedAssignmentError`); no effect on the 13 CSVs, but a month missing in both levels would keep a NaN. |
| D-14 | M? | `ventechow.main` | In the output, `i_real` and `i_calculated` are each sorted **independently** (`sort_values`), so the pairs shown in the observed × calculated charts no longer refer to the same (Tr, td). The regression is computed before the sorting, so the numbers are right; the issue is presentation. To confirm in the front-end. Fixing it changes the JSON → deliberate change. |

### Node API
| ID | Sev. | Where | Issue |
|---|---|---|---|
| D-20 | H | `app.cjs` | The blob uses `req.file.originalname`. Hidroweb names files after the station code (`chuvas_C_01944009.csv`): **two users with the same station at the same time overwrite each other's file**, and whoever finishes first deletes the other's. Race condition. |
| D-21 | M | `app.cjs` | `cors()` open to any origin. |
| D-22 | M | `app.cjs`/`firebase.cjs` | `deleteFile` rethrows inside the `finally` of an async handler → unhandled rejection. |
| D-24 | L | `package.json` | Spurious dependencies: `fs` (placeholder package), `pandas` (unrelated npm package), `firebase` (unused client SDK), `nodemon` in production dependencies; unused `uploadFile`. |
| D-25 | L | `firebase.cjs` | Bucket hard-coded (`calculoidf.appspot.com`) while the URL comes from an environment variable. |

### Front-end
| ID | Sev. | Where | Issue |
|---|---|---|---|
| D-30 | M | `Main.js` | The `status === 400/500` branches inside the `try` are **unreachable** (axios throws for non-2xx) → every failure falls into the `catch` with a generic message; everything via `alert()`. |
| D-32 | L | `Main.js`, `About.js` | `height: 100vw` (width used as height) — layout bug. |
| D-33 | L | `Report.js` | `console.log` of the data in production. |
| D-34 | M | `package.json` | CRA is deprecated; accidental dependencies `npm` and `i` (an `npm i npm`), unnecessary `dotenv`; testing-library installed with no tests. |
| D-35 | L | `Input.js` | Leftover `action`/`method`/`enctype` attributes on the form. |
| D-36 | L | `About.js` | The text says the IDF is fitted "to models such as the Gumbel equation" — Gumbel is a distribution, and the app evaluates five. |
| D-37 | M | `Report.js` | **Removed outliers are not shown**; "period of N years" counts years that may have been removed as outliers. Transparency gap. |
| D-38 | L | `IdfGraph.js` | `ChartJS.register` on every render. |

### Cross-cutting
| ID | Sev. | Issue |
|---|---|---|
| D-40 | H | **No automated tests** in `CalculoIDF_API` and `CalculoIDF_front`. `CalculoIDF_python` has the characterization suite of spec 001 (golden files for 13 real and 8 synthetic series, plus unit tests of the suite's own comparator and regeneration script), run by GitHub Actions on every push and required on `main`; the modules in `src/` have no unit tests. |
| D-41 | L | All versioned files showed up as modified **only by line endings**. Addressed in `CalculoIDF_python` by `.gitattributes` (2026-10-27); the other repositories still need it. |
| D-42 | L | READMEs mentioned an MIT license and a `LICENSE` file that does not exist; the Python README described functions that do not exist ("monthly mean"). The Python README was rewritten on 2026-10-27; the other READMEs still need it. |

## 7. What is good and should be preserved

- The computation core is **modular and readable**, with docstrings — which is why the migration can change its wrapper without touching it.
- The quality rule "consisted first, raw only to fill gaps within the consisted period" is conservative and correct for the target audience.
- The user warnings (< 30 years, empty years, use of raw data) already exist — the **transparency** pattern is established and only needs to be extended.
- Commit messages already follow Conventional Commits (`feat:`, `fix:`).
