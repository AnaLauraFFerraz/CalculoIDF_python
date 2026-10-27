# CalculoIDF — Python

Computation core of CalculoIDF: it reads a rainfall series downloaded from Hidroweb (ANA) and determines the **Intensity-Duration-Frequency (IDF) equation** `i = k · Tr^m / (c + td)^n` for the station.

The project started as an undergraduate thesis in Civil Engineering (UFOP, 2023). The exact code described in the thesis is preserved in the [`tcc-2023` release](https://github.com/AnaLauraFFerraz/CalculoIDF_python/releases/tag/tcc-2023). The project is being modernized; see `docs/`.

## What it does

From the monthly maximum daily rainfall of one station (Hidroweb CSV), the pipeline:

1. prefers consisted data and fills gaps with raw data, then trims the series to complete October–September water years and takes the annual maxima;
2. removes outliers (Grubbs and Grubbs & Beck tests, 10%);
3. fits five probability distributions (log-normal, Pearson III, log-Pearson III, theoretical and finite Gumbel) and keeps the one with the highest R²;
4. computes rainfall for return periods of 2 to 100 years and disaggregates it into durations from 5 minutes to 24 hours;
5. fits the IDF equation separately for 5–60 min and 60–1440 min, and reports the mean relative error and the Nash-Sutcliffe efficiency.

Series with fewer than 10 water years are rejected. The full methodology is in [`docs/knowledge/01-domain-and-methodology.md`](docs/knowledge/01-domain-and-methodology.md).

## Current state

- Deployed as a Google Cloud Function (1st gen, Python 3.10), entry point `process_request` in `src/main.py`. The function receives a `gs://` URL to a CSV previously uploaded to Firebase Storage by the [Node API](https://github.com/AnaLauraFFerraz/CalculoIDF_API); the web interface is [CalculoIDF_front](https://github.com/AnaLauraFFerraz/CalculoIDF_front).
- Dependencies pinned in `requirements.txt` (pandas 2.0.1, numpy 1.24.3, scipy 1.10.1).
- 13 real Hidroweb CSVs in `src/csv/`.
- No automated tests yet.

The planned changes (characterization tests, then a single FastAPI service replacing the Node API and the Cloud Function) are described in [`docs/adr/`](docs/adr/) and [`docs/knowledge/04-roadmap.md`](docs/knowledge/04-roadmap.md).

## Running locally

Requires [uv](https://docs.astral.sh/uv/).

```powershell
uv venv --python 3.10 .venv
uv pip install -r requirements.txt
```

Importing `src/main.py` creates a Google Cloud Storage client and fails without GCP credentials.

## Structure

| Path | Content |
|---|---|
| `src/main.py` | HTTP handler and pipeline orchestration |
| `src/process_data.py` | Data preparation and annual maxima per water year |
| `src/outlier_test.py`, `src/grubbs_test.py` | Outlier tests and Grubbs critical values |
| `src/distributions.py`, `src/yn_sigman.py` | Distributions, R² and finite Gumbel tables |
| `src/k_coefficient.py`, `src/disaggregation_coef.py` | Frequency factors and disaggregation coefficients |
| `src/ventechow.py` | IDF fitting, error metrics and output JSON |
| `src/gcs_utils.py` | Download and deletion of the CSV in Cloud Storage |
| `docs/adr/` | Architecture decision records |
| `docs/knowledge/` | Domain, current system, decisions, roadmap and technical notes |

## License

No license has been chosen yet.
