# 01 — Domain and methodology (what CalculoIDF computes, and how)

> Knowledge base of the CalculoIDF project. This document describes the **domain** and the **methodology** implemented in the undergraduate thesis. For the implementation (files, functions, technical debt), see `02-current-system.md`.

## 1. Project origin

- **Work:** "Aplicação web para a determinação da equação intensidade-duração-frequência" (*A web application for determining the intensity-duration-frequency equation*) — undergraduate thesis in Civil Engineering, Escola de Minas, Federal University of Ouro Preto (UFOP), 2023 (full citation in the `tcc-2023` releases of the repositories).
- **Advisor:** Prof. D.Sc. Walliston dos Santos Fernandes. **Co-advisor:** Prof. M.Sc. Maria Luíza Teófilo Gandini.
- **Methodological basis:** the step-by-step procedure for obtaining the IDF equation follows GANDINI (2016).
- **Stated goal:** automate the computation of the IDF equation and make it accessible to professionals **without training in statistical hydrology**, while keeping it accurate and reliable.

This goal is the product's "why" and should guide UX decisions: the typical user is an engineer who needs an IDF equation to size drainage works, not a statistician.

## 2. Domain concepts

**IDF (Intensity-Duration-Frequency) curve / equation:** the relationship between rainfall intensity (i), event duration (td) and frequency, expressed as the return period (Tr). Used in urban drainage, flood control, transportation works (bridges, culverts), water resources management and erosion prevention.

**Equation form adopted:**

```
i = k · Tr^m / (c + td)^n
```

- `i` — intensity (mm/h)
- `Tr` — return period (years)
- `td` — duration (minutes)
- `k, m, c, n` — parameters fitted per station (and, in this project, per duration range)

**Return period (Tr):** the average time between events that equal or exceed a given magnitude. Design reference: 2 to 10 years for minor drainage, 50 to 100 for major drainage (CETESB/DAEE, 1986).

**Rain gauge × recording rain gauge:** rain gauges (*pluviômetros*) record the daily total; recording gauges (*pluviógrafos*) record rainfall continuously. The project works with **daily data** (rain gauge) and reaches sub-daily durations through **disaggregation coefficients** — an important premise and a known limitation.

**Water year (hydrological year):** in Southeast Brazil, from **October to September**, so that it starts and ends in the dry season. The analysis uses complete water years only. The app warns that the computation is only valid for regions where the water year is October–September.

**Raw × consisted data (ANA/Hidroweb convention):** `NivelConsistencia = 1` (raw, untreated) and `2` (consisted: checked, gaps corrected, interpolated). ANA recommends at least 10 years without gaps for hydrometeorological indicators.

**Data source:** Hidroweb / SNIRH (ANA, Brazil's National Water and Sanitation Agency) — `https://www.snirh.gov.br/hidroweb/serieshistoricas`. The downloaded CSV has a metadata header (12 lines), `;` as separator, `,` as decimal mark and ISO-8859-1 encoding. Each row is one **month** of one station with, among other columns, `EstacaoCodigo`, `NivelConsistencia`, `Data`, `TipoMedicaoChuvas`, `Maxima` (maximum daily rainfall in the month), `Total`, `DiaMaxima`, `NumDiasDeChuva`. Column names are kept in Portuguese because they are the source format.

## 3. Methodological pipeline (as implemented)

### 3.1 Data preparation
1. Read `NivelConsistencia`, `Data`, `Maxima`.
2. Prefer **consisted** data; fill gaps (NaN or 0) with the corresponding **raw** values. Values before the first and after the last consisted date are **not** filled with raw data (a quality choice). No consisted data → use raw data and warn.
3. Gap in both levels → rainfall taken as **0** (limitation: may underestimate the year's maximum).
4. Trim to complete water years (October–September); compute the **annual maximum** (`Pmax_anual`) per water year.
5. Remove empty years at both ends; years with a maximum of 0 in the middle of the series are excluded and **reported** to the user.
6. **Fewer than 10 years** → analysis stops ("Dados não são suficientes para completar a análise" — *insufficient data to complete the analysis*). **10 to 29 years** → analysis runs with a quality warning.

### 3.2 Outlier detection
Two tests; a value is only an outlier **if both tests flag it** (for the upper bound):

- **Grubbs (1969):** `Tn = (xn − x̄) / s`; outlier if `Tn > Tcritical` (tabulated; **10% significance level**, n from 3 to 147).
- **Grubbs & Beck (1972):** bounds on a natural-log scale
  - `xh = exp(ln(p̄) + kn · ln(s))`, `xl = exp(ln(p̄) − kn · ln(s))`
  - `kn = −3.62201 + 6.28446·n^0.25 − 2.49835·n^0.5 + 0.491436·n^0.75 − 0.037911·n` (10%)

Iterative removal until no outliers remain. *(See `02-current-system.md`, D-07: the statistics are not recomputed after each removal — a possible methodological divergence to validate.)*

### 3.3 Probability distributions (estimating extreme events)
Exceedance frequency by **Weibull**: `F = m / (N + 1)` (m = rank in the series sorted in descending order). Non-exceedance = `1 − F`.

Parameters: mean and standard deviation of the values and of their log10; skewness `g` and `gw`; `α = 4/g²`, `αw = 4/gw²`; `yn` and `σn` (finite Gumbel, tabulated up to n = 150).

| Distribution | Factor / variable | Rainfall |
|---|---|---|
| Log-normal | `KN = Φ⁻¹(1 − F)`; `WTr = x̄log + KN·σlog` | `P = 10^WTr` |
| Pearson III | `YTr = Γ⁻¹(F or 1−F, α, β=1)` depending on the sign of g; `KP = (g/2)(YTr − α)` | `P = x̄ + KP·σ` |
| Log-Pearson III | same with `gw, αw`; `KLP = (gw/2)(YTrw − αw)` | `P = 10^(x̄log + KLP·σlog)` |
| Theoretical Gumbel | `y = −ln(−ln(1−F))`; `KGT = 0.7797·y − 0.45` | `P = x̄ + KGT·σ` |
| Finite Gumbel | `KGF = (y − yn)/σn` | `P = x̄ + KGF·σ` |

**Selection:** the distribution with the **highest R²** (square of the Pearson correlation between observed and theoretical values). The thesis itself lists replacing this with **goodness-of-fit tests** (Kolmogorov-Smirnov, Anderson-Darling) as future work.

### 3.4 "Observed" intensities (Ven Te Chow + disaggregation)
1. For Tr ∈ {2, 5, 10, 20, 30, 50, 75, 100} years, compute K for the chosen distribution.
2. 1-day rainfall: `xTr = x̄ + K·s` (or on a log scale for log-normal/log-Pearson).
3. Disaggregate with fixed coefficients (CETESB, 1980; Occhipinti & Santos, 1966):

| Ratio | Coef. | Ratio | Coef. |
|---|---|---|---|
| 1 day / 24 h | 1.14 | 24 h / 60 min | 0.42 |
| 24 h / 12 h | 0.85 | 24 h / 30 min | 0.311 |
| 24 h / 10 h | 0.82 | 24 h / 25 min | 0.283 |
| 24 h / 8 h | 0.78 | 24 h / 20 min | 0.252 |
| 24 h / 6 h | 0.72 | 24 h / 15 min | 0.218 |
| 24 h / 4 h | 0.63 | 24 h / 10 min | 0.168 |
| 24 h / 2 h | 0.52 | 24 h / 5 min | 0.106 |

4. Intensity = rainfall for the duration ÷ duration (mm/h). These are the "observed intensities" (`i_real`) used in the fit — note that they are already derived from a model, not sub-daily measurements.

### 3.5 Fitting the IDF equation
- **Two sets** fitted separately: **5 ≤ td ≤ 60 min** and **60 ≤ td ≤ 1440 min** (60 min belongs to both).
- Initial guess: `k = 500, m = 0.1, c = 10, n = 0.7`.
- Intended bounds: `k ∈ [100, 2000]`, `m ∈ [0, 3]`, `c ∈ [0, 100]`, `n ∈ [0, 10]` *(declared but not applied — see D-01)*.
- Optimizer: `scipy.optimize.minimize`, **BFGS** method; objective function = **sum of relative errors** between `i_calculated` and `i_real`.

### 3.6 Goodness of fit
- **Relative error:** `|i_calc − i_real| / i_real × 100`. Mean < 5% excellent; 5–10% adequate; > 10% indicates a problem in the procedure.
- **Nash-Sutcliffe:** `NS = 1 − Σ(i_real − i_calc)² / Σ(i_real − ī_real)²`. NS > 0.75 very good; > 0.65 good; > 0.5 satisfactory; ≤ 0.5 unsatisfactory.
- Linear regression `i_real × i_calc` (slope and intercept) for the charts.

## 4. Known limitations and future work (from the thesis itself, section 5.1)

1. Criterion for choosing between the raw and the consisted series.
2. Specific support for raw data.
3. Other outlier tests.
4. Other distributions (e.g., **GEV**).
5. Other optimization techniques.
6. **Goodness-of-fit tests** instead of R².
7. **Water year** configurable per region of Brazil.
8. Handling of **non-continuous data** (gaps).

Additional limitations identified in the September–October 2026 review: fixed disaggregation coefficients (not regionalized, and not testable without recording-gauge data); gaps treated as zero; **stationarity** assumption for the series (see `05-enso-el-nino.md`).

## 5. Quick glossary

| Term | Meaning |
|---|---|
| Pmax_anual | Annual maximum daily rainfall per water year |
| F | Exceedance probability (Weibull) |
| K | Frequency factor (Ven Te Chow) |
| Tr | Return period (years) |
| td | Rainfall duration (min) |
| i_real | "Observed" intensity (derived from the distribution + disaggregation) |
| i_calculated | Intensity from the fitted IDF equation |
| NS | Nash-Sutcliffe efficiency |
| ENSO / ONI / RONI | El Niño–Southern Oscillation / Oceanic Niño Index / Relative ONI |

## 6. Core references

- CHOW, V. T.; MAIDMENT, D. R.; MAYS, L. W. *Applied Hydrology*. McGraw-Hill, 1988.
- CETESB. *Drenagem Urbana: manual de projeto*. 1980 (and CETESB/DAEE, 2nd ed., 1986).
- OCCHIPINTI, A. G.; SANTOS, P. M. 1966 (disaggregation coefficients).
- GRUBBS, F. E. 1969; GRUBBS, F. E.; BECK, G. 1972.
- NAGHETTINI, M.; PINTO, E. J. A. *Hidrologia Estatística*. CPRM, 2007.
- GANDINI, M. L. T. 2016 (step-by-step methodology adopted).
- ANA. Hidroweb / SNIRH; *Cartilha de Estatísticas de Dados Fluviométricos e Pluviométricos*.
