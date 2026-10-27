# 05 — ENSO / El Niño: groundwork for the climate features

> Research from September–October 2026 on how to connect CalculoIDF to El Niño. Basis for spec 008 and for backlog items B-10, B-11 and B-12.

## 1. Current situation (checked on 2026-10-20)

**NOAA CPC — ENSO Diagnostic Discussion of 2026-10-08:**
- Status: **El Niño Advisory** (in effect since 2026-08-13).
- Monthly Niño-3.4: **+2.1 °C** (it was +1.4 °C in July). Niño-3: +3.0 °C; Niño-1+2: +3.9 °C.
- Chance of a **historic** event (RONI ≥ +2.5 °C, above any El Niño since 1950): 54% in SON, **83% in OND**, 70% in NDJ.
- A **strong to very strong** El Niño is likely through **Jan–Mar 2027** (> 83%).
- The CPC now uses the **RONI** (Relative Oceanic Niño Index) as its strength reference.

**INMET (Brazil's National Meteorology Institute, 2026 bulletins):** > 90% probability of a very strong El Niño (anomalies > 2.0 °C) in spring and summer; persisting until at least early 2027; **above-average rainfall in the South**, **below average in the Center-North**; risk of heat waves and fires. Monthly updates.

> Update this section with each monthly bulletin while the feature is under development.

## 2. Why this matters for the method (the central idea)

Ven Te Chow's frequency analysis assumes the annual maxima are **independent and identically distributed**: the series is **stationary** and each year is a realization of the same process. But the ENSO phase is a known variable that **conditions** rainfall in South America. If that holds for a station, the annual maxima do not come from a single distribution — they come from a **mixture of regimes**.

In other words, the feature does not "decorate the app with climate": it **tests the validity of the assumption** on which the whole IDF curve is built. It is the natural, tractable bridge to the broader discussion of non-stationarity and climate change — ENSO has an index documented since 1950, while detecting a trend requires longer series.

**Methodological fit:** the ONI/RONI peaks in **DJF**, a quarter that falls **inside** the October–September water year. Classifying each water year by ENSO phase is clean, with no attribution ambiguity. The rainy season in Southeast Brazil (October–March) is also contained in it.

## 3. The six feature ideas (from the cheapest/most robust to the most ambitious)

| # | Feature | Cost | Statistical robustness | Destination |
|---|---|---|---|---|
| 1 | **Classify each water year by ENSO phase** (and strength); chart points colored by phase | Low | High (it is just labeling) | 008 – P1 |
| 2 | **ENSO representativeness warning for the series** (e.g., "a 28-year series with only 4 moderate/strong El Niño years") — fits the existing warning system | Very low | High | 008 – P1 |
| 3 | **Station ENSO sensitivity index**: ratio between the mean of the maxima in El Niño years and the overall mean + **Mann-Whitney U** test (non-parametric, good for small, skewed samples) | Low | Medium — best cost-benefit | 008 – P2 |
| 4 | **IDF curves conditioned on phase** (El Niño years only) | Medium | **Low**: 30 years ≈ 8 El Niño years; requires a minimum-sample guard and a "exploratory, do not use for design" label | B-10 |
| 5 | **Non-stationary GEV** with the index as a covariate: `μ(t) = μ₀ + μ₁·ONI(t)`, using **all** years; compare with the stationary model (likelihood-ratio test) | High | **High** — the correct answer to the sample problem of item 4 | B-11 (converges with B-05) |
| 6 | **Season context panel**: current ENSO state next to the station's history ("entering a very strong El Niño; here, strong El Niño years had maxima X% above average") | Medium | Depends on 3 | B-12 (with 007) |

Recommended core: **2 + 1 + 3**. Scientific extension (publishable): **5**. Season context: **6**, once the AI feature is ready.

## 4. Honest caveats (they change the expected outcome)

1. **The ENSO signal in Southeast Brazil is weak and ambiguous.** The teleconnection is strong in the South (above-average rainfall) and in the North/Northeast (drought); the Southeast is a transition zone. The thesis stations are in Minas Gerais: the result is quite likely to be "**no detectable signal at this station**" — and the app must say so plainly, because it is a valid result, not a failure.
2. **ENSO conditions seasonal totals much more consistently than daily extremes.** The app's variable is the annual maximum daily rainfall, later disaggregated with fixed coefficients. The path ENSO → daily maximum is noisier. Moreover, there is no reason to assume the **disaggregation coefficients are invariant to the ENSO phase** — and this cannot be tested without recording-gauge data. A limitation to declare, not a blocker.
3. **Statistical power.** Slicing the series destroys sample size; that is why item 5 is superior to items 3 and 4.
4. **The outlier test may be erasing the signal.** The maxima removed as outliers by Grubbs / Grubbs & Beck may be precisely the extremes of El Niño years. The ENSO analysis must look at the series **before** the removal (and show the phase of the removed years) — otherwise the feature measures the effect of ENSO on a series from which the strongest effects were removed.
5. **ONI × RONI.** The ONI measures the absolute SST anomaly in Niño-3.4; with the background warming of the oceans, recent years tend to be classified as "more El Niño" than they would be in relative terms. The RONI discounts the mean tropical anomaly — and it is what the CPC uses today. Recommendation: **RONI as default**, ONI as a documented alternative. (Open decision in spec 008.)

## 5. Phase classification (convention to fix in the spec)

- Episode by the CPC criterion: index ≥ +0.5 °C (El Niño) or ≤ −0.5 °C (La Niña) for **5 consecutive overlapping seasons**.
- Assignment to the water year: by the **DJF** value contained in it (alternative: the episode peak).
- Strength (usual convention, not official CPC): weak 0.5–0.9; moderate 1.0–1.4; strong 1.5–1.9; very strong ≥ 2.0.
- Data: a static historical table (small, stable) **versioned in the repository**, with its retrieval date — not an API dependency at computation time.

## 6. Sources

- NOAA CPC — ENSO Diagnostic Discussion: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml
- NOAA CPC — strength probabilities (RONI): https://cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/strengths.php
- NOAA CPC — historical ONI table: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
- INMET — El Niño 2026: monitoring, forecasts and impacts (in Portuguese): https://portal.inmet.gov.br/noticias/el-ni%C3%B1o-2026-saiba-detalhes-sobre-o-monitoramento-previs%C3%B5es-e-os-poss%C3%ADveis-impactos-do-fen%C3%B4meno-no-brasil
- INMET — El Niño 2026-2027 panel (in Portuguese): https://portal.inmet.gov.br/uploads/notastecnicas/Painel-El-Ni%C3%B1o-3-Edi%C3%A7%C3%A3o.pdf
