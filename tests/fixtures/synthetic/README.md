# Synthetic series S1–S8

Edge cases of spec 001 (FR-007, research R-06) in Hidroweb format, for the characterization
suite. Their golden files are `tests/golden/synthetic/<id>.json`.

- **Synthetic, not ANA data.** Generated once by `scripts/make_synthetic_fixtures.py` (fixed
  seed) and committed; the committed bytes are the fixtures. Do not regenerate, edit or re-save
  them: a change alters their SHA-256 and fails the suite until the golden files are
  regenerated. `.gitattributes` (`*.csv -text`) keeps the bytes as written.
- **Station codes** `9999001`–`9999008` are arbitrary and identify no real station.
- **Format** copied from the real exports in `src/csv/`: ISO-8859-1, CRLF, 12 lines before the
  column header (the first three state that the file is synthetic), `;` separator with a
  trailing `;` on data rows, `,` decimal mark, level 1 rows before level 2 rows, newest month
  first. Raw and consisted values are equal except in the months left empty.

| ID | Covers (FR-007) | Period | Expected today | Marker |
|---|---|---|---|---|
| S1 | fewer than 10 complete water years | Jan/2000–Jun/2008, both levels | insufficient-data message | — |
| S2 | 10 to 29 years | Jan/1980–Jun/2000, both levels | full output, `sample_size_above_30_years: false` | — |
| S3 | no consisted data | Jan/1960–Jun/1995, level 1 only | full output, `empty_consistent_data: true` | — |
| S4 | empty water year in the middle | Jan/1970–Jun/2005, both levels; water year 1985 (Oct/1985–Sep/1986) empty in both | `empty_years: [1985]` | — |
| S5 | series starting in October | Oct/1980–Mar/2002, both levels | the complete water year 1980 is dropped (`first_year` 1981) | known defect D-02 / C-02 |
| S6 | series ending in September | Mar/1980–Sep/2001, both levels | the complete water year 2000 is dropped (`last_year` 1999) | known defect D-02 / C-02 |
| S7 | Oct/1990–Sep/2000, 10 complete water years | both levels | 8 years remain: insufficient-data message | known defect D-02 / C-02 |
| S8 | months missing in both consistency levels | consisted Jan/1965–Jun/2000, raw Mar/1968–Jun/2000; water year 1966 (Oct/1966–Sep/1967) empty in the consisted level, before the first raw month | `empty_years: [1966]`, through `fillna(0)` in `process_data.merge_and_fill_data` | suspected D-13 (no fix item) |

S8 uses a whole water year because a single missing month is neutralized: the per-year maximum
skips NaN, so the output would be the same with or without the `fillna` (research R-06).

C-01 (D-01: the BFGS fit ignores its declared bounds) affects every fitted result, real and
synthetic, so no series carries a marker for it.
