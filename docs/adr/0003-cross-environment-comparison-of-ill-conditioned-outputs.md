# ADR 0003 — Cross-environment comparison of ill-conditioned outputs: structure-only for post-fit fields

- **Status:** Accepted (2026-10-29)
- **Date:** 2026-10-29 (opened and decided)
- **Authorship:** project maintainer
- **Scope:** `CalculoIDF_python` — the characterization suite (spec 001) and the library upgrade (spec 002)
- **Related:** ADR 0001 (version spike, numerical reproducibility); spec 001, research R-03 and R-05; constitution principle I; debt D-01 and backlog item C-01

---

## Context

### What the characterization suite compares

The golden files are generated in a reference environment equal to production: Linux x86-64, Python 3.10, the pins of `requirements.txt` and, since research R-03, pinned CPU kernels (`OPENBLAS_CORETYPE=Haswell`, `NPY_DISABLE_CPU_FEATURES` with the AVX-512 list). In that environment the comparison is exact (`==`). In any other environment (the Windows development machine) spec 001 (FR-002) planned per-field numeric tolerances, set by measuring the differences and applying a safety factor: the measured maximum × 10, rounded up to one significant digit, with a floor of `rel 1e-12` (research R-05).

### What the measurement found (task T017, 2026-10-29)

Golden files from CI run 40916698777 (Linux, Python 3.10.22) against a run on Windows (Python 3.10.22, same pins, same kernel variables). An independent script on another CPU (Intel i7-9750H, Linux VM) reproduced all 13 golden files exactly, so the output is deterministic on Linux with the pinned kernels; the difference below is between operating systems. The two series that end in the insufficient-data message coincide. For the 11 series with numeric output:

| Stage | Field group | Numeric leaves | Differing leaves | Max abs Δ | Max rel Δ |
|---|---|---:|---:|---:|---:|
| Pre-fit | `graph_data.**` | 1191 | 88 | 9.948e-13 | 6.131e-15 |
| Pre-fit | `**` (includes `i_real`) | 1334 | 180 | 6.821e-13 | 2.169e-15 |
| Post-fit | `parameters.**` | 88 | 63 | 7.545e+01 | 1.795e-01 |
| Post-fit | `intensity_graph_data_*.i_calculated[*]` | 1320 | 1088 | 1.375e+00 | 1.698e-02 |
| Post-fit | `intensity_graph_data_*.regression.*` | 44 | 38 | 4.213e-01 | 3.482e+00 |
| Post-fit | `mean_relative_errors.*` | 22 | 18 | 8.688e-02 | 2.919e-02 |
| Post-fit | `ns.*` | 22 | 18 | 6.826e-04 | 6.896e-04 |

The full table per series is in `specs/001-characterization-tests/research.md` ("Measurement results").

Everything computed before the fit agrees to rounding noise (about 1e-13). Every non-trivial difference appears after the Ven Te Chow fit, which is ill-conditioned and ignores its declared bounds (D-01): small platform differences are amplified into the parameters `k, m, c, n`, which move by up to 17.9% while the fitted intensities move by at most 1.7%. The cause of the residual platform difference (operating system, compiler, math library) was not isolated.

### What the literal rule would give

Applying the approved R-05 rule to these maxima gives relative tolerances of about 2 for `parameters.**`, 0.2 for `i_calculated`, 40 for `regression.*`, 0.3 for `mean_relative_errors.*` and 7e-3 for `ns.*`. A relative tolerance of 0.2 (20%) on the intensities cannot detect the perturbations that spec 001 (SC-003) requires the suite to detect — a change of 0.001 in a disaggregation coefficient of 0.1 to 1.1 is a relative change of 0.1% to 1% — so the post-fit comparison in tolerant mode would pass almost any change. Research R-05 says to report such a case instead of widening a tolerance silently.

## Decision

**Post-fit fields are compared by structure only outside the reference environment; pre-fit fields keep numeric tolerances; the exact comparison in the reference environment stays the only value check for post-fit fields.**

- A new rule kind, `structure`, in `tests/golden/tolerances.json`. For the matching fields it compares types, keys, list lengths and NaN-ness (a `NaN` against a number), and does not compare values. Every `structure` rule carries a mandatory, non-empty `reason`; the suite refuses to run in tolerant mode without it.
- Post-fit groups that get the `structure` rule, with D-01 as the recorded reason: `parameters.**`, `intensity_graph_data_*.i_calculated[*]`, `intensity_graph_data_*.regression.*`, `mean_relative_errors.*`, `ns.*`.
- Pre-fit groups (`graph_data.**` and the default `**`, which includes `i_real`) keep numeric tolerances set by the R-05 rule from the measurement.
- In the reference environment (`exact` mode, the CI job) nothing changes: every field is compared with `==`.
- The dividing line is pre-fit versus post-fit, because that is where the measured behavior changes by many orders of magnitude (relative differences of about 1e-15 before the fit, 1e-4 to 1e+0 after it), not the relevance of a field to the engineer.

## Alternatives considered

### A. Keep the literal rule (measured maximum × 10) — rejected

It is the approved rule and needs no change. The resulting tolerances (0.2 to 40, relative) make the post-fit comparison nearly empty while appearing to be a check, which is worse than declaring it empty.

### B. Per-series tolerances — rejected

It narrows the values for the series that behave well, but the worst series are the ones that matter (17.9% in `parameters.**` and 348% in `regression.*` for `chuvas_C_02346328`), 13 series × 5 groups would be tuned to one measurement on one machine, and every new series (spec 003 onward, synthetic cases included) would need a new measurement. No principled bound exists for an ill-conditioned fit.

### C. `null` — no cross-environment value comparison at all — rejected

It is honest about post-fit fields, but it also drops the pre-fit numeric checks, which are tight and reliable (1e-12 against a measured 1e-13) and are exactly what detects a change in a disaggregation coefficient, a distribution or the outlier test.

### D. Structure-only for post-fit fields, numeric tolerances for pre-fit fields — chosen

It keeps every check that carries information outside the reference environment and states plainly which check does not exist there.

## Consequences

### Positive

- Tolerant mode makes no claim it cannot support: its numeric tolerances are tight (about 1e-12) and measured, and its post-fit rule is declared with a reason.
- A change in the pre-fit pipeline (coefficients, distributions, outlier test, the data stage) is detected on any machine.
- The CI job in the reference environment remains the single place where post-fit values are checked, and it blocks the merge.

### Negative and accepted costs

- On a non-reference machine a change that only touches the fit (the initial guess, the objective, the optimizer options) is invisible: the structure stays the same. It is caught only by the CI job, or by running the suite in a reference-equivalent Linux environment (a VM or WSL with the pinned kernels, where the fingerprint matches and the mode is `exact`).
- A fit that silently converges to a wrong but finite value passes the tolerant comparison.
- The approved spec 001 (FR-002 and SC-003) and plan described per-field numeric tolerances for every field; they are amended accordingly (spec 001, "Amended: 2026-10-29 (ADR 0003)", and the plan).

### Consequences for spec 002 (library upgrade)

An upgrade of Python, pandas, scipy or numpy changes the reference environment. Acceptance of the upgrade is:

1. **pre-fit fields** within the numeric tolerances;
2. **post-fit fields** structurally equal (types, keys, list lengths, NaN-ness);
3. a **deliberate re-baseline of the golden files in the new reference environment**, with the reason, the table of differences per station and the changed fields recorded in `tests/golden/CHANGELOG.md` (constitution principle I).

Where an upgrade reproduces the golden files exactly, no re-baseline happens and the exact comparison remains the criterion. The 2026-10-27 spike (ADR 0001: Python 3.11/3.12, pandas 2.3 and scipy 1.15 with numpy 1.26.4 reproduced the output bit for bit) predates the CPU-kernel finding (research R-03) and compared versions on the same host, so it does not prove parity against the golden files generated in CI; spec 002 must re-verify it in the reference environment.

### Constitution, principle I

Principle I (v1.0.0) requires that, across environments, comparisons "use the per-field tolerance declared by the characterization suite". The decision is read as a declared per-field rule — pre-fit fields carry a numeric tolerance, post-fit fields carry a structural rule with a recorded reason — and the same-environment clause ("the output stays identical") is kept by the exact mode. No amendment is proposed.

*Counter-argument.* A structural rule is not a tolerance: it removes the numeric check for those fields outside the reference environment, and a strict reading of "per-field tolerance" does not cover it. Under that reading the principle's wording needs a PATCH clarification ("per-field rule"), which remains available. The ADR is accepted on the first reading, which rests on the exact comparison remaining mandatory and blocking in the reference environment, and on the alternative (A) producing a numeric tolerance that checks nothing.

## Reversibility

A **two-way door**. The `structure` rule is data in `tests/golden/tolerances.json`; replacing it with a numeric rule changes one entry and its measurement record. No golden file is affected.

## When to revisit

- When C-01 (fix of D-01: bounded, well-conditioned fit) is delivered: the post-fit fields should then be stable across environments and the `structure` rules should become measured numeric tolerances.
- When a second reference environment is needed (for example after the library upgrade of spec 002), to check whether the differences between operating systems shrink or grow.
- If a regression in the fit goes undetected locally and is caught only by CI more than once, reconsider running the suite in a reference-equivalent Linux environment as the default local setup.
