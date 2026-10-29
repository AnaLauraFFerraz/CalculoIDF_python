# Feature Specification: Characterization tests for the IDF computation (golden files)

**Feature Branch**: `001-characterization-tests`

**Created**: 2026-10-29

**Status**: Approved (2026-10-29)

**Input**: User description: "Freeze the current behavior of the IDF computation on real Hidroweb
series, so that any future change that alters results is detected before it reaches the user."

**Depends on**: —

**Blocks**: 002 (prerequisite of ADR 0001), C-01, C-02, C-03

## Current System Context

No repository has automated tests (D-40). The backend repository holds 13 real Hidroweb CSV
files that today are only used for manual runs. There are suspected defects that change results
(D-01, D-02, D-07) and that will be fixed later — the tests must tell "the migration changed a
number" apart from "the fix changed a number".

## Clarifications

### Session 2026-10-29

- Q: Which tolerance values does the comparison use (FR-002)? → A: The references are generated
  in a reference environment equal to production (Linux, the production Python version and the
  pinned library versions); the automatic check runs in that same environment and requires
  exact (bit-for-bit) equality. In any other environment the comparison uses per-field
  tolerances, with `k, m, c, n` separate from intensities and metrics. The values are set during
  planning by measuring the actual differences between the development machine and the
  reference environment, and are recorded with the references.

## User Scenarios & Testing *(mandatory)*

> Primary user: the maintainer. Protected user: the engineer who receives the results.

### User Story 1 - Detect any change in the results (Priority: P1)

As the maintainer, I want to run a suite that processes each reference series and compares the
full result with a frozen reference, so I know immediately whether a change altered any number.

**Why this priority**: without it, the migration (002) and the methodological fixes are blind.

**Independent Test**: run the suite on a clean checkout; it passes. Change a disaggregation
coefficient by one decimal place; it fails, pointing to the affected fields and stations.

**Acceptance Scenarios**:

1. **Given** the unchanged code, **When** the suite runs, **Then** every reference series passes.
2. **Given** a change that alters any numeric value of the output beyond the tolerance, **When**
   the suite runs, **Then** it fails and shows, per station, the field path, the expected value
   and the actual value.
3. **Given** a series whose processing ends in a business message (e.g., insufficient data),
   **When** the suite runs, **Then** the message is also compared with the reference.

---

### User Story 2 - Update references deliberately (Priority: P2)

As the maintainer, I want an explicit procedure to regenerate the reference of one or more
stations that requires recording the reason, so that intentional fixes are documented.

**Why this priority**: without a deliberate path, the temptation is to "rewrite everything until
it passes", which defeats the protection.

**Independent Test**: regenerate the reference of one station giving a reason; the change log
gets an entry with date, station, changed fields and reason; the suite passes again.

**Acceptance Scenarios**:

1. **Given** an intentional change, **When** the maintainer regenerates the references without
   giving a reason, **Then** the procedure refuses.
2. **Given** a reason, **When** the regeneration finishes, **Then** the change log describes what
   changed.

---

### User Story 3 - Document current behavior in edge cases (Priority: P2)

As the maintainer, I want test cases that record the current behavior in edge situations —
including suspected defects — so that fixing each one later is a visible decision.

**Why this priority**: the known defects (D-02, D-07, D-01) will be fixed later; freezing their
current behavior now is what makes each fix an explicit, visible change.

**Independent Test**: each edge case has a series (real or synthetic, in Hidroweb format) and a
reference; cases marked as "suspected behavior" point to the corresponding backlog item.

**Acceptance Scenarios**:

1. **Given** a series with fewer than 10 complete water years, **Then** the recorded result is
   the insufficient-data message.
2. **Given** a series with 10 to 29 years, **Then** the result records the "sample shorter than
   30 years" warning.
3. **Given** a series with no consisted data, **Then** the result records the use of raw data.
4. **Given** a series with an empty water year in the middle, **Then** the result records that
   year as empty.
5. **Given** a series that **starts in October** or **ends in September**, **Then** the result
   records the current behavior — which drops one complete water year in each case — marked as a
   known defect (D-02 / C-02). Mandatory case: series Oct/1990–Sep/2000 (10 complete years),
   which today results in 8 years and "insufficient data".
6. **Given** a series with a month missing in both consistency levels, **Then** the result
   records the current behavior for that month.

---

### User Story 4 - Automatic protection on every change (Priority: P3)

As the maintainer, I want the suite to run automatically on every push and to block integration
if it fails.

**Why this priority**: it removes the dependence on remembering to run the suite, but the suite
already protects the project when run manually (User Stories 1–3).

**Independent Test**: open a PR with a numeric change; the automatic check fails and blocks the
merge.

**Acceptance Scenarios**:

1. **Given** a change that alters a numeric value of the output, **When** it is pushed and
   proposed for integration, **Then** the automatic check fails and the integration is blocked.

---

### Edge Cases

- Minimal numeric differences caused by floating point or by math libraries, with no logic
  change.
- Library warning messages during execution (they must not break the comparison).
- The output mixes types (a list or `false` in `empty_years`; years as text) — the comparison
  must handle the format as it is today.
- The computation depends on cloud services to obtain the file — the tests must not depend on
  them.
- Upgrading the numerical libraries can change the fitted parameters (`k, m, c, n`) much more
  than the fitted intensities (D-01); the tolerance must take this into account per field.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The suite MUST process each of the 13 real reference series and compare the
  **full** output with that series' frozen reference.
- **FR-002**: The comparison MUST use a configurable numeric tolerance, definable per field. In
  the reference environment (equal to production: Linux, the production Python version and the
  pinned library versions) the comparison MUST require exact (bit-for-bit) equality. In any other
  environment (e.g., the Windows development machine) it MUST use per-field tolerances, with the
  fitted parameters (`k, m, c, n`) set separately from the intensities and the metrics. The
  tolerance values are not fixed in this spec: they are set during planning by measuring the
  actual differences between the development machine and the reference environment, and are
  recorded with the references.
- **FR-003**: On failure, the suite MUST report station, field path, expected value and actual
  value.
- **FR-004**: The suite MUST run without network access and without cloud credentials.
- **FR-005**: The suite MUST be deterministic: two consecutive runs in the same environment
  produce the same result.
- **FR-006**: Regenerating references MUST require a reason and record it in a versioned change
  log.
- **FR-007**: There MUST be edge cases covering: < 10 years, 10–29 years, no consisted data, an
  empty year in the middle, a series starting in October, a series ending in September, a month
  missing in both consistency levels.
- **FR-008**: Cases that freeze suspected behavior MUST be identified as such and reference the
  fix item (C-01, C-02, C-03).
- **FR-009**: The references MUST be generated in the library environment currently in
  production, and record that environment (language and library versions, platform).
- **FR-010**: The suite MUST NOT require changes to the behavior of the production code; minimal
  adjustments to make it testable are only accepted if provably neutral (the suite itself is the
  proof).
- **FR-011**: The suite MUST run automatically on every push, in the reference environment, and
  block integration on failure.

### Key Entities

- **Reference series**: an input file in Hidroweb format (real or synthetic), with an identifier
  and a description of what it covers.
- **Frozen reference (golden)**: the full expected output for a series, plus metadata
  (generation environment, date).
- **Reference change log**: date, affected series, changed fields, reason.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of the 13 real series and of the 7 edge cases covered.
- **SC-002**: The full suite runs in under 60 seconds on a development machine.
- **SC-003**: A change of 0.001 in any disaggregation coefficient, in the optimization's initial
  guess or in a value of the statistical tables makes the suite fail (sensitivity test done
  once, manually).
- **SC-004**: Zero network calls during execution.
- **SC-005**: Every reference update in the history has a recorded reason.

## Assumptions

- The current production environment can be reproduced as a reference environment (Linux, same
  Python version and the pinned library versions), both for generating the references and for
  the automatic check.
- The 13 CSVs in the repository can be used as test fixtures.

## Out of Scope

- Fixing any defect (D-01, D-02, D-07 etc.) — only documenting the current behavior.
- Front-end tests (they belong to 003).
- Validating whether the current results are **right** from a hydrological standpoint — the
  suite guarantees that they **do not change by accident**.

## Constitution Compliance

Directly implements principles **I** (numerical correctness) and **II** (characterize before
restructuring).
