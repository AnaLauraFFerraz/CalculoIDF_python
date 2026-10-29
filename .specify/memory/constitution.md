<!--
Sync Impact Report
- Version change: template (unversioned) → 1.0.0 (initial ratification; based on draft 0.2.0
  from kickoff-sdd/repo/.specify/memory/constitution.md)
- Principles (template slots → named principles; template had 5 slots, project defines 8):
  - [PRINCIPLE_1_NAME] → I. Numerical correctness is non-negotiable
  - [PRINCIPLE_2_NAME] → II. Characterize before restructuring
  - [PRINCIPLE_3_NAME] → III. Methodological and uncertainty transparency
  - [PRINCIPLE_4_NAME] → IV. AI interprets, never computes
  - [PRINCIPLE_5_NAME] → V. Simplicity and justified boundaries
  - added → VI. Spec before code, with human approval
  - added → VII. Faithfulness to official sources and provenance
  - added → VIII. Accessible to the non-specialist
- Added sections: preamble (project purpose); [SECTION_2_NAME] → Additional Constraints;
  [SECTION_3_NAME] → Development Workflow
- Removed sections: none
- Changes relative to draft 0.2.0: structure adapted to the template; normative wording
  tightened to MUST/MUST NOT; workflow step 1 lists /speckit-implement and /speckit-converge
  and the separate `docs:` approval commit, matching CLAUDE.md. Principle I replaces "beyond
  the tolerance defined in spec 001" with identical output in the same environment and the
  per-field tolerance of the characterization suite across environments. No principle
  redefined.
- Ratified: 2026-10-29 | Last Amended: 2026-10-29
- Templates: not modified (scope of this command); plan/spec/tasks templates read the
  constitution at runtime.
- Follow-up TODOs:
  - Workflow step 6 (CI blocks merge) describes a rule for CI that does not exist yet.
-->

# CalculoIDF Constitution

CalculoIDF determines the Intensity-Duration-Frequency equation of rainfall from official
rain-gauge series, for professionals **without training in statistical hydrology**. Its
results feed the design of drainage works: a wrong number here becomes an undersized culvert
there. The principles below follow from that.

## Core Principles

### I. Numerical correctness is non-negotiable

Every change that alters any numeric value of the output MUST be **deliberate, justified and
traceable**: golden files are updated explicitly, the reason is recorded in the reference
changelog, and the PR shows the table of differences per station. Structural changes
(refactoring, migration, infrastructure upgrades) MUST NOT change results: the output stays
identical in the same environment, and comparisons across environments use the per-field
tolerance declared by the characterization suite. Methodological fixes are changes of their own, never made "along the way"
in another change.

*Rationale:* the product's value is the reliability of the number; a silent error is worse
than a visible failure.

### II. Characterize before restructuring

No restructuring starts without tests anchored on **observable behavior** (data in → result
out) covering the affected path. Characterization tests freeze the current behavior —
**including known defects**, marked as such — so that fixing a defect is an explicit decision,
not a side effect.

*Rationale:* behavioral tests are what make it safe to change the structure.

### III. Methodological and uncertainty transparency

Every result MUST be presented with the **method that produced it** (chosen distribution and
alternatives, removed outliers, years analyzed and discarded, consistency level of the data,
provenance) and with its **quality metrics** (mean relative error, Nash-Sutcliffe). Known
limitations of the series MUST always be reported. No number is shown without the context
needed to judge it. "No detectable signal" and "insufficient sample" are valid results and
MUST be communicated as such.

*Rationale:* a non-specialist user cannot tell on their own when a result is fragile.

### IV. AI interprets, never computes

Features based on language models MUST NOT produce or change numeric values. They only
interpret results already computed, are grounded exclusively in those results, are always
labeled as auxiliary interpretation and go through a check that every number they cite exists
in the input data.

*Rationale:* language models can be fluently wrong; the computation has to be deterministic
and auditable.

### V. Simplicity and justified boundaries

Every new service, network boundary, storage or external dependency MUST have an explicit
justification through the four-question test (axes of change, scaling profile,
security/availability requirements, ownership by different teams) and MUST be recorded in an
ADR. When in doubt, fewer pieces.

*Rationale:* every boundary charges latency, failure modes and maintenance every day; the
project has a single maintainer.

### VI. Spec before code, with human approval

No implementation starts without a spec and a plan **approved by the maintainer** (`Approved`
status with a date in the header). Specs describe what and why; technology decisions belong in
the plan. `[NEEDS CLARIFICATION]` markers MUST be resolved before approval.

*Rationale:* the tooling checks that a plan exists, not that someone approved it.

### VII. Faithfulness to official sources and provenance

Rainfall data comes from ANA (Hidroweb/SNIRH); climate indices come from official sources
(NOAA CPC, INMET). Every analysis MUST record its provenance: source, station, period,
consistency level, retrieval date. Static reference data (statistical tables, ENSO indices) is
versioned in the repository with its origin and date.

*Rationale:* reproducibility is a requirement of engineering documentation.

### VIII. Accessible to the non-specialist

Wording, error messages and warnings MUST be written for engineers without statistical
training: they say what happened, why it matters and what to do. Errors are never generic when
the cause is known.

*Rationale:* it has been the stated goal of the project since the thesis.

## Additional Constraints

- **Numerical core in Python** (pandas, numpy, scipy) — non-negotiable; it is not rewritten in
  another language.
- Architecture follows the accepted ADRs in `docs/adr/` (see ADR 0001 — backend unification —
  and ADR 0002 — repository structure).
- **Languages:** code, identifiers, comments, logs, test names, documentation, file and folder
  names, and commit messages in English. The user interface and every user-facing message are
  in **Brazilian Portuguese (pt-BR)**.
- Units: intensity in mm/h, duration in minutes, return period in years.
- Upload limit: 2 MB (until a different decision is recorded).
- Secrets (API keys, credentials) are never in the front-end or in the repository.
- Dependencies: no unused dependency; pinned versions; upgrading numerical libraries is a
  **separate** change, verified by the golden files.

## Development Workflow

1. Spec (`/speckit-specify`) → clarification (`/speckit-clarify`) → **approval** → plan
   (`/speckit-plan`) → **approval** → tasks (`/speckit-tasks`) → consistency analysis
   (`/speckit-analyze`) → implementation (`/speckit-implement`, `/speckit-converge`).
   Approval is recorded as `**Status**: Approved (YYYY-MM-DD)` in the header of the spec and of
   the plan, in a separate `docs:` commit.
2. One branch per feature, with the same name as the spec folder.
3. Commits follow Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
   `chore:`, `ci:`), with an English subject and no `Co-Authored-By` trailer.
4. Architectural decisions are recorded as ADRs (extended Nygard format; an accepted ADR is not
   edited — it is superseded).
5. Golden files are updated only through the explicit procedure, with the reason recorded.
6. The characterization suite runs in CI on every push and blocks the merge if it fails.

## Governance

This constitution supersedes other project practices. Amendments are made through a PR that
describes the change, the reason and the impact on existing specs, and that bumps the version:

- **MAJOR** — a principle removed or redefined incompatibly;
- **MINOR** — a new principle or section, or a material expansion;
- **PATCH** — wording, clarification, correction.

Every spec and every plan MUST declare compliance with the principles (or explicitly justify
the exception).

**Version**: 1.0.0 | **Ratified**: 2026-10-29 | **Last Amended**: 2026-10-29
