# 04 — Roadmap: features, order and why

> Everything discussed as an improvement or a feature, prioritized. Items with a drafted spec live in `specs/NNN-*/spec.md`; the rest are in the **backlog** (section 5), ready to become a spec when their turn comes.

## 1. Ordering principle

**Each step protects the next one.** Characterization tests protect the migration; the migration (one boundary fewer) makes validation, errors and integrations cheaper; validation and errors lay the ground for external integrations (ANA, AI), which bring new failure modes.

## 2. Proposed sequence

| Phase | Spec | What | Depends on | Why in this position |
|---|---|---|---|---|
| 0 | — | Decide ADR 0002 (repositories), `.gitattributes` (D-41), `specify init`, ratify the constitution | — | Defines where the specs live |
| 1 | **001** | Characterization tests (*golden files*) | — | Blocking prerequisite of ADR 0001; does not change behavior |
| 1 | **002** | Backend unification (ADR 0001) | 001, ADR 0001 accepted | Removes the boundary before features cross it |
| 2 | **003** | Front-end modernization (CRA → Vite + TS) | v0 contract | CRA is deprecated; typed base for everything that follows |
| 2 | — | Methodological fixes C-01, C-02, C-03 (section 4) | 002 | Only after parity is proven — otherwise one cannot tell whether a difference came from the migration or from the fix |
| 2 | **004** | CSV validation + error taxonomy + data quality report | 002 | Validation lives next to the knowledge of the format |
| 2 | **005** | Integrated Hidroweb search | 004 | Biggest friction reduction for real use; brings a new failure mode (ANA down) |
| 2 | **006** | Full IDF table + export (PDF, spreadsheet, equation) | 003 | The most "professional" feature: the calculation record goes into the engineer's project |
| 2 | **007** | AI interpretation of the results | 004, 006 | Needs stable, well-described results to interpret |
| 2 | **008** | ENSO analysis (core) | 001 (+ 003 for the visual part) | See section 3 |

Spec numbers reflect this order. Spec Kit numbers specs at creation; if specs are created in a different order, the folders must be renamed.

**Phase 1** of the current cycle covers the backend only: 001 → 002, and C-02 if it fits. Everything else is **phase 2** (decision of 2026-10-27, see `03-architecture-and-decisions.md`).

## 3. The El Niño timing dilemma

The 2026–27 El Niño is happening **now** and is expected to last until early 2027 (see `05-enso-el-nino.md`). An ENSO feature is worth more **during** the event. Following the order above, 008 would arrive late.

**Possible way out:** the computational part of 008 (classifying years, sensitivity index, representativeness warning) consists of **pure** Python functions added to the pipeline — and the computation core **does not change** with the migration (ADR 0001). So the core of 008 could be implemented right after 001, on the current stack, with no rework on the Python side. The rework would be limited to the display (redone in 003, an almost mechanical component migration).

Trade-off: bringing it forward = taking advantage of the season (and perhaps a timely paper/presentation) at the cost of touching the old stack and of carrying one more feature through the 002 parity work.

**Decided on 2026-10-27:** 008 is **not** brought forward; missing the 2026–27 El Niño window is an accepted cost (see `03-architecture-and-decisions.md`).

## 4. Methodological fixes (they change numbers — treat as deliberate changes)

| ID | Origin | Fix | How to handle it |
|---|---|---|---|
| C-01 | D-01 | Actually apply the optimization bounds (e.g., L-BFGS-B) | Short ADR + updated golden files + before/after comparison on the 13 CSVs. Do it together with the numpy 2 / pandas 3 upgrade (step 2b), since the current fit is ill-conditioned and changes with numpy |
| C-02 | D-02 | Do not drop a complete water year when the series starts in October / ends in September | **Confirmed bug** (can push a valid series below 10 years). Spec 001 freezes the current behavior; fix it right after 002 — the highest-priority fix |
| C-03 | D-07 | Recompute statistics after each outlier removal (if the literature confirms it) | Validate against Naghettini & Pinto / the advisors before changing |

Suggested process: each fix becomes a **small spec** (or an ADR amendment + PR), and the PR shows the table of numerical differences per station. This is excellent material for a "thesis revision" paper.

## 5. Backlog (no spec yet)

| ID | Feature | Value for real use | Link |
|---|---|---|---|
| B-01 | **Configurable return periods and durations**; manual choice of distribution (automatic by default) | Project-specific regulatory requirements | — |
| B-02 | **Water year configurable per region** | Unlocks use outside Southeast Brazil | Thesis future work 7; relevant for 005 (nationwide search) |
| B-03 | **Comparison** between stations and with published IDF equations for the municipality | Answers "can I use this station in my project?" | — |
| B-04 | **Analysis history** with a shareable link | Turns the calculator into a work tool | Requires persistence (and maybe login) → ADR |
| B-05 | **GEV + goodness-of-fit tests** (KS, Anderson-Darling) instead of R² | Methodological credibility | Thesis future work 4 and 6; basis for B-11 |
| B-06 | Other optimization methods | Robustness of the fit | Thesis future work 5; partly in C-01 |
| B-07 | Method for gaps / non-continuous series | Today a gap = 0 (D-03) | Thesis future work 8; 004 only **reports**, the method belongs here |
| B-08 | Support for purely raw data and a raw × consisted criterion | — | Thesis future work 1 and 2 |
| B-09 | Other outlier tests | — | Thesis future work 3; related to C-03 |
| B-10 | ENSO: IDF curves conditioned on phase (exploratory, with a minimum-sample guard) | Research | 008 |
| B-11 | ENSO: **non-stationary GEV** with ONI/RONI as a covariate | Methodologically correct version; publishable | 008 + B-05 |
| B-12 | ENSO: **season context panel** (current ENSO state × station history) | Connects the app to the present moment | 007 + 008; depends on an external source |
| B-13 | Map of stations near a point | Station discovery | 005 (story P3) |

## 6. The original list of ideas (September 2026) and where each item went

| Original idea | Where it is | Comment from the analysis |
|---|---|---|
| Update the stack | 002 + 003 (+ ADR 0001) | Use the occasion to rethink the architecture, not just the versions |
| Automated search integrated with Hidroweb | 005 | Highest value for real use; reduces dependence on the fragile CSV format |
| Automated tests | 001 (+ tests inside each spec) | The 13 CSVs are ready-made fixtures; do it **before** the migration |
| Better error handling | 004 | Today there are unreachable error branches (D-30) and errors returned as HTTP 200 (D-04) |
| CSV content validation | 004 | Do not over-invest before deciding on 005, which reduces the importance of the upload |
| AI integration to analyze the results | 007 | AI **interprets, never computes** |
| (October 2026) Connect the project to El Niño | 008 + B-10/11/12 | Touches the central premise of the method: stationarity |
