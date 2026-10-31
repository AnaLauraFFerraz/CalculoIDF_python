# 03 — Architecture: reasoning, decisions and open questions

> Record of the architecture reasoning discussed since September 2026, of the decisions already made and of those still open. Formal ADRs live in `docs/adr/`; this document holds the "why behind the why", written as study material.

## 1. Decision log

| Date | Decision | Status | Where |
|---|---|---|---|
| 2026-09 | Update the stack **before** any new feature | Decided | — |
| 2026-09 | Write characterization tests (*golden files*) **before** changing the structure | Decided (plan approved) | spec 001 |
| 2026-09 | Unify the backend in Python (FastAPI), removing the Node API and Firebase Storage | **Proposed** (amended on 2026-10-27: deployment on Vercel) | ADR 0001 |
| 2026-10 | Use **SDD with GitHub Spec Kit** in the new cycle | Decided | `06-spec-kit-guide.md` |
| 2026-10 | Create a dedicated Claude Project for CalculoIDF | Decided | — |
| 2026-10-27 | **State of the thesis code preserved**: `tcc-2023` tag + immutable release (thesis PDF + attestation) in each repository — `CalculoIDF_python` `bb1ab14`, `CalculoIDF_API` `06b7474`, `CalculoIDF_front` `f3bc7bb`. The original repositories will **never be deleted** (they hold the tags) | Done | `tcc-2023` releases |
| 2026-10-27 | **Phase 1 = backend modernization**: spec 001 → spec 002 (and C-02, if it fits). New features move to phase 2. Spec 008 is **not** brought forward — this settles the dilemma in `04` §3, accepting the loss of the 2026–27 El Niño window. Context: the Cloud Function's Python 3.10 runtime was deprecated on 2026-10-04 and will be decommissioned on 2027-04-04; the Node.js 18 configured in the Vercel project blocks new front-end builds | Decided | `04` §2–3 |
| 2026-10-27 | 002 × 004 boundary: modeling the **v0** contract as a Pydantic `response_model` goes into 002, without changing the JSON (verified by the golden files). The error taxonomy and the v1 contract stay in 004 | Decided | plan of 002 |
| 2026-10-27 | **Repository structure**: keep separate repositories; Spec Kit, constitution, specs and ADRs in `CalculoIDF_python` | **Accepted** | ADR 0002 |
| 2026-10-27 | **Finding (spike)**: running the thesis pipeline on the 13 CSVs with different library versions, only the **numpy 1.x → 2.x** upgrade changes results — `k, m, c, n` by up to ~18%, while the fitted curve (`i_calculated`) changes by at most ~1.2% and mean error/NS barely change. Python 3.11/3.12, pandas 2.3/3.1 and scipy 1.15 change nothing on their own (pandas 3 vs 2.3: differences ~1e-16). Likely cause: ill-conditioned fit (unbounded BFGS — D-01) amplifying rounding differences. pandas 3 requires numpy ≥ 2.0.2 | Recorded | section 4, items 4 and 5 |
| 2026-10-27 | **Backend deployment target: Vercel Functions, Python runtime, Hobby plan** (zero cost without a billing account; one platform instead of three). Cloud Run and Render rejected | Decided — amendment to ADR 0001 | ADR 0001 |
| 2026-10-27 | **The library upgrade in 002 stops at step 2a** (Python 3.12, pandas 2.3.3, numpy 1.26.4, scipy 1.15.3 — exact parity). numpy 2 / pandas 3 / current scipy leave the migration and go into phase 2 as a deliberate change, together with C-01 | Decided | ADR 0001; section 4, item 4 |
| 2026-10-27 | **Finding**: the code deployed to the Cloud Function (version 48, 2023-09-01) differs from the `tcc-2023` tag only in a line break (`main.py`), comments (`process_data.py`), a commented-out `# bounds=OPTIMIZATION_BOUNDS` and the final `pprint` (`ventechow.py`), and three commented-out lines in `requirements.txt`. **No difference affects the numbers**: production and tag are equivalent. The commented-out line confirms that the optimizer bounds were never applied (D-01) | Resolved | section 4, item 10 |
| 2026-10-27 | **Languages**: code, identifiers, comments, logs, test names, documentation, folder and file names, and commit messages in **English**. User-facing text (front-end, API error messages shown to users, reports) stays in **Brazilian Portuguese** | Decided | `CLAUDE.md`, constitution |
| 2026-10-27 | **Commits** carry **no** `Co-Authored-By` trailer | Decided | `CLAUDE.md` |
| 2026-10-29 | **Cross-environment comparison of the characterization suite**: exact (bit-for-bit) in the reference environment; elsewhere, numeric tolerances for pre-fit fields and structure-only for post-fit fields (`k, m, c, n` and everything derived from the fit), because the measured Windows × Linux differences after the fit cannot be bounded by a tolerance that still detects the required changes | **Accepted** | ADR 0003 |
| 2026-10-29 | **Finding**: the 2026-10-27 spike parity claim (Python 3.11/3.12, pandas 2.3.3 and scipy 1.15.3 with numpy 1.26.4 reproduce the output bit for bit) predates the CPU-kernel finding and compared versions on the same host; it must be re-verified against the golden files in the reference environment (spec 002) | Recorded | ADR 0003, "Consequences for spec 002" |
| 2026-10-31 | **Ruleset on `main`**: required status check `characterization`, branch up to date before merging, pull request required with 0 approvals, merge commits only. Reason for merge commits only: `research.md` of spec 001 cites commit SHAs as evidence, and squash or rebase would rewrite them | Done | spec 001, plan M-1 and research ("Ruleset and blocking demonstration") |

## 2. The reasoning behind the backend unification (summary of ADR 0001)

### 2.1 Every boundary has to pay for itself
Every network boundary charges a daily price: latency, **partial failure modes**, deployment surface, credentials, fragmented observability and cross-repository coordination. This is not an argument against boundaries — it is their price. The question for each one: **what does it give me that I would not have without it?**

**The four-question test.** Does the boundary separate things that:
1. **change for different reasons?** (independent axes of change)
2. **scale differently?**
3. have **different security/availability requirements?**
4. belong to **different teams**? (Conway)

If every answer is "no", the boundary is **accidental**. In CalculoIDF: the Node API changes whenever the Python JSON changes (1: no); there are no teams (4: no); no distinct requirements (3: no). Scale (2) is the only argument with some weight — see 2.4.

### 2.2 The decisive symptom: the boundary cuts a concept in half
The knowledge of "what a valid Hidroweb CSV is" (`skiprows=12`, ISO-8859-1, columns) lives in Python; the size limit and the natural validation point live in Node. To validate early, one either **duplicates knowledge** or **pays the full round trip** to find out the file was invalid. *When a boundary forces a choice between duplicating knowledge and paying for a useless round trip, it is in the wrong place.*

### 2.3 Storage used as transport (*claim check* out of context)
Firebase Storage persists nothing: the file is deleted right afterwards. It is a **means of transport** between services — the *claim check* pattern, legitimate when the payload does not fit in the message or the consumer is asynchronous. Here the payload is capped at 2 MB and the flow is synchronous. A whole dependency (credentials, lifecycle, its own failure mode) is paid for a nonexistent problem — and it also introduced a real bug: the **file-name race condition** (D-20).

### 2.4 Which side survives: the one with the least replaceable dependency
When a boundary collapses, the side whose dependency is harder to replace stays. Python carries scipy/pandas/numpy; the HTTP layer (CORS, *rate limit*, *multipart*, limits) is a commodity. Node does nothing FastAPI cannot do — it does the same thing in another language: the signature of an accidental boundary.

**The honest counter-argument:** Python is *CPU-bound* (BFGS), Node is *I/O-bound*; separating them would allow independent scaling. True, but it does not apply to this load (~1–2 s computation, small DataFrame, low demand). If it grows, the answer is a **queue with a job ID + polling**, not a language boundary.

### 2.5 Why FastAPI (and where to deploy)
- **Pydantic:** declarative validation at the edge (= the validation feature) and *response models* that turn the hand-built dictionary into a **contract**.
- Automatic **OpenAPI** → types for the TypeScript front-end.
- **async** for future external calls (ANA, AI provider).
- **Deployment:** ADR 0001 originally proposed Cloud Run (scale to zero, generous limits). The zero-cost requirement reopened the choice, settled on 2026-10-27 in favor of Vercel (Python runtime, Hobby plan) — see the amendment in ADR 0001.

### 2.6 General lessons (for the architecture study)
- **Two-way × one-way doors** (reversible × irreversible decisions): the unification is a two-way door — the core does not change, only its wrapper. Over-deliberating reversible decisions costs more than getting them wrong.
- **Tests anchored on observable behavior** (CSV in → JSON out), not on internal structure, are what give the freedom to reorganize the structure. They turn a "risky refactoring" into a "verifiable refactoring".
- **Architecture is contextual:** the 2023 design answered the constraints of the time (HTTP-invoked Cloud Function, CORS/payload limits, the front/API pattern learned). Reevaluating when the context changes is the normal cycle, not an error correction.
- **The best moment to simplify is before adding features**: every feature that crosses a boundary doubles the work.
- **Last responsible moment** (ADR 0002): when the benefit of a restructuring only shows up in the future and its cost is paid now, postponing the decision — with a written reevaluation trigger — is a legitimate choice, not procrastination.
- **Numerical parity depends on the whole environment** (finding of 2026-10-27): an ill-conditioned optimizer turns rounding differences (1e-16) into 18% differences in the parameters. Golden files must record the versions and platform they were generated on.
- **What is deployed is the real system** (finding of 2026-10-27): the repository and production may diverge when deployment is manual. Characterizing behavior requires knowing which of the two one is talking about.

## 3. ADR format adopted

Extended Nygard format, as in ADR 0001: **Status · Context (including why the current state exists) · Decision · Alternatives considered (with reason for rejection) · Rationale · Consequences (positive, negative, risks and mitigation) · Prerequisites · Reversibility · When to revisit.**

Lifecycle rules:
- Status: `Proposed` → `Accepted` (with date) → eventually `Superseded by ADR NNNN`.
- **Accepted ADRs are neither edited nor deleted**; a change of course = a new ADR that supersedes the previous one. A `Proposed` ADR may be amended.
- The value of the format lies in its history, including the decisions that went wrong.

## 4. Open decisions

1. **ADR 0001 → `Accepted`?** Prerequisite: spec 001 completed. Deployment target already decided (Vercel). Before accepting, measure *cold start* and recheck the Hobby plan limits.
2. ~~ADR 0002 — repository structure~~ — **resolved** on 2026-10-27: separate repositories, Spec Kit in `CalculoIDF_python` (ADR 0002 `Accepted`).
3. **TypeScript in the front-end** — record it as part of the plan of spec 003 (or as an ADR).
4. ~~Upgrading the Python libraries as a separate step~~ — **decided** on 2026-10-27 (002 stops at step 2a). Spike record:
   - **Step 2a** (**exact** parity shown on the 13 CSVs): Python 3.12 + pandas 2.3.3 + numpy 1.26.4 + scipy 1.15.3. Replacing scikit-learn with `numpy.polyfit` (difference ~1e-14 in the regression, which is output only) goes into its own commit, with a declared tolerance.
   - **Step 2b** (numpy 2.x, pandas 3.x, current scipy): **changes numbers**. It is not "parity within tolerance": it is a deliberate change (principle I), with updated golden files, a table of differences and a justification. Phase 2, together with C-01.
5. **Fixes for D-01, D-02 and D-07** change results. Each must be a deliberate change, with updated golden files and a recorded justification — and, since they touch the thesis methodology, each deserves a short ADR ("Methodological fix: …"). **C-01 gained a new motivation:** a well-conditioned fit (bounds applied, scaled parameters, smooth objective function) tends to give the same result across versions and platforms.
6. **Data source for the automated search** (ANA's legacy SOAP API × HidroWebService with registration) — see spec 005.
7. **AI provider and cost** (spec 007) and whether the feature is optional/usage-limited.
8. **ONI × RONI** to classify ENSO years (spec 008).
9. ~~Zero-cost deployment target for the unified backend~~ — **decided** on 2026-10-27: Vercel Functions, Python runtime, Hobby plan (amendment to ADR 0001).
10. ~~Deployed code × `tcc-2023` tag~~ — **resolved** on 2026-10-27: the differences are cosmetic (formatting, comments, `pprint`). (a) None changes numbers; (b) the golden files of spec 001 characterize the repository code, which is numerically identical to production; (c) a second release is not needed — the ZIP of the deployed code is kept locally as a record.

## 5. Target architecture (if ADR 0001 is accepted)

```
[Browser] ── React (Vercel) ──HTTPS──▶ [Python service: FastAPI as a Vercel Function (Python runtime)]
                                          ├─ edge: restricted CORS, rate limit, 2 MB limit, validation (Pydantic)
                                          ├─ computation core (current modules, untouched)
                                          ├─ future: ANA client (search), ONI/RONI table (ENSO), AI client
                                          └─ versioned response contract (OpenAPI → front-end types)
```

No Firebase Storage, no Node API and no Google Cloud. The `CalculoIDF_API` repository is **archived**, not deleted (it preserves reversibility and the historical record of the thesis).
