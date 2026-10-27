# ADR 0001 — Unify the backend in Python and remove the Node API and Firebase Storage

- **Status:** Proposed
- **Date:** 2026-09-01 · amended on 2026-10-20 and 2026-10-27
- **Authorship:** project maintainer
- **Scope:** `CalculoIDF_front`, `CalculoIDF_API`, `CalculoIDF_python`
- **Supersedes:** the implicit architecture defined during the undergraduate thesis (2023)

---

## Context

### How the system works today

The application's essential behavior is simple: **a Hidroweb CSV goes in, a JSON with the fitted IDF equation comes out.** Today that behavior is delivered by three separately deployed services:

1. **`CalculoIDF_front`** — React 18 on Create React App, hosted on Vercel. Sends the CSV via `multipart/form-data` to the API.
2. **`CalculoIDF_API`** — Node/Express on Render. Receives the file (`multer`, in memory, 2 MB cap), applies CORS and *rate limiting*, uploads the CSV to Firebase Storage, calls the Cloud Function with the file URL, relays the JSON to the front-end and deletes the Storage object.
3. **`CalculoIDF_python`** — Google Cloud Function. Downloads the CSV from Storage, runs all the hydrological and statistical processing (`process_data`, `outlier_test`, `distributions`, `k_coefficient`, `disaggregation_coef`, `ventechow`) and returns the results JSON.

The thesis flow diagram (Figure 8) has **ten numbered arrows** to deliver this "CSV in, JSON out".

### Why it was built this way

The design was not a mistake — it answered the constraints of 2023:

- Cloud Functions are invoked over HTTP, and sending `multipart/form-data` from the browser straight to the function ran into CORS, authentication and payload limits. Passing a **storage URL** was the established path.
- The front/API split was the standard design for a web application, and Firebase Storage integrated naturally with the rest of the Google ecosystem already in use.
- The goal of the work was to show that the IDF computation could be automated, not to optimize the deployment topology.

### What changed

- **The stack has to be updated anyway.** Create React App is deprecated; Firebase 9, pandas 2.0.1 and scipy 1.10.1 are outdated. There will be migration work regardless — this is the right and cheapest moment to reevaluate the topology, because the marginal cost is close to zero.
- **The next features cross the boundary.** CSV content validation, specific error handling, integrated Hidroweb search and AI analysis all require both halves of the backend to agree on the same subject.

### Additional evidence (amendment of 2026-10-20, still Proposed)

A detailed code review found defects **caused by the boundary itself**, which reinforce the decision:

- **File-name race condition.** The API stores the blob under the original upload name. Hidroweb names files after the station code (`chuvas_C_01944009.csv`), so two users analyzing the same station at the same time overwrite each other's file — and whoever finishes first deletes the other's. Without intermediate storage, the problem disappears by construction.
- **Cleanup on the happy path only.** On error, the Cloud Function deletes neither the temporary file nor the blob; the API also tries to delete in `finally` — double deletion, one of them fails, and the failure is rethrown inside an async handler.
- **Errors that do not cross the boundary.** Business errors leave Python as a string with HTTP 200; the 400/500 handling branches in the front-end are unreachable. Three layers, three error conventions.

### Platform deadlines and numerical reproducibility (amendment of 2026-10-27, still Proposed)

- **Deadlines of the current platforms.** The Cloud Function runs on 1st-gen Cloud Run functions with the Python 3.10 runtime, deprecated on 2026-10-04 and decommissioned on 2027-04-04 — after that the function can no longer be updated and may be disabled. In the Vercel project, the Node.js 18 configured for the front-end has been discontinued and blocks new builds. The migration is no longer just desirable; it has a deadline.
- **Zero-cost requirement** for the deployment, which reopened the choice of platform (see Decision and alternative E).
- **Version spike (2026-10-27).** The `tcc-2023` pipeline was run on the 13 CSVs under several combinations of versions. Result: Python 3.11/3.12, pandas 2.3 and scipy 1.15 reproduce the output **bit for bit**; upgrading **numpy 1.x → 2.x** changes the `k, m, c, n` parameters by up to ~18% (the fitted curve changes by at most ~1.2%), because the unbounded BFGS fit is ill-conditioned and amplifies rounding differences. pandas 3 requires numpy ≥ 2.0.2. This refines the migration steps (see "Execution in steps").

---

## Decision

Unify `CalculoIDF_API` and `CalculoIDF_python` into **a single Python service (FastAPI), deployed as a Vercel Function with the Python runtime, on Vercel's Hobby plan**.

As a consequence:

- The front-end sends the CSV **directly** to the Python service, via `multipart/form-data`.
- **Firebase Storage leaves the synchronous processing path** and, with it, the dependency on Google Cloud: the Cloud Function is shut down at the end of the migration.
- Front-end and backend are deployed on the **same platform** (Vercel), as separate projects, each from its own repository (ADR 0002).
- The `CalculoIDF_API` repository is **archived** (kept read-only as a historical record of the thesis, not deleted).
- The computation core (`process_data.py` … `ventechow.py`) is **not changed** by this decision. Only its wrapper changes: the HTTP edge and the deployment.

### Execution in steps (each verified by the golden files)

1. **Wrap** the pipeline in FastAPI with the 2023 versions pinned, on Python 3.11 → **exact** parity. Runs locally and in CI (Vercel's Python runtime starts at 3.12).
2. **(2a) Upgrade** to Python 3.12, pandas 2.3.3, numpy 1.26.4 and scipy 1.15.3 → **exact** parity (shown in the spike). **First version deployed to Vercel.** Replacing scikit-learn with `numpy.polyfit` is a commit of its own, with a tolerance declared only for the regression (difference ~1e-14).
3. **Decommission** the Node API, Firebase Storage and the Cloud Function; archive `CalculoIDF_API`.

**Outside this decision:** upgrading to numpy 2.x, pandas 3.x and current scipy (formerly "step 2b") **changes numbers** and is therefore not treated as migration: it is a deliberate change (principle I of the constitution), planned together with methodological fix C-01.

---

## Alternatives considered

### A. Keep the current architecture and only update versions

Least immediate effort and no topology regression risk. **Rejected:** it preserves a boundary that does not pay for itself (see Rationale) and makes each of the five planned features be implemented twice or pay for a useless network round trip. The cost comes back with every feature instead of being paid once.

### B. Keep both services but change the transport (send the CSV in the request body, no Storage)

Removes the Firebase dependency and two network round trips. **Rejected:** it fixes the most visible symptom but keeps the boundary that cuts through the concept of "a valid Hidroweb CSV" — the real problem. It would still require two coordinated deployments for every change to the output contract.

### C. Unify in Node, rewriting the computations in JavaScript

Would keep a single JS stack in front-end and backend. **Rejected outright:** the core depends on `scipy` (`norm.ppf`, `gamma.ppf`, `pearsonr`, `minimize`), `pandas` and `numpy`. There is no mature equivalent in JS, and rewriting validated statistical code is the most efficient way to introduce silent errors in a system whose output feeds drainage design.

### D. Unify in Python with FastAPI on Vercel (Python runtime, Hobby plan) — **chosen**

See Rationale.

### E. Unify in Python with FastAPI in a container on Cloud Run — original proposal, rejected on 2026-10-27

Scale to zero, generous limits, a container (the same artifact in CI and in production). The free tier covers this load comfortably, but it **requires a billing account**, and Artifact Registry and Cloud Build are billed separately — zero cost depends on alerts and cleanup policies; it is not guaranteed. It keeps two platforms (Vercel + Google Cloud). **Rejected** because of the zero-cost requirement and principle V (fewer pieces). It remains the natural path if the service ever needs a container (see "When to revisit").

### F. Unify in Python with FastAPI on Render's free plan — rejected

Zero cost, but the service sleeps after 15 minutes without traffic and takes about a minute to wake up — unacceptable for a sporadically used tool, where almost every request would be a *cold start*. It would also keep two platforms.

---

## Rationale

### 1. Every network boundary has to pay for itself

A boundary between services charges a continuous price: latency, partial failure modes, deployment surface, credentials, fragmented observability and cross-repository coordination. This is not an argument against boundaries — it is their price. The criterion used: **does the boundary separate things that (a) change for different reasons, (b) scale differently, (c) have different security or availability requirements, or (d) belong to different teams?**

In this system, all four answers are no. The Node API changes whenever the format of the Python JSON changes — they are not independent axes of change. There are no separate teams and no distinct security requirements. Scale is addressed in item 4.

### 2. The boundary cuts through a concept

Knowledge of the Hidroweb file format (`skiprows=12`, `ISO-8859-1` encoding, `NivelConsistencia`/`Data`/`Maxima` columns) lives in Python. The 2 MB limit and the natural input-validation point live in Node. In other words: the boundary cuts through the single concept of "what a valid Hidroweb CSV is".

That is why the validation feature is awkward today: either the format knowledge is duplicated in Node, or the full round trip to Python is paid to find out that the file was invalid. **When a boundary forces a choice between duplicating knowledge and paying for a useless round trip, it is in the wrong place.**

### 3. Firebase Storage is not being used as storage

Nothing is persisted: the object is deleted right after the computation. It works as **transport** between two services. That is the *claim check* pattern, legitimate when the payload does not fit in the message or when the consumer is asynchronous. Here the payload is capped at 2 MB and the flow is synchronous (request-response). A whole external dependency — with its own credentials, lifecycle and failure mode — is paid for to solve a problem the system does not have.

### 4. Which side survives: the one with the least replaceable dependency

When a boundary collapses, one decides which side absorbs the other. The rule applied: **the side with the least replaceable dependency stays.** Python carries scipy/pandas/numpy; Node's HTTP layer (CORS, *rate limit*, *multipart*, size limit, orchestration) is a *commodity* and exists equally well in any language. The current Node service delivers no capability that FastAPI does not — it does the same thing in another language, which is the signature of an accidental boundary.

The honest argument **for** separation is scale: Python is *CPU-bound* (BFGS optimization) and Node is *I/O-bound*, and separating them would allow independent scaling. The argument is real but does not apply to this load: the computation takes on the order of 1–2 seconds on a small DataFrame, and demand is low. If it ever grows, the correct answer is a **queue with a job ID and polling**, not a language boundary — which would not solve that problem anyway.

### 5. Why FastAPI

- **Pydantic** offers declarative validation at the edge, which is exactly the planned CSV validation feature, and turns the output dictionary hand-built at the end of `ventechow.py` into an explicit contract (*response model*) instead of a convention.
- **Automatic OpenAPI** gives the front-end a typed, versionable contract.
- **`async`** fits the planned external calls well (Hidroweb API, AI provider).

### 6. Why Vercel (amendment of 2026-10-27)

- **Zero cost without a billing account** on the Hobby plan, for this load.
- **One platform instead of three** (Vercel, Render, Google Cloud): one dashboard, one deployment model (push to Git), one place for logs and environment variables — principle V.
- **Fits within the limits** (checked on 2026-10-27; recheck in the plan): Python bundle up to 500 MB uncompressed (the measured dependencies add up to ~243 MB), 2 GB / 1 vCPU, maximum duration of 300 s, request body up to 4.5 MB (the app's limit is 2 MB), Python 3.12–3.14.
- **Scales to zero**, like the Cloud Function.
- FastAPI is loaded as an ASGI application without adaptation; the same app runs locally with Uvicorn.

The honest counter-argument: a Function is not a container. Control over the full environment (the image that runs in CI and in production) is lost. This weighs less than it seems: the dependencies are the same *wheels* pinned in the lockfile, and numpy's instruction dispatch varies with the CPU on any cloud — the numerical guarantee comes from the golden files in CI, not from the platform.

---

## Consequences

### Positive

- One service, one deployment, one set of logs. Tracing an error no longer crosses three systems.
- Two network round trips and one external dependency fewer on the critical path; faster response to the user.
- Input validation becomes possible **at the edge**, next to the format knowledge — without duplication.
- The five planned features are implemented in a single place.
- Smaller credential surface: `FIREBASE_CREDENTIALS`, `FIREBASE_STORAGE_BUCKET_URL` and the Google Cloud credentials cease to exist.
- A single hosting platform for the whole system.

### Negative and accepted costs

- The demonstration of a polyglot architecture, which had some showcase value, is lost.
- There is migration work: rebuilding the HTTP edge (CORS, *rate limit*, size limit, *multipart* handling), configuring the Python project on Vercel and redoing the deployment pipeline.
- The front-end depends on a single endpoint; if the service is down, the whole application is down. Acceptable: today an outage of any of the three already has the same effect, with a higher cumulative probability.
- *CPU-bound* computation and HTTP handling share the same process — mitigable and, if needed, solved with the queue described in item 4.
- Dependence on platform-specific features (entrypoint configuration, Hobby plan limits) and on the platform's terms of use.
- An in-memory *rate limit* applies per Function instance, not globally.

### Risks and mitigation

| Risk | Mitigation |
|---|---|
| Silent numerical regression during the migration | **Blocking prerequisite:** *golden file* tests on the 13 CSVs in `src/csv/` before any structural change |
| A numerical difference attributed to the migration when it comes from the library upgrade | Separate steps; numpy 2 outside the migration (spike of 2026-10-27) |
| Different *multipart* parsing behavior between `multer` and FastAPI | End-to-end integration test with a real upload, including a malformed file and one above the limit |
| High *cold start* (importing pandas and scipy) | Measure after the first deployment (baseline: the current 10-step flow) |
| Bundle above the platform limit | Minimal dependency list; exclude CSVs, tests and fixtures from the deployed bundle |
| Changes to the Hobby plan's limits, prices or terms | Recheck in the plan of 002; alternative E documented; portable app (standard ASGI) |

---

## Prerequisites

This decision **must not be executed before** the *golden file* test suite exists: for each of the 13 CSVs in `CalculoIDF_python/src/csv/`, run the pipeline and freeze the output JSON as the reference. Specified in `specs/001-characterization-tests/spec.md`; the execution of the decision is specified in `specs/002-backend-unification/spec.md` (final folder names are set when the specs are created).

The repository structure after the unification is covered by ADR 0002 (separate repositories; Spec Kit in `CalculoIDF_python`).

The rationale is architectural, not just hygiene: tests anchored on **observable behavior** (CSV in → JSON out), not on internal structure, are what give the freedom to reorganize the structure. They turn a "risky refactoring" into a "verifiable refactoring" — and they are exactly the tests that prove the unification did not change a single number.

---

## Reversibility

A **two-way door**. The computation core is not touched; only the wrapper changes. Reverting means redeploying the Cloud Function (while the runtime still allows it) and reactivating the archived repository — which is why `CalculoIDF_API` is archived rather than deleted. Moving from Vercel to another platform (e.g., Cloud Run) only requires packaging the same ASGI app in a container.

## When to revisit

This decision should be revisited if:

- the computation time grows enough to block requests (e.g., comparing many stations at once) — the answer is then an asynchronous queue, not a new service in another language;
- a genuinely "Node-shaped" need appears: Next.js with API routes, websockets, or an authentication BFF layer;
- the project gets more than one person maintaining distinct parts of the system, creating an independent axis of change that does not exist today;
- the service exceeds the Hobby plan limits (duration, memory, bundle size, request body), the plan's terms stop applying to the project, or the service needs a container — alternative E then becomes the natural choice.
