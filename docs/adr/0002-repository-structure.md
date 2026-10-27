# ADR 0002 — Repository structure: keep separate repositories, with Spec Kit in the backend repository

- **Status:** Accepted (2026-10-27)
- **Date:** 2026-10-20 (opened) · 2026-10-27 (decided)
- **Authorship:** project maintainer
- **Scope:** `CalculoIDF_python`, `CalculoIDF_front`, `CalculoIDF_API`
- **Related:** ADR 0001 (backend unification, `Proposed`); adoption of SDD with GitHub Spec Kit

---

## Context

### How the repositories are today

Three public GitHub repositories, one per layer of the thesis system:

| Repository | Role | Deployment | Thesis state |
|---|---|---|---|
| `CalculoIDF_front` | React 18 / Create React App | Vercel, through Git integration (push to `main` → production) | tag `tcc-2023` → `f3bc7bb` |
| `CalculoIDF_API` | Node/Express (intermediary) | Render | tag `tcc-2023` → `06b7474` |
| `CalculoIDF_python` | Computation core | Google Cloud Function (1st gen, Python 3.10), manual deployment | tag `tcc-2023` → `bb1ab14` |

On 2026-10-27 the state of each repository described in the thesis was preserved with a `tcc-2023` tag and an immutable release (thesis PDF and attestation attached). Those tags exist only in the original repositories.

### Why the decision is needed now

1. **Spec Kit assumes specs live next to the code.** The flow creates one branch per feature, writes `specs/NNN-…/`, and `/speckit-implement` changes the code in the same repository. Where to run `specify init` is the first decision of the cycle.
2. **Of the eight drafted specs, five (004 to 008) change backend and front-end together**, almost always through the API contract.
3. **Phase 1 of the project is backend only.** The current cycle covers specs 001 (characterization tests) and 002 (backend unification), driven also by platform deadlines: the Cloud Function's Python 3.10 runtime was deprecated on 2026-10-04 and will be decommissioned on 2027-04-04. In 002, the only front-end change is the API URL (an environment variable). The specs that cross both layers belong to phase 2.
4. **There is a single maintainer**, and no other contributors are expected.

## Decision

**Keep the repositories separate. Spec Kit is initialized only in `CalculoIDF_python`, which becomes the project's reference repository.**

As a consequence:

- `CalculoIDF_python` holds `.specify/` (with the constitution), `specs/` and `docs/adr/`. Every spec of the project lives here, including those that affect the front-end.
- `CalculoIDF_front` stays independent and **without Spec Kit** in this phase. Its README points to `CalculoIDF_python` as the source of specs and decisions.
- `CalculoIDF_API` follows ADR 0001: it stays as it is until the decommissioning planned in spec 002 and is then archived.
- **No repository is merged, renamed or deleted.** The `tcc-2023` tags and the immutable releases stay where they are.

Rules for work that crosses repositories (phase 2 onward):

- A spec that affects the front-end has a **"Front-end impact"** section; the front-end PR references the spec through an absolute link (`CalculoIDF_python/specs/NNN-…`).
- The **API contract** is the boundary between the repositories: the backend publishes the OpenAPI document and the front-end generates its types from it (spec 003).
- Contract changes follow **expand and then contract**: the backend publishes the new version additively (or versioned), the front-end migrates, and only then is the old version removed. That way each repository can be deployed on its own at any time.

## Alternatives considered

### A. Monorepo (`backend/`, `frontend/`, `specs/`, `docs/adr/` at the root) — rejected for now

**The best argument in favor (steelman).** By ADR 0001's own boundary test, the separation between repositories is **accidental** in the long run: front-end and backend change together whenever the contract changes, they do not scale for reasons a repository solves, they have no distinct security requirements and they do not belong to different teams. In a monorepo, one constitution and one specs folder apply to everything; a single PR changes contract, backend and front-end atomically; `/speckit-implement` reaches both layers; a feature's history lives in one place. Vercel and the backend platform deploy from subfolders. And the decision gets more expensive with every spec created before it.

**Why it was rejected now.** The benefit of the monorepo shows up when features cross both layers — phase 2. The cost is paid in phase 1: creating a new repository, importing the histories (with `git filter-repo` the SHAs are rewritten and the "thesis" commit no longer exists under the same SHA in the new repository; with `git subtree` they are preserved, but the history becomes interleaved), reconfiguring the Vercel project for a subfolder and adjusting the backend pipeline. In phase 1, the only point of contact with the front-end is an environment variable. Paying now a cost whose return only comes later, in a short cycle, is the wrong trade — as long as the reevaluation is written down (see "When to revisit").

### C. Separate code repositories + a dedicated specs/docs repository — rejected

"Neutral" specs, but it breaks the Spec Kit flow (spec, branch and code in different places), creates a third repository to keep in sync and is the option with the highest risk of documentation diverging from code.

### D. Spec Kit in each repository, duplicated constitution — rejected for now

Each repository would work on its own, but two identical constitutions would have to be kept in sync by hand and cross-cutting features would be split into two specs. With no front-end work in phase 1, the gain is nil. It may be revisited in phase 2 if the chosen option proves insufficient.

## Rationale

### The four-question test, applied to repositories

| Question | Phase 1 (001, 002) | Phase 2 (003 to 008) |
|---|---|---|
| Do they change for different reasons? | **Yes, in practice:** the backend migration does not change the front-end, except for one environment variable | **No:** five of the specs change both sides through the contract |
| Do they scale differently? | Irrelevant for repositories: each layer already deploys independently | Same |
| Different security/availability requirements? | No | No |
| Different teams? | No (single maintainer) | No |

The honest reading: **in the long run the separation is accidental** — the monorepo argument is correct. What changes the decision is the *when*. In phase 1, the two layers effectively do not change together, and the restructuring would have an immediate cost and a future benefit. So the decision is to postpone the merge until the last responsible moment, with the trigger written down.

### Why Spec Kit lives in the backend

- It is where phase 1 work happens: specs 001 and 002 change only `CalculoIDF_python` (and, at the end of 002, shut down `CalculoIDF_API`).
- The backend holds the domain knowledge (computation core) and will own the API contract.
- `docs/adr/` has belonged to this repository since ADR 0001.

## Consequences

### Positive

- **Zero migration cost.** No history is rewritten; the `tcc-2023` tags and the immutable releases remain valid, with the original SHAs.
- Vercel, Render and the Cloud Function keep working as they are throughout phase 1.
- The Spec Kit cycle (branch per feature, `/speckit-implement`) works without adaptation for specs 001 and 002.
- Two-way door: the merge remains available later, through the same procedure.

### Negative and accepted costs

- In phase 2, cross-cutting features will require **two coordinated PRs**, and `/speckit-implement` run in the backend does not change the front-end without extra configuration.
- The front-end has no constitution of its own; applying the principles to it depends on cross-references.
- The history of a cross-cutting feature is split between two repositories.

### Risks and mitigation

| Risk | Mitigation |
|---|---|
| Spec and front-end implementation diverge | Mandatory "Front-end impact" section in the specs; front-end PR references the spec; front-end types generated from OpenAPI |
| A contract change breaks the front-end between two deployments | Expand and then contract; contract versioning (v0 → v1 in 004) |
| `/speckit-implement` does not reach the front-end | In phase 2, give Claude Code access to the front-end folder in the same session (confirm the additional-directories feature with `claude --help` before relying on it), or adopt option D |
| The merge gets more expensive with every spec | Explicit reevaluation trigger at the start of phase 2 (below) |
| The original repositories get deleted by mistake, losing the `tcc-2023` tags | Explicit constraint: the three repositories are **never deleted**; `CalculoIDF_API` is archived, not deleted |

## Prerequisites and steps

1. Add `.gitattributes` to `CalculoIDF_python` (D-41) and renormalize line endings, in a commit of its own.
2. Version `docs/adr/0001-…` and `docs/adr/0002-…` in the remote repository.
3. Run `specify init` at the root of `CalculoIDF_python` (confirm flags with `specify init --help`).
4. Ratify the constitution with `/speckit-constitution` from the draft.
5. Update the `CalculoIDF_front` README to point to `CalculoIDF_python` as the source of specs and decisions.

## Reversibility

A **two-way door**. A later merge into a monorepo can be done by importing the existing repositories into a new one (with `git subtree add` without `--squash`, preserving the SHAs) and moving `specs/`, `docs/adr/` and `.specify/` to the root. The cost grows only slightly with each spec, because the specs will already be in a single place.

## When to revisit

This decision should be revisited:

- **at the start of phase 2**, before creating the first spec that changes front-end and backend together (003 or 004);
- if cross-cutting features routinely require coordinated PRs in both repositories;
- if the backend comes to be deployed on the same platform as the front-end, which would make a single project natural (Vercel was chosen as the backend target in the 2026-10-27 amendment to ADR 0001 — weigh this at the start of phase 2);
- if someone else starts contributing to the project.
