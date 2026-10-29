# Contract: tolerance file

Path: `tests/golden/tolerances.json`. Used only in `tolerant` mode (any environment whose
fingerprint differs from the golden metadata). Values are set by the measurement task
(research R-05); the shape below is fixed now, the numbers are not.

Two rule kinds (ADR 0003):

- **numeric** (default): numbers are compared with `math.isclose(rel_tol, abs_tol)`. Used for
  pre-fit fields, whose cross-environment differences are rounding noise.
- **structure**: only types, keys, list lengths and NaN-ness are compared; values are not.
  Used for post-fit fields, whose cross-environment differences are amplified by the
  ill-conditioned fit (D-01). A mandatory, non-empty `reason` records why.

```json
{
  "measured_at": "YYYY-MM-DD",
  "reference_environment": { "os": "Linux", "python": "3.10", "numpy": "1.24.3", "...": "..." },
  "measured_environment": { "os": "Windows", "python": "3.10", "numpy": "1.24.3", "...": "..." },
  "safety_factor": 10,
  "rules": [
    { "pattern": "parameters.**",                          "kind": "structure", "reason": "D-01: ...", "measured_max_abs": null, "measured_max_rel": null },
    { "pattern": "intensity_graph_data_*.i_calculated[*]", "kind": "structure", "reason": "D-01: ...", "measured_max_abs": null, "measured_max_rel": null },
    { "pattern": "intensity_graph_data_*.regression.*",    "kind": "structure", "reason": "D-01: ...", "measured_max_abs": null, "measured_max_rel": null },
    { "pattern": "mean_relative_errors.*",                 "kind": "structure", "reason": "D-01: ...", "measured_max_abs": null, "measured_max_rel": null },
    { "pattern": "ns.*",                                   "kind": "structure", "reason": "D-01: ...", "measured_max_abs": null, "measured_max_rel": null },
    { "pattern": "graph_data.**",                          "rel_tol": null, "abs_tol": null, "measured_max_abs": null, "measured_max_rel": null },
    { "pattern": "**",                                     "rel_tol": 1e-12, "abs_tol": 0.0, "measured_max_abs": null, "measured_max_rel": null }
  ]
}
```

- Pattern syntax: dot-separated path; `*` matches one key segment, `[*]` any list index, `**`
  any suffix. First match wins.
- `kind` is `"numeric"` or `"structure"`; when absent it is `"numeric"`. A numeric rule has
  `rel_tol` and `abs_tol`; a structure rule has `reason` and no tolerances.
- Tolerances apply to numbers only. Keys, list lengths, order, types, booleans and strings are
  always compared exactly, in both modes. A `structure` rule relaxes only the comparison of
  numeric **values**: a number must still be a number (not a boolean, string or list), and `NaN`
  must still be `NaN` (a `NaN` against a number is a difference).
- `measured_max_abs` / `measured_max_rel` are recorded for every rule, structure rules included,
  as the evidence behind the rule.
- `null` tolerances in a numeric rule are placeholders until the measurement task fills them;
  the suite refuses to run in `tolerant` mode while any numeric rule has a `null` tolerance, or
  while any structure rule has a missing or empty `reason` (clear error message).
- In `exact` mode (the reference environment) the rule kinds are ignored: every number is
  compared with `==`.
