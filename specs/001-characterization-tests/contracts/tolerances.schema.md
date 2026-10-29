# Contract: tolerance file

Path: `tests/golden/tolerances.json`. Used only in `tolerant` mode (any environment whose
fingerprint differs from the golden metadata). Values are set by the measurement task
(research R-05); the shape below is fixed now, the numbers are not.

```json
{
  "measured_at": "YYYY-MM-DD",
  "reference_environment": { "os": "Linux", "python": "3.10", "numpy": "1.24.3", "...": "..." },
  "measured_environment": { "os": "Windows", "python": "3.10", "numpy": "1.24.3", "...": "..." },
  "safety_factor": 10,
  "rules": [
    { "pattern": "parameters.**",                          "rel_tol": null, "abs_tol": null, "measured_max_abs": null, "measured_max_rel": null },
    { "pattern": "intensity_graph_data_*.i_calculated[*]", "rel_tol": null, "abs_tol": null, "measured_max_abs": null, "measured_max_rel": null },
    { "pattern": "intensity_graph_data_*.regression.*",    "rel_tol": null, "abs_tol": null, "measured_max_abs": null, "measured_max_rel": null },
    { "pattern": "mean_relative_errors.*",                 "rel_tol": null, "abs_tol": null, "measured_max_abs": null, "measured_max_rel": null },
    { "pattern": "ns.*",                                   "rel_tol": null, "abs_tol": null, "measured_max_abs": null, "measured_max_rel": null },
    { "pattern": "graph_data.**",                          "rel_tol": null, "abs_tol": null, "measured_max_abs": null, "measured_max_rel": null },
    { "pattern": "**",                                     "rel_tol": 1e-12, "abs_tol": 0.0, "measured_max_abs": null, "measured_max_rel": null }
  ]
}
```

- Pattern syntax: dot-separated path; `*` matches one key segment, `[*]` any list index, `**`
  any suffix. First match wins.
- Tolerances apply to numbers only. Keys, list lengths, order, types, booleans and strings are
  always compared exactly, in both modes.
- `null` values are placeholders until the measurement task fills them; the suite refuses to
  run in `tolerant` mode while any rule has a `null` tolerance (clear error message).
