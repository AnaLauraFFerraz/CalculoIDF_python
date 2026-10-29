# Contract: golden file

Path: `tests/golden/{real|synthetic}/<series-id>.json`. UTF-8, LF, keys sorted, indent 2,
trailing newline. Written only by `scripts/update_golden.py`.

```json
{
  "metadata": {
    "series_id": "chuvas_C_01844000_CV",
    "input_path": "src/csv/chuvas_C_01844000_CV.csv",
    "input_sha256": "<64 hex>",
    "generated_at": "2026-10-30T12:00:00Z",
    "environment": {
      "os": "Linux",
      "machine": "x86_64",
      "python": "3.10.x",
      "numpy": "1.24.3",
      "pandas": "2.0.1",
      "scipy": "1.10.1",
      "scikit-learn": "1.3.0"
    },
    "marker": null
  },
  "result_kind": "object",
  "result": { "...": "full normalized output of main.main (v0 contract)" }
}
```

- `result_kind = "object"`: `result` is the v0 success object, normalized (numpy scalars →
  Python scalars, arrays → lists). Types are preserved: `empty_years` is `false` or a list;
  `year_range` values are strings; `sample_size_above_30_years` is a boolean.
- `result_kind = "string"`: `result` is the decoded business message, e.g.
  `"Dados não são sufientes para completar a análise"` (typo preserved).
- `result_kind = "exception"`: `result` is `{"type": "<qualified name>", "message": "<str>"}`.
- `marker`: `null` or `{"status": "known-defect" | "suspected", "debt": "D-02", "fix": "C-02",
  "note": "<one line>"}`.
- The `environment` fingerprint used to choose `exact` mode compares `os`, `machine`, Python
  `major.minor` and the four library versions.
- `NaN`, `Infinity` and `-Infinity` are accepted in `result`: they are written as the tokens
  Python's `json` module emits (`NaN`, `Infinity`, `-Infinity`), which it reads back exactly.
  The file is therefore JSON as Python reads it, not strict RFC 8259 JSON.
- `generated_at` is the time of the last **content** change. A file is rewritten only when
  `result_kind`, `result`, `metadata.input_sha256` or `metadata.marker` change; a regeneration
  that would change only `generated_at` (or `environment`) leaves the file byte-for-byte.
