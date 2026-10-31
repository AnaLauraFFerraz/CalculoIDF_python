"""Measure how far this environment is from the golden files; see contracts/update-golden-cli.md.

Runs the pipeline here and compares with the golden files without tolerances, grouped by the
field patterns of tests/golden/tolerances.json. Writes nothing under tests/golden/.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "characterization"))

import compare  # noqa: E402


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_leaves(value, path=""):
    if isinstance(value, dict):
        for key in value:
            yield from _numeric_leaves(value[key], f"{path}.{key}" if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _numeric_leaves(item, f"{path}[{index}]")
    elif _is_number(value):
        yield path


def _group_of(path, groups):
    for pattern, regex in groups:
        if regex.match(path):
            return pattern
    return "(no group)"


def _measure(entry, groups):
    """Return `(stats, failures)`; stats maps group -> {leaves, differing, max_abs, max_rel}."""
    try:
        golden = compare.load_golden(entry)
    except compare.GoldenMissingError as exc:
        return {}, [f"{entry['id']}: {exc}"]

    failures = []
    if golden["metadata"]["input_sha256"] != entry["sha256"]:
        failures.append(f"{entry['id']}: input changed (sha256 differs from the golden metadata)")
    kind, result = compare.run_series(entry["path"])
    if kind != golden["result_kind"]:
        failures.append(f"{entry['id']}: result_kind {golden['result_kind']!r} -> {kind!r}")

    stats = {}
    for path in _numeric_leaves(golden["result"]):
        group = stats.setdefault(_group_of(path, groups),
                                 {"leaves": 0, "differing": 0, "max_abs": 0.0, "max_rel": 0.0})
        group["leaves"] += 1

    for diff in compare.compare(golden["result"], result, "exact"):
        if _is_number(diff.expected) and _is_number(diff.actual):
            group = stats[_group_of(diff.path, groups)]
            group["differing"] += 1
            if math.isfinite(diff.expected) and math.isfinite(diff.actual):
                delta = abs(diff.actual - diff.expected)
                group["max_abs"] = max(group["max_abs"], delta)
                group["max_rel"] = max(group["max_rel"], delta / abs(diff.expected) if diff.expected else math.inf)
            else:
                failures.append(f"{entry['id']}: {diff.path}: non-finite value {diff.expected!r} -> {diff.actual!r}")
        else:
            failures.append(f"{entry['id']}: {diff.path}: {diff.rule}: {diff.expected!r} -> {diff.actual!r}")
    return stats, failures


def _fmt(value):
    return "0" if value == 0 else f"{value:.3e}"


def build_report(selected, groups):
    lines = ["# Environment difference measurement", ""]
    lines.append(f"- **Measured environment**: {json.dumps(compare.fingerprint(), sort_keys=True)}")
    environments = set()
    per_series, failures = {}, []
    for entry in selected:
        stats, series_failures = _measure(entry, groups)
        per_series[entry["id"]] = stats
        failures += series_failures
        try:
            environments.add(json.dumps(compare.load_golden(entry)["metadata"]["environment"], sort_keys=True))
        except compare.GoldenMissingError:
            pass
    for environment in sorted(environments):
        lines.append(f"- **Golden environment**: {environment}")
    lines += ["", "## Per field group and series", "",
              "| Field group | Series | Numeric leaves | Differing leaves | Max abs Δ | Max rel Δ |",
              "|---|---|---|---|---|---|"]
    totals = {}
    for pattern, _ in groups:
        for series_id, stats in per_series.items():
            if pattern in stats:
                s = stats[pattern]
                lines.append(f"| `{pattern}` | {series_id} | {s['leaves']} | {s['differing']} | "
                             f"{_fmt(s['max_abs'])} | {_fmt(s['max_rel'])} |")
                t = totals.setdefault(pattern, {"leaves": 0, "differing": 0, "max_abs": 0.0, "max_rel": 0.0})
                t["leaves"] += s["leaves"]
                t["differing"] += s["differing"]
                t["max_abs"] = max(t["max_abs"], s["max_abs"])
                t["max_rel"] = max(t["max_rel"], s["max_rel"])
    lines += ["", "## Maximum per field group (all selected series)", "",
              "| Field group | Numeric leaves | Differing leaves | Max abs Δ | Max rel Δ |",
              "|---|---|---|---|---|"]
    for pattern, t in totals.items():
        lines.append(f"| `{pattern}` | {t['leaves']} | {t['differing']} | {_fmt(t['max_abs'])} | {_fmt(t['max_rel'])} |")
    lines += ["", "## Non-numeric differences and other failures", ""]
    lines += [f"- {failure}" for failure in failures] if failures else ["None."]
    return "\n".join(lines) + "\n", failures


def run(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", nargs="+", metavar="ID", help="series ids; default: all")
    parser.add_argument("--out", type=Path, help="write the Markdown report to this file")
    args = parser.parse_args(argv)

    registry = {entry["id"]: entry for entry in compare.series_registry()}
    selected_ids = args.series or list(registry)
    unknown = [series_id for series_id in selected_ids if series_id not in registry]
    if unknown:
        print(f"error: unknown series {unknown}; valid ids: {sorted(registry)}", file=sys.stderr)
        return 2

    patterns = [rule["pattern"] for rule in compare.load_tolerances()["rules"]]
    groups = [(pattern, compare._pattern_regex(pattern)) for pattern in patterns]
    report, failures = build_report([registry[i] for i in selected_ids], groups)

    if args.out:
        args.out.write_text(report, encoding="utf-8", newline="\n")
        print(f"report written to {args.out}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(report)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
