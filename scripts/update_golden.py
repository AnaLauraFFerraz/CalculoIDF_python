"""Regenerate golden files. Requires a recorded reason; see contracts/update-golden-cli.md."""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "characterization"))

import compare  # noqa: E402

CHANGELOG_HEADER = (
    "# Golden files changelog\n\n"
    "Append-only. One entry per run of `scripts/update_golden.py` that changed at least one "
    "golden file.\n"
)


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason", required=True, help="why the golden files are being (re)generated")
    parser.add_argument("--series", nargs="+", metavar="ID", help="series ids; default: all")
    parser.add_argument("--allow-non-reference", action="store_true",
                        help="run outside the reference environment; requires --out")
    parser.add_argument("--out", type=Path, help="scratch directory used with --allow-non-reference")
    return parser


def serialize(golden):
    return json.dumps(golden, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def _write(path, golden):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(serialize(golden))


def _build_golden(entry, environment, generated_at):
    kind, result = compare.run_series(entry["path"])
    return {
        "metadata": {
            "series_id": entry["id"],
            "input_path": entry["input_path"],
            "input_sha256": entry["sha256"],
            "generated_at": generated_at,
            "environment": environment,
            "marker": entry["marker"],
        },
        "result_kind": kind,
        "result": result,
    }


def _content_key(golden):
    return json.dumps(
        [golden["result_kind"], golden["result"], golden["metadata"]["input_sha256"],
         golden["metadata"]["marker"]], sort_keys=True)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _describe_change(old, new):
    paths = []
    if old["result_kind"] != new["result_kind"]:
        paths.append("<result_kind>")
    if old["metadata"]["input_sha256"] != new["metadata"]["input_sha256"]:
        paths.append("<input_sha256>")
    if old["metadata"]["marker"] != new["metadata"]["marker"]:
        paths.append("<marker>")

    diffs = compare.compare(old["result"], new["result"], "exact")
    paths += [d.path for d in diffs]
    max_abs = max_rel = 0.0
    for d in diffs:
        if _is_number(d.expected) and _is_number(d.actual) \
                and math.isfinite(d.expected) and math.isfinite(d.actual):
            delta = abs(d.actual - d.expected)
            max_abs = max(max_abs, delta)
            max_rel = max(max_rel, delta / abs(d.expected) if d.expected else math.inf)
    if not paths:
        paths.append("<result>")
    return {"leaves": len(paths), "paths": paths, "max_abs": max_abs, "max_rel": max_rel}


def _format_delta(value):
    return "0" if value == 0 else f"{value:.3e}"


def _format_table(changes):
    lines = ["| Series | Changed leaves | Max abs Δ | Max rel Δ |", "|---|---|---|---|"]
    for series_id, summary in changes:
        if summary is None:
            lines.append(f"| {series_id} | initial generation | - | - |")
        else:
            lines.append(f"| {series_id} | {summary['leaves']} | "
                         f"{_format_delta(summary['max_abs'])} | {_format_delta(summary['max_rel'])} |")
    return "\n".join(lines)


def _trigger():
    server, repository, run_id = (os.environ.get(v) for v in
                                  ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    if server and repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return "local (reference environment)"


def _append_changelog(golden_root, timestamp, reason, environment, changes, table):
    path = golden_root / "CHANGELOG.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = [
        f"\n## {timestamp}\n",
        f"- **Series**: {', '.join(series_id for series_id, _ in changes)}",
        f"- **Environment**: {', '.join(f'{k}={v}' for k, v in environment.items())}",
        f"- **Trigger**: {_trigger()}",
        "- **Reason**:",
        "",
        *(f"> {line}" if line else ">" for line in reason.splitlines()),
        "",
        table,
        "",
        "Changed fields:",
        "",
    ]
    for series_id, summary in changes:
        paths = ["initial generation"] if summary is None else summary["paths"]
        entry += [f"<details><summary>{series_id}</summary>", ""]
        entry += [f"- `{p}`" for p in paths]
        entry += ["", "</details>", ""]
    text = "\n".join(entry)
    if not path.exists():
        text = CHANGELOG_HEADER + text
    with open(path, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def run(argv, golden_root=None, reference_fingerprint=None):
    golden_root = compare.GOLDEN_ROOT if golden_root is None else golden_root
    reference_fingerprint = compare.REFERENCE_FINGERPRINT if reference_fingerprint is None \
        else reference_fingerprint
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exit_:
        return exit_.code

    reason = args.reason.strip()
    if not reason:
        print("error: --reason must not be empty", file=sys.stderr)
        return 2

    registry = {entry["id"]: entry for entry in compare.series_registry()}
    selected = args.series or list(registry)
    unknown = [series_id for series_id in selected if series_id not in registry]
    if unknown:
        print(f"error: unknown series {unknown}; valid ids: {sorted(registry)}", file=sys.stderr)
        return 2

    environment = compare.fingerprint()
    scratch = False
    if not compare.fingerprints_match(environment, reference_fingerprint):
        if not args.allow_non_reference:
            print("error: this is not the reference environment "
                  f"({reference_fingerprint}); golden files are generated in CI. "
                  "Use --allow-non-reference --out <dir> for a local experiment.", file=sys.stderr)
            return 3
        if args.out is None:
            print("error: --allow-non-reference requires --out", file=sys.stderr)
            return 2
        if args.out.resolve() == compare.GOLDEN_ROOT.resolve() \
                or compare.GOLDEN_ROOT.resolve() in args.out.resolve().parents:
            print("error: --out must not be inside tests/golden/", file=sys.stderr)
            return 2
        golden_root, scratch = args.out, True

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    changes = []
    for series_id in selected:
        entry = registry[series_id]
        golden = _build_golden(entry, environment, timestamp)
        path = compare.golden_path(entry, golden_root)
        summary = None
        if not scratch and path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if _content_key(existing) == _content_key(golden):
                continue
            summary = _describe_change(existing, golden)
        _write(path, golden)
        print(f"wrote {entry['kind']}/{series_id}.json")
        changes.append((series_id, summary))

    if scratch:
        return 0
    if not changes:
        print("No golden file changed; no changelog entry added.")
        return 0

    table = _format_table(changes)
    print(table)
    _append_changelog(golden_root, timestamp, reason, environment, changes, table)
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
