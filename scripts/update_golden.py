"""Regenerate golden files. Requires a recorded reason; see contracts/update-golden-cli.md."""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "characterization"))

import compare  # noqa: E402


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


def _build_golden(entry, environment):
    kind, result = compare.run_series(entry["path"])
    return {
        "metadata": {
            "series_id": entry["id"],
            "input_path": entry["input_path"],
            "input_sha256": entry["sha256"],
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "environment": environment,
            "marker": entry["marker"],
        },
        "result_kind": kind,
        "result": result,
    }


def run(argv, golden_root=compare.GOLDEN_ROOT, reference_fingerprint=compare.REFERENCE_FINGERPRINT):
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exit_:
        return exit_.code

    if not args.reason.strip():
        print("error: --reason must not be empty", file=sys.stderr)
        return 2

    registry = {entry["id"]: entry for entry in compare.series_registry()}
    selected = args.series or list(registry)
    unknown = [series_id for series_id in selected if series_id not in registry]
    if unknown:
        print(f"error: unknown series {unknown}; valid ids: {sorted(registry)}", file=sys.stderr)
        return 2

    environment = compare.fingerprint()
    if not compare.fingerprints_match(environment, reference_fingerprint):
        if not args.allow_non_reference:
            print("error: this is not the reference environment "
                  f"({reference_fingerprint}); golden files are generated in CI. "
                  "Use --allow-non-reference --out <dir> for a local experiment.", file=sys.stderr)
            return 3
        if args.out is None:
            print("error: --allow-non-reference requires --out", file=sys.stderr)
            return 2
        golden_root = args.out

    for series_id in selected:
        entry = registry[series_id]
        _write(compare.golden_path(entry, golden_root), _build_golden(entry, environment))
        print(f"wrote {entry['kind']}/{series_id}.json")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
