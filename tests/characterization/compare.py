"""Normalization, comparison and series registry of the characterization suite."""
import bootstrap

bootstrap.apply()

import contextlib  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
from importlib import metadata  # noqa: E402
from typing import Any, NamedTuple  # noqa: E402

import numpy as np  # noqa: E402

REPO_ROOT = bootstrap.REPO_ROOT
REAL_CSV_DIR = REPO_ROOT / "src" / "csv"
SYNTHETIC_CSV_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic"
GOLDEN_ROOT = REPO_ROOT / "tests" / "golden"
TOLERANCES_PATH = GOLDEN_ROOT / "tolerances.json"

LIBRARIES = ("numpy", "pandas", "scipy", "scikit-learn")
MODES = ("exact", "tolerant")
RULE_KINDS = ("numeric", "structure")
MODE_VARIABLE = "CALCULOIDF_COMPARE_MODE"


def _pinned_versions():
    pins = {}
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for library in LIBRARIES:
        match = re.search(rf"^{re.escape(library)}==(\S+)\s*$", text, re.MULTILINE | re.IGNORECASE)
        pins[library] = match.group(1)
    return pins


REFERENCE_FINGERPRINT = {
    "os": "Linux",
    "machine": "x86_64",
    "python": "3.10",
    **_pinned_versions(),
    **bootstrap.KERNEL_VARIABLES,
}


def fingerprint():
    return {
        "os": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        **{library: metadata.version(library) for library in LIBRARIES},
        **{variable: os.environ.get(variable) for variable in bootstrap.KERNEL_VARIABLES},
    }


def _major_minor(version):
    return ".".join(version.split(".")[:2])


def fingerprints_match(first, second):
    if first["os"] != second["os"] or first["machine"] != second["machine"]:
        return False
    if _major_minor(first["python"]) != _major_minor(second["python"]):
        return False
    if any(first.get(variable) != second.get(variable) for variable in bootstrap.KERNEL_VARIABLES):
        return False
    return all(first[library] == second[library] for library in LIBRARIES)


def _to_builtin(value):
    if isinstance(value, dict):
        return {(k.item() if isinstance(k, np.generic) else k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize(value):
    return json.loads(json.dumps(_to_builtin(value), sort_keys=True))


def run_series(csv_path):
    """Run `main.main` and return `(result_kind, result)` with the result normalized."""
    import main as pipeline

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = pipeline.main(str(csv_path))
    except Exception as exc:
        return "exception", {"type": f"{type(exc).__module__}.{type(exc).__qualname__}", "message": str(exc)}
    if isinstance(result, str):
        result = json.loads(result)
    result = normalize(result)
    return ("string" if isinstance(result, str) else "object"), result


# Markers of known or suspected defects, keyed by series id. C-01 (D-01, BFGS ignores the
# declared bounds) affects every fitted result, so it has no per-station marker.
MARKERS = {
    "chuvas_C_02346117": {
        "status": "suspected",
        "debt": "D-07",
        "fix": "C-03",
        "note": "4 low outliers removed without recomputing the statistics",
    },
    "S5": {
        "status": "known-defect",
        "debt": "D-02",
        "fix": "C-02",
        "note": "starts in October: the complete first water year is dropped",
    },
    "S6": {
        "status": "known-defect",
        "debt": "D-02",
        "fix": "C-02",
        "note": "ends in September: the complete last water year is dropped",
    },
    "S7": {
        "status": "known-defect",
        "debt": "D-02",
        "fix": "C-02",
        "note": "10 complete water years trimmed to 8: insufficient data",
    },
    "S8": {
        "status": "suspected",
        "debt": "D-13",
        "fix": None,
        "note": "pandas 3 Copy-on-Write turns fillna(inplace=True) into a no-op, so the empty "
                "water year reaches the fit as NaN",
    },
}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(kind, path):
    return {
        "id": path.stem,
        "kind": kind,
        "path": path,
        "input_path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha256(path),
        "marker": MARKERS.get(path.stem),
    }


def series_registry():
    entries = [_entry("real", p) for p in sorted(REAL_CSV_DIR.glob("*.csv"))]
    entries += [_entry("synthetic", p) for p in sorted(SYNTHETIC_CSV_DIR.glob("S*.csv"))]
    return entries


class GoldenMissingError(FileNotFoundError):
    pass


def golden_path(entry, golden_root=GOLDEN_ROOT):
    return golden_root / entry["kind"] / f"{entry['id']}.json"


def load_golden(entry, golden_root=GOLDEN_ROOT):
    path = golden_path(entry, golden_root)
    if not path.exists():
        raise GoldenMissingError(
            f"no golden file for series {entry['id']!r} at {path}; "
            "generate it with scripts/update_golden.py")
    return json.loads(path.read_text(encoding="utf-8"))


class ToleranceNotSetError(RuntimeError):
    pass


def load_tolerances(path=TOLERANCES_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def select_mode(golden_environment):
    forced = os.environ.get(MODE_VARIABLE)
    if forced:
        if forced not in MODES:
            raise ValueError(f"{MODE_VARIABLE} must be one of {MODES}, got {forced!r}")
        return forced
    return "exact" if fingerprints_match(fingerprint(), golden_environment) else "tolerant"


def _pattern_regex(pattern):
    parts = []
    for token in re.split(r"(\*\*|\[\*\]|\*)", pattern):
        if token == "**":
            parts.append(".*")
        elif token == "[*]":
            parts.append(r"\[\d+\]")
        elif token == "*":
            parts.append(r"[^.\[\]]*")
        else:
            parts.append(re.escape(token))
    return re.compile("^" + "".join(parts) + "$")


def _compile_rules(tolerances):
    null_tolerances, missing_reasons = [], []
    for rule in tolerances["rules"]:
        pattern, kind = rule["pattern"], rule.get("kind", "numeric")
        if kind not in RULE_KINDS:
            raise ValueError(
                f"unknown kind {kind!r} in the tolerance rule for pattern {pattern!r}; "
                f"expected one of {RULE_KINDS}")
        if kind == "structure":
            if "rel_tol" in rule or "abs_tol" in rule:
                raise ValueError(f"the structure rule for pattern {pattern!r} must not carry rel_tol or abs_tol")
            reason = rule.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                missing_reasons.append(pattern)
        elif rule.get("rel_tol") is None or rule.get("abs_tol") is None:
            null_tolerances.append(pattern)
    if null_tolerances:
        raise ToleranceNotSetError(
            "tolerant mode needs measured tolerances, but these rules in tests/golden/tolerances.json "
            f"are still null: {', '.join(null_tolerances)}. Run scripts/measure_env_diff.py (task T017-T018).")
    if missing_reasons:
        raise ToleranceNotSetError(
            "structure rules in tests/golden/tolerances.json need a non-empty reason, "
            f"but none was given for: {', '.join(missing_reasons)}")
    return [(_pattern_regex(rule["pattern"]), rule) for rule in tolerances["rules"]]


class Diff(NamedTuple):
    path: str
    expected: Any
    actual: Any
    rule: str


_MISSING = "<missing>"


def _kind(value):
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if value is None:
        return "null"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _both_nan(first, second):
    return isinstance(first, float) and isinstance(second, float) and math.isnan(first) and math.isnan(second)


def _compare_numbers(expected, actual, path, mode, rules, diffs):
    if _both_nan(expected, actual) or expected == actual:
        return
    if mode == "exact":
        diffs.append(Diff(path, expected, actual, "exact"))
        return
    for regex, rule in rules:
        if regex.match(path):
            if rule.get("kind", "numeric") == "structure":
                if not (math.isfinite(expected) and math.isfinite(actual)):
                    diffs.append(Diff(path, expected, actual, f"structure {rule['pattern']}: non-finite mismatch"))
            elif not math.isclose(expected, actual, rel_tol=rule["rel_tol"], abs_tol=rule["abs_tol"]):
                label = f"tolerant {rule['pattern']} rel_tol={rule['rel_tol']} abs_tol={rule['abs_tol']}"
                diffs.append(Diff(path, expected, actual, label))
            return
    diffs.append(Diff(path, expected, actual, "tolerant: no matching rule"))


def _walk(expected, actual, path, mode, rules, diffs):
    shown = path or "<root>"
    kind_expected, kind_actual = _kind(expected), _kind(actual)
    if kind_expected != kind_actual:
        diffs.append(Diff(shown, expected, actual, f"type {kind_expected} != {kind_actual}"))
    elif kind_expected == "object":
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else key
            if key not in actual:
                diffs.append(Diff(child, expected[key], _MISSING, "key"))
            elif key not in expected:
                diffs.append(Diff(child, _MISSING, actual[key], "key"))
            else:
                _walk(expected[key], actual[key], child, mode, rules, diffs)
    elif kind_expected == "list":
        if len(expected) != len(actual):
            diffs.append(Diff(shown, f"length {len(expected)}", f"length {len(actual)}", "length"))
            return
        for index, (e, a) in enumerate(zip(expected, actual)):
            _walk(e, a, f"{path}[{index}]", mode, rules, diffs)
    elif kind_expected == "number":
        _compare_numbers(expected, actual, shown, mode, rules, diffs)
    elif expected != actual:
        diffs.append(Diff(shown, expected, actual, "exact"))


def compare(expected, actual, mode, tolerances=None):
    """Return the list of `Diff`s between two normalized results; empty means equal."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    rules = _compile_rules(tolerances) if mode == "tolerant" else None
    diffs = []
    _walk(expected, actual, "", mode, rules, diffs)
    return diffs
