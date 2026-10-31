import pytest

import compare

SERIES = compare.series_registry()
MAX_REPORTED_DIFFS = 50


def _report(series_id, diffs):
    lines = [f"series {series_id}: {len(diffs)} difference(s)"]
    for d in diffs[:MAX_REPORTED_DIFFS]:
        lines.append(f"  path={d.path} expected={d.expected!r} actual={d.actual!r} rule={d.rule}")
    if len(diffs) > MAX_REPORTED_DIFFS:
        lines.append(f"  ... and {len(diffs) - MAX_REPORTED_DIFFS} more")
    return "\n".join(lines)


@pytest.mark.golden
@pytest.mark.parametrize("entry", SERIES, ids=[entry["id"] for entry in SERIES])
def test_matches_golden(entry):
    try:
        golden = compare.load_golden(entry)
    except compare.GoldenMissingError as exc:
        pytest.fail(str(exc), pytrace=False)

    expected_sha = golden["metadata"]["input_sha256"]
    assert entry["sha256"] == expected_sha, (
        f"series {entry['id']}: input changed ({entry['input_path']} sha256 {entry['sha256']}, "
        f"golden was generated from {expected_sha})")

    mode = compare.select_mode(golden["metadata"]["environment"])
    tolerances = compare.load_tolerances() if mode == "tolerant" else None

    kind, result = compare.run_series(entry["path"])
    diffs = []
    if kind != golden["result_kind"]:
        diffs.append(compare.Diff("<result_kind>", golden["result_kind"], kind, "exact"))
    diffs += compare.compare(golden["result"], result, mode, tolerances)

    assert not diffs, f"[{mode} mode]\n" + _report(entry["id"], diffs)
