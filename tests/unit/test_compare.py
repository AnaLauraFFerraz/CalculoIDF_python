import math

import pytest

import compare
from compare import ToleranceNotSetError, compare as run_compare

TOLERANCES = {
    "rules": [
        {"pattern": "parameters.**", "rel_tol": 1e-3, "abs_tol": 0.0},
        {"pattern": "series.values[*]", "rel_tol": 1e-6, "abs_tol": 0.0},
        {"pattern": "ns.*", "rel_tol": 1e-4, "abs_tol": 0.0},
        {"pattern": "**", "rel_tol": 1e-12, "abs_tol": 0.0},
    ]
}


def paths(diffs):
    return [d.path for d in diffs]


@pytest.mark.parametrize("mode", ["exact", "tolerant"])
def test_identical_structures_pass(mode):
    data = {"a": [1, 2.5, {"b": "x"}], "c": None, "d": True, "e": float("nan")}
    assert run_compare(data, data, mode, TOLERANCES) == []


def test_differing_leaf_is_reported_with_its_path():
    expected = {"graph": {"F": [1.0, 2.0, 3.0]}}
    actual = {"graph": {"F": [1.0, 2.5, 3.0]}}
    diffs = run_compare(expected, actual, "exact")
    assert len(diffs) == 1
    assert diffs[0].path == "graph.F[1]"
    assert (diffs[0].expected, diffs[0].actual) == (2.0, 2.5)


@pytest.mark.parametrize("mode", ["exact", "tolerant"])
@pytest.mark.parametrize("expected,actual", [
    (False, []),
    ([], False),
    (1, True),
    (True, 1),
    ("2001", 2001),
    (2001, "2001"),
    (None, 0),
])
def test_type_mismatches_fail_in_both_modes(mode, expected, actual):
    diffs = run_compare({"v": expected}, {"v": actual}, mode, TOLERANCES)
    assert paths(diffs) == ["v"]
    assert diffs[0].rule.startswith("type")


@pytest.mark.parametrize("mode", ["exact", "tolerant"])
def test_list_length_and_order(mode):
    assert paths(run_compare({"v": [1, 2]}, {"v": [1, 2, 3]}, mode, TOLERANCES)) == ["v"]
    assert paths(run_compare({"v": [1, 2]}, {"v": [2, 1]}, mode, TOLERANCES)) == ["v[0]", "v[1]"]


@pytest.mark.parametrize("mode", ["exact", "tolerant"])
def test_extra_and_missing_keys(mode):
    diffs = run_compare({"a": 1, "b": 2}, {"a": 1, "c": 3}, mode, TOLERANCES)
    assert paths(diffs) == ["b", "c"]
    assert [d.rule for d in diffs] == ["key", "key"]


def test_nan_equals_nan_but_not_a_number():
    assert run_compare({"v": float("nan")}, {"v": float("nan")}, "exact") == []
    assert paths(run_compare({"v": float("nan")}, {"v": 1.0}, "exact")) == ["v"]


def test_exact_rejects_one_ulp_that_tolerant_accepts():
    value = 0.1
    neighbour = math.nextafter(value, 1.0)
    assert paths(run_compare({"x": value}, {"x": neighbour}, "exact")) == ["x"]
    assert run_compare({"x": value}, {"x": neighbour}, "tolerant", TOLERANCES) == []


def test_tolerant_uses_the_first_matching_rule():
    expected = {"parameters": {"parameters_1": {"k1": 100.0}}, "other": 100.0}
    actual = {"parameters": {"parameters_1": {"k1": 100.05}}, "other": 100.05}
    diffs = run_compare(expected, actual, "tolerant", TOLERANCES)
    assert paths(diffs) == ["other"]
    assert "**" in diffs[0].rule


def test_pattern_syntax():
    expected = {"series": {"values": [1.0, 1.0]}, "ns": {"parameter_1": 1.0}}
    actual = {"series": {"values": [1.0, 1.0000005]}, "ns": {"parameter_1": 1.00005}}
    assert run_compare(expected, actual, "tolerant", TOLERANCES) == []
    actual = {"series": {"values": [1.0, 1.00001]}, "ns": {"parameter_1": 1.001}}
    assert paths(run_compare(expected, actual, "tolerant", TOLERANCES)) == ["ns.parameter_1", "series.values[1]"]


def test_tolerant_refuses_to_run_with_null_tolerances():
    unset = {"rules": [{"pattern": "parameters.**", "rel_tol": None, "abs_tol": None},
                       {"pattern": "**", "rel_tol": 1e-12, "abs_tol": 0.0}]}
    with pytest.raises(ToleranceNotSetError, match=r"parameters\.\*\*"):
        run_compare({"a": 1}, {"a": 1}, "tolerant", unset)
    assert run_compare({"a": 1}, {"a": 1}, "exact") == []


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        run_compare({}, {}, "approximate")


def test_normalize_converts_numpy_values():
    import numpy as np

    out = compare.normalize({"a": np.float64(1.5), "b": np.array([1, 2]), "c": np.bool_(True), "d": (1, 2)})
    assert out == {"a": 1.5, "b": [1, 2], "c": True, "d": [1, 2]}
    assert type(out["c"]) is bool


def test_reference_fingerprint_matches_the_pins_in_requirements():
    requirements = (compare.REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    ref = compare.REFERENCE_FINGERPRINT
    assert (ref["os"], ref["machine"], ref["python"]) == ("Linux", "x86_64", "3.10")
    for library in compare.LIBRARIES:
        assert f"{library}=={ref[library]}" in requirements


def test_only_the_station_with_removed_outliers_carries_a_marker():
    real = [e for e in compare.series_registry() if e["kind"] == "real"]
    assert len(real) == 13
    marked = {e["id"]: e["marker"] for e in real if e["marker"] is not None}
    assert list(marked) == ["chuvas_C_02346117"]
    assert (marked["chuvas_C_02346117"]["debt"], marked["chuvas_C_02346117"]["fix"]) == ("D-07", "C-03")


def test_fingerprints_match_ignores_python_patch_only():
    base = dict(compare.REFERENCE_FINGERPRINT, python="3.10.12")
    assert compare.fingerprints_match(base, dict(base, python="3.10.22"))
    assert not compare.fingerprints_match(base, dict(base, python="3.11.1"))
    assert not compare.fingerprints_match(base, dict(base, os="Windows"))
    assert not compare.fingerprints_match(base, dict(base, numpy="2.0.0"))


def test_unpinned_kernels_never_count_as_the_reference():
    base = dict(compare.REFERENCE_FINGERPRINT, python="3.10.12")
    assert base["OPENBLAS_CORETYPE"] == "Haswell"
    assert "AVX512F" in base["NPY_DISABLE_CPU_FEATURES"]
    assert compare.fingerprints_match(base, compare.REFERENCE_FINGERPRINT)
    assert not compare.fingerprints_match(base, dict(base, OPENBLAS_CORETYPE=None))
    assert not compare.fingerprints_match(base, dict(base, OPENBLAS_CORETYPE="SkylakeX"))
    assert not compare.fingerprints_match(base, dict(base, NPY_DISABLE_CPU_FEATURES=None))
    legacy = {k: v for k, v in base.items() if k not in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES")}
    assert not compare.fingerprints_match(legacy, base)


def test_fingerprint_reports_the_effective_kernel_variables(monkeypatch):
    monkeypatch.delenv("OPENBLAS_CORETYPE", raising=False)
    assert compare.fingerprint()["OPENBLAS_CORETYPE"] is None
    monkeypatch.setenv("OPENBLAS_CORETYPE", "Haswell")
    assert compare.fingerprint()["OPENBLAS_CORETYPE"] == "Haswell"
