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


STRUCTURE_RULE = {"pattern": "fit.**", "kind": "structure", "reason": "D-01: ill-conditioned fit"}
NUMERIC_DEFAULT = {"pattern": "**", "rel_tol": 1e-12, "abs_tol": 0.0}
WITH_STRUCTURE = {"rules": [STRUCTURE_RULE, NUMERIC_DEFAULT]}
NAN = float("nan")
INF = float("inf")


def fit(value):
    return {"fit": {"k": value}}


@pytest.mark.parametrize("expected,actual", [(100.0, 855.25), (1, 2.5), (0.0, -3.0)])
def test_structure_rule_accepts_different_finite_numbers(expected, actual):
    first = {"fit": {"k": expected, "values": [expected, 2.0]}}
    second = {"fit": {"k": actual, "values": [actual, 7.0]}}
    assert run_compare(first, second, "tolerant", WITH_STRUCTURE) == []


@pytest.mark.parametrize("expected,actual", [
    (1.0, True), (0.0, False), (1, True), (True, 1.0), (False, 0),
    (1.0, "1.0"), ("1.0", 1.0),
    (1.0, [1.0]), ([1.0], 1.0),
    (1.0, None), (None, 1.0),
])
def test_structure_rule_still_checks_that_a_number_is_a_number(expected, actual):
    diffs = run_compare(fit(expected), fit(actual), "tolerant", WITH_STRUCTURE)
    assert paths(diffs) == ["fit.k"]
    assert diffs[0].rule.startswith("type")


@pytest.mark.parametrize("expected,actual", [(NAN, 1.0), (1.0, NAN)])
def test_structure_rule_distinguishes_nan_from_a_number(expected, actual):
    diffs = run_compare(fit(expected), fit(actual), "tolerant", WITH_STRUCTURE)
    assert paths(diffs) == ["fit.k"]
    assert "structure" in diffs[0].rule


def test_structure_rule_accepts_nan_against_nan():
    assert run_compare(fit(NAN), fit(NAN), "tolerant", WITH_STRUCTURE) == []


@pytest.mark.parametrize("expected,actual", [
    (INF, 1.0), (1.0, INF), (-INF, 1.0), (1.0, -INF), (INF, -INF), (-INF, INF),
    (NAN, INF), (INF, NAN), (NAN, -INF), (-INF, NAN),
])
def test_structure_rule_reports_a_non_finite_mismatch(expected, actual):
    diffs = run_compare(fit(expected), fit(actual), "tolerant", WITH_STRUCTURE)
    assert paths(diffs) == ["fit.k"]
    assert "structure" in diffs[0].rule and "non-finite mismatch" in diffs[0].rule


@pytest.mark.parametrize("value", [INF, -INF])
def test_structure_rule_accepts_equal_infinities(value):
    assert run_compare(fit(value), fit(value), "tolerant", WITH_STRUCTURE) == []


def test_structure_rule_still_checks_list_lengths_and_keys():
    expected = {"fit": {"a": [1.0, 2.0], "b": 1.0}}
    actual = {"fit": {"a": [1.0], "c": 2.0}}
    diffs = run_compare(expected, actual, "tolerant", WITH_STRUCTURE)
    assert paths(diffs) == ["fit.a", "fit.b", "fit.c"]
    assert [d.rule for d in diffs] == ["length", "key", "key"]


def test_structure_rule_does_not_relax_strings_and_booleans():
    expected = {"fit": {"name": "log-normal", "flag": True}}
    actual = {"fit": {"name": "Gumbel", "flag": False}}
    assert paths(run_compare(expected, actual, "tolerant", WITH_STRUCTURE)) == ["fit.flag", "fit.name"]


def test_exact_mode_ignores_the_rule_kind():
    diffs = run_compare(fit(100.0), fit(100.0000001), "exact", WITH_STRUCTURE)
    assert paths(diffs) == ["fit.k"]
    assert diffs[0].rule == "exact"


def test_exact_mode_does_not_validate_structure_rules():
    invalid = {"rules": [{"pattern": "fit.**", "kind": "structure"},
                         {"pattern": "**", "kind": "fuzzy", "rel_tol": None, "abs_tol": None}]}
    assert run_compare(fit(1.0), fit(1.0), "exact", invalid) == []


@pytest.mark.parametrize("kind_fields", [{}, {"kind": "numeric"}])
def test_numeric_rules_behave_the_same_with_or_without_the_kind(kind_fields):
    tolerances = {"rules": [dict(pattern="**", rel_tol=1e-3, abs_tol=0.0, **kind_fields)]}
    assert run_compare({"k": 100.0}, {"k": 100.05}, "tolerant", tolerances) == []
    assert paths(run_compare({"k": 100.0}, {"k": 101.0}, "tolerant", tolerances)) == ["k"]


@pytest.mark.parametrize("reason", [None, "", "   ", "\t\n"])
def test_tolerant_refuses_a_structure_rule_without_a_reason(reason):
    rule = {"pattern": "parameters.**", "kind": "structure"}
    if reason is not None:
        rule["reason"] = reason
    tolerances = {"rules": [rule, NUMERIC_DEFAULT]}
    with pytest.raises(ToleranceNotSetError, match=r"reason.*parameters\.\*\*|parameters\.\*\*.*reason"):
        run_compare({"a": 1}, {"a": 1}, "tolerant", tolerances)


def test_tolerant_refuses_an_unknown_rule_kind():
    tolerances = {"rules": [{"pattern": "fit.**", "kind": "fuzzy", "rel_tol": 1e-3, "abs_tol": 0.0},
                            NUMERIC_DEFAULT]}
    with pytest.raises(ValueError, match=r"kind.*fuzzy|fuzzy.*kind"):
        run_compare({"a": 1}, {"a": 1}, "tolerant", tolerances)


@pytest.mark.parametrize("tolerance_fields", [{"rel_tol": 1e-3}, {"abs_tol": 0.0}, {"rel_tol": None, "abs_tol": None}])
def test_tolerant_refuses_a_structure_rule_that_carries_tolerances(tolerance_fields):
    rule = dict(STRUCTURE_RULE, **tolerance_fields)
    with pytest.raises(ValueError, match=r"fit\.\*\*.*(rel_tol|abs_tol)"):
        run_compare({"a": 1}, {"a": 1}, "tolerant", {"rules": [rule, NUMERIC_DEFAULT]})


def test_tolerant_names_the_pattern_of_a_numeric_rule_without_tolerances():
    tolerances = {"rules": [{"pattern": "fit.**", "kind": "numeric"}, NUMERIC_DEFAULT]}
    with pytest.raises(ToleranceNotSetError, match=r"fit\.\*\*"):
        run_compare({"a": 1}, {"a": 1}, "tolerant", tolerances)


def test_first_match_wins_when_a_structure_rule_precedes_a_numeric_one():
    tolerances = {"rules": [STRUCTURE_RULE, NUMERIC_DEFAULT]}
    expected = {"fit": {"k": 1.0}, "other": 1.0}
    actual = {"fit": {"k": 2.0}, "other": 2.0}
    assert paths(run_compare(expected, actual, "tolerant", tolerances)) == ["other"]


def test_first_match_wins_when_a_numeric_rule_precedes_a_structure_one():
    tolerances = {"rules": [{"pattern": "fit.k", "rel_tol": 1e-12, "abs_tol": 0.0}, STRUCTURE_RULE]}
    expected = {"fit": {"k": 1.0, "m": 1.0}}
    actual = {"fit": {"k": 2.0, "m": 2.0}}
    assert paths(run_compare(expected, actual, "tolerant", tolerances)) == ["fit.k"]
