import re

import compare

POST_FIT_PATTERNS = {
    "parameters.**",
    "intensity_graph_data_*.i_calculated[*]",
    "intensity_graph_data_*.regression.*",
    "mean_relative_errors.*",
    "ns.*",
}


def committed():
    return compare.load_tolerances()


def test_committed_rules_pass_the_tolerant_mode_validation():
    # Runs the same compilation path as the suite: unknown kinds, null tolerances and structure
    # rules without a reason raise, whatever the comparison mode of the current run.
    assert compare.compare({}, {}, "tolerant", committed()) == []


def test_every_rule_records_its_measurement():
    for rule in committed()["rules"]:
        assert isinstance(rule["measured_max_abs"], (int, float)), rule["pattern"]
        assert isinstance(rule["measured_max_rel"], (int, float)), rule["pattern"]


def test_post_fit_groups_are_structure_rules_and_the_rest_numeric():
    kinds = {rule["pattern"]: rule.get("kind", "numeric") for rule in committed()["rules"]}
    assert {p for p, kind in kinds.items() if kind == "structure"} == POST_FIT_PATTERNS
    assert kinds["graph_data.**"] == kinds["**"] == "numeric"


def test_the_default_rule_is_last_and_matches_everything():
    rules = committed()["rules"]
    assert rules[-1]["pattern"] == "**"
    assert rules[-1]["rel_tol"] > 0


def test_measurement_metadata_is_present():
    tolerances = committed()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", tolerances["measured_at"])
    assert tolerances["safety_factor"] == 10
    for key in ("reference_environment", "measured_environment"):
        assert set(tolerances[key]) == set(compare.REFERENCE_FINGERPRINT), key


def test_the_recorded_reference_environment_is_the_reference_and_the_measured_one_is_not():
    tolerances = committed()
    assert compare.fingerprints_match(tolerances["reference_environment"], compare.REFERENCE_FINGERPRINT)
    assert not compare.fingerprints_match(tolerances["measured_environment"], compare.REFERENCE_FINGERPRINT)
