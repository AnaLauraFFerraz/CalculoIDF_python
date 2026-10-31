import json
import sys
from pathlib import Path

import pytest

import compare

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import update_golden  # noqa: E402

REFERENCE = dict(compare.REFERENCE_FINGERPRINT, python="3.10")
OTHER_ENVIRONMENT = dict(compare.REFERENCE_FINGERPRINT, os="Windows", machine="AMD64")


def make_entry(series_id, marker=None, sha="aa"):
    return {"id": series_id, "kind": "real", "path": Path(f"{series_id}.csv"),
            "input_path": f"src/csv/{series_id}.csv", "sha256": sha, "marker": marker}


@pytest.fixture
def world(monkeypatch, tmp_path):
    """Two fake series with controllable results, running in the 'reference' environment."""
    state = {
        "entries": [make_entry("A"), make_entry("B")],
        "results": {"A": ("object", {"a": 1.0, "b": [1.0, 2.0], "c": "x"}),
                    "B": ("string", "Dados não são sufientes para completar a análise")},
    }
    monkeypatch.setattr(compare, "series_registry", lambda: state["entries"])
    monkeypatch.setattr(compare, "run_series", lambda path: state["results"][Path(path).stem])
    monkeypatch.setattr(compare, "fingerprint", lambda: dict(REFERENCE, python="3.10.12"))
    state["root"] = tmp_path / "golden"
    state["run"] = lambda *argv: update_golden.run(
        list(argv), golden_root=state["root"], reference_fingerprint=REFERENCE)
    return state


def golden_files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()) \
        if root.exists() else []


@pytest.mark.parametrize("argv", [[], ["--reason", ""], ["--reason", "   "]])
def test_missing_or_blank_reason_exits_2_and_writes_nothing(world, argv):
    assert world["run"](*argv) == 2
    assert golden_files(world["root"]) == []


def test_unknown_series_exits_2(world, capsys):
    assert world["run"]("--reason", "r", "--series", "Z") == 2
    assert "valid ids" in capsys.readouterr().err
    assert golden_files(world["root"]) == []


def test_non_reference_environment_without_flag_exits_3(world):
    code = update_golden.run(["--reason", "r"], golden_root=world["root"],
                             reference_fingerprint=OTHER_ENVIRONMENT)
    assert code == 3
    assert golden_files(world["root"]) == []


def test_allow_non_reference_writes_only_to_out_and_has_no_changelog(world, tmp_path):
    scratch = tmp_path / "scratch"
    code = update_golden.run(["--reason", "r", "--allow-non-reference", "--out", str(scratch)],
                             golden_root=world["root"], reference_fingerprint=OTHER_ENVIRONMENT)
    assert code == 0
    assert golden_files(world["root"]) == []
    assert golden_files(scratch) == ["real/A.json", "real/B.json"]


def test_allow_non_reference_requires_out(world):
    code = update_golden.run(["--reason", "r", "--allow-non-reference"], golden_root=world["root"],
                             reference_fingerprint=OTHER_ENVIRONMENT)
    assert code == 2


def test_allow_non_reference_refuses_out_inside_the_golden_directory(world, monkeypatch):
    monkeypatch.setattr(compare, "GOLDEN_ROOT", world["root"])
    code = update_golden.run(["--reason", "r", "--allow-non-reference", "--out", str(world["root"] / "x")],
                             golden_root=world["root"], reference_fingerprint=OTHER_ENVIRONMENT)
    assert code == 2
    assert golden_files(world["root"]) == []


def test_initial_generation_writes_files_and_a_changelog_entry(world, capsys):
    assert world["run"]("--reason", "initial characterization") == 0
    assert golden_files(world["root"]) == ["CHANGELOG.md", "real/A.json", "real/B.json"]

    text = (world["root"] / "real" / "A.json").read_bytes().decode("utf-8")
    assert text.endswith("}\n") and "\r" not in text
    golden = json.loads(text)
    assert golden["metadata"]["series_id"] == "A"
    assert golden["metadata"]["input_sha256"] == "aa"
    assert golden["metadata"]["environment"]["python"] == "3.10.12"
    assert golden["result_kind"] == "object"

    changelog = (world["root"] / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "> initial characterization" in changelog
    assert "| A | initial generation | - | - |" in changelog
    assert "| Series | Changed leaves | Max abs Δ | Max rel Δ |" in capsys.readouterr().out


def test_unchanged_result_leaves_files_byte_identical_and_adds_no_entry(world, capsys):
    world["run"]("--reason", "first")
    before = {p: (world["root"] / p).read_bytes() for p in golden_files(world["root"])}
    capsys.readouterr()

    assert world["run"]("--reason", "second") == 0

    assert {p: (world["root"] / p).read_bytes() for p in golden_files(world["root"])} == before
    assert "No golden file changed" in capsys.readouterr().out


def test_changed_result_rewrites_only_that_file_and_logs_the_table(world, capsys):
    world["run"]("--reason", "first")
    untouched = (world["root"] / "real" / "B.json").read_bytes()
    capsys.readouterr()

    world["results"]["A"] = ("object", {"a": 1.5, "b": [1.0, 2.0], "c": "x"})
    assert world["run"]("--reason", "deliberate change\nsecond line") == 0

    assert (world["root"] / "real" / "B.json").read_bytes() == untouched
    assert json.loads((world["root"] / "real" / "A.json").read_text(encoding="utf-8"))["result"]["a"] == 1.5
    changelog = (world["root"] / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog.count("\n## ") == 2
    row = "| A | 1 | 5.000e-01 | 5.000e-01 |"
    assert row in changelog
    assert row in capsys.readouterr().out
    assert "- `a`" in changelog
    assert "> deliberate change\n> second line" in changelog
    assert "| B |" not in changelog.split("deliberate change")[1]


def test_marker_change_alone_counts_as_a_change(world):
    world["run"]("--reason", "first")
    world["entries"][0] = make_entry("A", marker={"status": "suspected", "debt": "D-07", "fix": "C-03",
                                                  "note": "n"})
    assert world["run"]("--reason", "mark A") == 0
    golden = json.loads((world["root"] / "real" / "A.json").read_text(encoding="utf-8"))
    assert golden["metadata"]["marker"]["debt"] == "D-07"
    changelog = (world["root"] / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "- `<marker>`" in changelog


def test_changed_leaves_and_deltas_count_every_differing_number(world):
    world["run"]("--reason", "first")
    world["results"]["A"] = ("object", {"a": 2.0, "b": [1.0, 2.5], "c": "y"})
    world["run"]("--reason", "three leaves")
    changelog = (world["root"] / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "| A | 3 | 1.000e+00 | 1.000e+00 |" in changelog
    assert all(f"- `{p}`" in changelog for p in ("a", "b[1]", "c"))
