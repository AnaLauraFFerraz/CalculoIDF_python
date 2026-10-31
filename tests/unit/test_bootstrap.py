import os
import subprocess
import sys

import bootstrap

EXPECTED_VARIABLES = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_CORETYPE": "Haswell",
    "NPY_DISABLE_CPU_FEATURES": "AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL",
}


def test_apply_sets_the_five_variables(monkeypatch):
    for variable in EXPECTED_VARIABLES:
        monkeypatch.setenv(variable, "other")
    bootstrap.apply()
    assert {variable: os.environ[variable] for variable in EXPECTED_VARIABLES} == EXPECTED_VARIABLES


def test_variables_are_set_before_numpy_is_imported_and_a_numpy_warning_is_not_fatal():
    code = (
        "import sys, os; sys.path.insert(0, sys.argv[1]); import bootstrap; bootstrap.apply(); "
        "assert 'numpy' not in sys.modules; import numpy; "
        "print(os.environ['OPENBLAS_CORETYPE'], os.environ['NPY_DISABLE_CPU_FEATURES'])"
    )
    env = {k: v for k, v in os.environ.items() if k not in EXPECTED_VARIABLES}
    done = subprocess.run([sys.executable, "-W", "always", "-c", code, str(bootstrap.REPO_ROOT / "tests" / "characterization")],
                          env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["Haswell", EXPECTED_VARIABLES["NPY_DISABLE_CPU_FEATURES"]]
