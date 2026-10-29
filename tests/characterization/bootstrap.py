"""Process setup shared by the pytest suite and the maintenance scripts.

Must run before numpy is imported (BLAS thread variables) and before `main` is imported
(the import-time `storage.Client()` in `gcs_utils.py`).
"""
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class _BlockedClient:
    """Stands in for `storage.Client`: any use fails loudly."""

    def __getattr__(self, name):
        raise RuntimeError(
            f"Cloud Storage is not available in the characterization suite (storage_client.{name})")


def _install_storage_stub():
    try:
        import google.cloud as google_cloud
    except ImportError:
        google = sys.modules.setdefault("google", types.ModuleType("google"))
        google_cloud = types.ModuleType("google.cloud")
        google.cloud = google_cloud
        sys.modules["google.cloud"] = google_cloud

    stub = types.ModuleType("google.cloud.storage")
    stub.Client = lambda *args, **kwargs: _BlockedClient()
    sys.modules["google.cloud.storage"] = stub
    google_cloud.storage = stub


def apply():
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    _install_storage_stub()
