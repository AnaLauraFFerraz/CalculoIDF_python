import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "characterization"))

import bootstrap  # noqa: E402

bootstrap.apply()


def _blocked(*args, **kwargs):
    raise RuntimeError("network access is blocked in the characterization suite")


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
