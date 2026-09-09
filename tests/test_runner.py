from unittest.mock import MagicMock

import pytest

from hfdask.runner import run


def test_real_multiprocess_cluster():
    assert run("tests.workloads:calculate", workers=2, memory_limit="0",
               kwargs={"count": 5}) == [0, 1, 4, 9, 16]


def test_cleanup_on_failure(monkeypatch):
    cluster = MagicMock()
    client = MagicMock()
    monkeypatch.setattr("hfdask.runner.LocalCluster", cluster)
    monkeypatch.setattr("hfdask.runner.Client", client)
    with pytest.raises(RuntimeError, match="intentional"):
        run("tests.workloads:fail")
    cluster.return_value.__exit__.assert_called_once()
    client.return_value.__exit__.assert_called_once()
    assert cluster.call_args.kwargs["host"] == "127.0.0.1"
    assert cluster.call_args.kwargs["dashboard_address"] is None


def test_invalid_entrypoint():
    with pytest.raises(ValueError):
        run("not a module")
