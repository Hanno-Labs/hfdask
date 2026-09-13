from unittest.mock import MagicMock

import pytest

from hfdask.runner import run, run_script


def test_real_multiprocess_cluster(monkeypatch):
    monkeypatch.setattr(
        "hfdask.hardware.detect",
        lambda: {"cpu_threads": 3, "ram_bytes": 2**30, "gpus": []},
    )
    assert run("tests.workloads:calculate", workers=2, memory_limit="0", kwargs={"count": 5}) == [
        0,
        1,
        4,
        9,
        16,
    ]


def test_cleanup_on_failure(monkeypatch):
    cluster = MagicMock()
    client = MagicMock()
    monkeypatch.setattr("hfdask.runner.LocalCluster", cluster)
    monkeypatch.setattr("hfdask.runner.Client", client)
    monkeypatch.setattr(
        "hfdask.hardware.detect",
        lambda: {"cpu_threads": 5, "ram_bytes": 2**30, "gpus": []},
    )
    with pytest.raises(RuntimeError, match="intentional"):
        run("tests.workloads:fail")
    cluster.return_value.__exit__.assert_called_once()
    client.return_value.__exit__.assert_called_once()
    assert cluster.call_args.kwargs["host"] == "127.0.0.1"
    assert cluster.call_args.kwargs["dashboard_address"] is None
    assert cluster.call_args.kwargs["n_workers"] == 4
    assert cluster.call_args.kwargs["threads_per_worker"] == 1


def test_single_job_requires_one_scheduler_and_one_worker_core(monkeypatch):
    monkeypatch.setattr(
        "hfdask.hardware.detect",
        lambda: {"cpu_threads": 1, "ram_bytes": 2**30, "gpus": []},
    )
    with pytest.raises(RuntimeError, match="at least two"):
        run("tests.workloads:calculate")


def test_script_uses_default_distributed_client(tmp_path):
    from distributed import Client, LocalCluster

    output = tmp_path / "result.txt"
    script = tmp_path / "job.py"
    script.write_text(
        "import dask\n"
        "from distributed import get_client, get_worker\n"
        "from pathlib import Path\n"
        "@dask.delayed\n"
        "def task():\n"
        "    return get_worker().address\n"
        "if __name__ == '__main__':\n"
        f"    Path({str(output)!r}).write_text(task().compute())\n"
    )
    with (
        LocalCluster(
            n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None
        ) as cluster,
        Client(cluster, set_as_default=False) as client,
    ):
        run_script(client, str(script))
        assert output.read_text() in client.scheduler_info()["workers"]


@pytest.mark.parametrize("code", [0, 1, "failure"])
def test_script_exit_and_process_state_restored(tmp_path, code):
    import sys

    script = tmp_path / "job.py"
    script.write_text(f"raise SystemExit({code!r})")
    argv, paths = sys.argv, sys.path.copy()
    if code == 0:
        run_script(MagicMock(), str(script))
    else:
        with pytest.raises(RuntimeError, match="Script exited"):
            run_script(MagicMock(), str(script))
    assert sys.argv is argv
    assert sys.path == paths


def test_invalid_entrypoint():
    with pytest.raises(ValueError):
        run("not a module")
