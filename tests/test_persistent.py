import json
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from distributed import Client, LocalCluster
from huggingface_hub import HfApi

from hfdask.cluster import Cluster, boot_cluster
from hfdask.jobs import JobSpec
from hfdask.network import issue_credentials
from hfdask.runner import TOPOLOGY_METADATA


def manifest(job_nodes=1):
    return {
        "cluster_id": "cluster",
        "jobs": [{"namespace": "example", "id": "scheduler-job"}],
        "connection": {
            "schema": 2,
            "persistent": True,
            "job_nodes": job_nodes,
            "scheduler_worker": True,
        },
    }


def test_boot_and_restore():
    api = MagicMock(spec=HfApi)
    api.run_job.side_effect = [MagicMock(id="scheduler"), MagicMock(id="worker")]
    snapshots = []
    cluster = boot_cluster(
        JobSpec("example", "image", workers=2),
        scheduler_worker=True,
        api=api,
        on_submitted=lambda current: snapshots.append(current.manifest()),
    )
    public = json.loads(json.dumps(cluster.manifest()))
    assert public["connection"] == {
        "schema": 2,
        "persistent": True,
        "job_nodes": 2,
        "scheduler_worker": True,
        "startup_timeout": 1200,
        "node_flavors": ["cpu-basic", "cpu-basic"],
        "node_tags": [[], []],
    }
    assert cluster.client_credentials is not None
    assert "PRIVATE KEY" not in json.dumps(public)
    assert len(snapshots[0]["jobs"]) == 1
    assert [call.kwargs["ssh"] for call in api.run_job.call_args_list] == [True, False]
    assert all(
        call.kwargs["network_group"] == f"hfdask-{cluster.id}"
        for call in api.run_job.call_args_list
    )

    restored = Cluster.from_manifest(public, api=api)
    assert restored.client_credentials is None
    api.inspect_job.return_value.status.stage = "CANCELED"
    restored.close(timeout=0)
    assert api.cancel_job.call_count == 2
    assert api.run_job.call_count == 2


def test_boot_rejects_workload():
    with pytest.raises(ValueError, match="entrypoint"):
        boot_cluster(JobSpec("example", "image", "workload:run"))


def test_batch_requires_entrypoint():
    with pytest.raises(ValueError, match="entrypoint"):
        JobSpec("example", "image").command()


def test_client_connect_disconnect_keeps_scheduler(monkeypatch):
    from hfdask.client import connect

    server, credentials = issue_credentials(["server"], client_name="client")
    assert credentials is not None

    @contextmanager
    def no_tunnel(*args, **kwargs):
        yield

    monkeypatch.setattr("hfdask.client._ssh_tunnel", no_tunnel)
    monkeypatch.setattr("hfdask.client._wait_for_running", lambda *args, **kwargs: None)

    with server[0].security() as server_security:
        with LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=False,
            host="127.0.0.1",
            protocol="tls",
            security=server_security,
            dashboard_address=None,
        ) as cluster:
            port = int(cluster.scheduler_address.rsplit(":", 1)[1])
            with Client(
                cluster.scheduler_address,
                security=server_security,
                set_as_default=False,
            ) as coordinator:
                coordinator.set_metadata(TOPOLOGY_METADATA, [1])
            for _ in range(2):
                with connect(
                    manifest(),
                    credentials,
                    timeout=5,
                    local_port=port,
                    api=MagicMock(spec=HfApi),
                ) as client:
                    assert client.direct_to_workers is False
                    assert client.submit(abs, -17).result() == 17
            with Client(
                cluster.scheduler_address,
                security=server_security,
                set_as_default=False,
            ) as client:
                assert client.submit(sum, [1, 2, 3]).result() == 6
            assert cluster.status.name == "running"


def test_connect_rejects_manifest_before_tunnel(monkeypatch):
    from hfdask.client import connect

    tunnel = MagicMock()
    monkeypatch.setattr("hfdask.client._ssh_tunnel", tunnel)
    credentials, _ = issue_credentials(["client"])
    broken = manifest()
    broken["connection"]["schema"] = 1
    with pytest.raises(ValueError):
        with connect(broken, credentials[0]):
            pass
    tunnel.assert_not_called()


def test_ssh_tunnel_forwards_loopback_scheduler(monkeypatch):
    from hfdask.client import _ssh_tunnel

    process = MagicMock()
    process.poll.side_effect = [None, None]
    process.wait.return_value = 0
    monkeypatch.setattr("hfdask.client.shutil.which", lambda name: "/usr/bin/ssh")
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("hfdask.client.subprocess.Popen", popen)

    connection = MagicMock()
    connection.__enter__.return_value = connection
    connection.__exit__.return_value = None
    monkeypatch.setattr(
        "hfdask.client.socket.create_connection", lambda *args, **kwargs: connection
    )

    with _ssh_tunnel("job-id", local_port=18886, timeout=1, identity_file=None):
        pass
    command = popen.call_args.args[0]
    assert "127.0.0.1:18886:127.0.0.1:8786" in command
    assert command[-1] == "job-id@ssh.hf.jobs"
    process.terminate.assert_called_once()


def test_wait_for_running_rejects_terminal_job():
    from hfdask.client import _wait_for_running

    api = MagicMock(spec=HfApi)
    api.inspect_job.return_value.status.stage = "ERROR"
    with pytest.raises(RuntimeError, match="ERROR"):
        _wait_for_running(api, "example", "job", 0)
