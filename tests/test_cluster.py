import base64
import json
from unittest.mock import MagicMock

import pytest
from huggingface_hub import HfApi

from hfdask.cluster import Cluster, LaunchError, WorkerGroup, submit_cluster
from hfdask.jobs import JobSpec


def test_heterogeneous_groups():
    api = MagicMock(spec=HfApi)
    api.run_job.return_value.id = "job"
    submit_cluster(
        JobSpec("example", "image", "workload:run"),
        worker_groups=[WorkerGroup("cpu-performance", 2), WorkerGroup("a100-large", 3)],
        api=api,
    )
    assert [call.kwargs["flavor"] for call in api.run_job.call_args_list] == [
        "cpu-basic",
        "cpu-performance",
        "cpu-performance",
        "a100-large",
        "a100-large",
        "a100-large",
    ]
    command = api.run_job.call_args.kwargs["command"]
    config = json.loads(command[command.index("--cluster") + 1])
    assert config["job_nodes"] == 6
    assert config["node_tags"] == [[], [], [], [], [], []]


@pytest.mark.parametrize("colocated", [False, True])
@pytest.mark.parametrize(
    "scheduler,worker,expected",
    [
        (None, None, ["cpu-upgrade", "cpu-upgrade"]),
        ("cpu-basic", None, ["cpu-basic", "cpu-upgrade"]),
        (None, "cpu-performance", ["cpu-upgrade", "cpu-performance"]),
        ("cpu-basic", "cpu-performance", ["cpu-basic", "cpu-performance"]),
    ],
)
def test_hardware_overrides(colocated, scheduler, worker, expected):
    api = MagicMock(spec=HfApi)
    api.run_job.return_value.id = "job"
    submit_cluster(
        JobSpec("example", "image", "workload:run", flavor="cpu-upgrade"),
        scheduler_worker=colocated,
        scheduler_flavor=scheduler,
        worker_flavor=worker,
        api=api,
    )
    actual = [call.kwargs["flavor"] for call in api.run_job.call_args_list]
    assert actual == expected[:1] + expected[1:] * (2 - int(colocated))


@pytest.mark.parametrize("override", [{"scheduler_flavor": ""}, {"worker_flavor": " "}])
def test_empty_hardware_rejected_before_submission(override):
    api = MagicMock(spec=HfApi)
    with pytest.raises(ValueError, match="flavor"):
        submit_cluster(JobSpec("example", "image", "workload:run"), api=api, **override)
    api.run_job.assert_not_called()


@pytest.mark.parametrize("workers", [1, 2])
def test_scheduler_worker_reduces_jobs(workers):
    api = MagicMock(spec=HfApi)
    api.run_job.return_value.id = "job"
    cluster = submit_cluster(
        JobSpec("example", "image", "workload:run", workers=workers),
        scheduler_worker=True,
        api=api,
    )
    assert len(cluster.jobs) == workers
    for call in api.run_job.call_args_list:
        assert "--scheduler-worker" in call.kwargs["command"]


def test_cluster_network_group_mtls_secrets_and_aliases():
    api = MagicMock(spec=HfApi)
    api.run_job.return_value.id = "job"
    cluster = submit_cluster(JobSpec("example", "image", "workload:run", workers=2), api=api)
    assert len(cluster.jobs) == 3
    groups = {call.kwargs["network_group"] for call in api.run_job.call_args_list}
    assert groups == {f"hfdask-{cluster.id}"}
    certificates = set()
    for index, call in enumerate(api.run_job.call_args_list):
        kwargs = call.kwargs
        assert kwargs["network_aliases"] == (
            ["node-0", "scheduler"] if index == 0 else [f"node-{index}"]
        )
        assert kwargs["ssh"] is False
        assert kwargs["labels"]["hfdask-node"] == str(index)
        assert set(kwargs["secrets"]) == {
            "HFDASK_TLS_CA",
            "HFDASK_TLS_CERT",
            "HFDASK_TLS_KEY",
        }
        decoded = base64.b64decode(kwargs["secrets"]["HFDASK_TLS_CERT"])
        assert decoded.startswith(b"-----BEGIN CERTIFICATE-----")
        certificates.add(decoded)
    assert len(certificates) == 3
    assert "PRIVATE KEY" not in json.dumps(cluster.manifest())


def test_partial_launch_retains_handles():
    api = MagicMock(spec=HfApi)
    api.run_job.side_effect = [MagicMock(id="first"), ConnectionError("uncertain")]
    with pytest.raises(LaunchError) as caught:
        submit_cluster(JobSpec("example", "image", "workload:run", workers=1), api=api)
    assert caught.value.cluster.jobs[0].id == "first"
    api.cancel_job.assert_not_called()


def test_wait_verifies_cleanup():
    scheduler = MagicMock()
    worker = MagicMock()
    scheduler.status.return_value = "COMPLETED"
    worker.status.return_value = "CANCELED"
    assert Cluster("cluster", [scheduler, worker]).wait(timeout=0) == "COMPLETED"
    worker.cancel.assert_called_once()
    worker.status.assert_called_once()


def test_cleanup_timeout_not_success():
    job = MagicMock()
    job.status.return_value = "RUNNING"
    with pytest.raises(TimeoutError, match="unverified"):
        Cluster("cluster", [job]).close(timeout=0)
