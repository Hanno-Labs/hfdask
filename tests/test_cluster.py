from unittest.mock import MagicMock

import pytest
from huggingface_hub import HfApi

from hfdask.cluster import Cluster, Identity, LaunchError, WorkerGroup, submit_cluster
from hfdask.jobs import JobSpec


def test_heterogeneous_groups(monkeypatch):
    import json

    monkeypatch.setattr(Identity, "public_id", lambda self: self.secret.hex())
    api = MagicMock(spec=HfApi)
    api.run_job.return_value.id = "job"
    keys = [Identity(bytes([i]) * 32) for i in range(6)]
    submit_cluster(
        JobSpec("example", "image", "workload:run"),
        keys,
        public_relays=True,
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
    assert json.loads(command[command.index("--mesh") + 1])["hardware_detection"]


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
def test_hardware_overrides(monkeypatch, colocated, scheduler, worker, expected):
    monkeypatch.setattr(Identity, "public_id", lambda self: self.secret.hex())
    api = MagicMock(spec=HfApi)
    api.run_job.return_value.id = "job"
    keys = [Identity(bytes([i]) * 32) for i in range(3 - int(colocated))]
    submit_cluster(
        JobSpec("example", "image", "workload:run", flavor="cpu-upgrade"),
        keys,
        public_relays=True,
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
        submit_cluster(
            JobSpec("example", "image", "workload:run"), [], public_relays=True, api=api, **override
        )
    api.run_job.assert_not_called()


@pytest.mark.parametrize("workers", [1, 2])
def test_scheduler_worker_reduces_jobs(monkeypatch, workers):
    monkeypatch.setattr(Identity, "public_id", lambda self: self.secret.hex())
    api = MagicMock(spec=HfApi)
    api.run_job.return_value.id = "job"
    keys = [Identity(bytes([i]) * 32) for i in range(workers)]
    cluster = submit_cluster(
        JobSpec("example", "image", "workload:run", workers=workers),
        keys,
        public_relays=True,
        scheduler_worker=True,
        api=api,
    )
    assert len(cluster.jobs) == workers
    for call in api.run_job.call_args_list:
        assert "--scheduler-worker" in call.kwargs["command"]


def test_cluster_secrets_and_ownership(monkeypatch):
    monkeypatch.setattr(Identity, "public_id", lambda self: self.secret.hex())
    api = MagicMock(spec=HfApi)
    api.run_job.return_value.id = "job"
    spec = JobSpec("example", "image", "workload:run", workers=2)
    keys = [Identity(bytes([i]) * 32) for i in range(3)]
    cluster = submit_cluster(spec, keys, public_relays=True, api=api)
    assert len(cluster.jobs) == 3
    for index, call in enumerate(api.run_job.call_args_list):
        kwargs = call.kwargs
        assert kwargs["secrets"] == {"HFDASK_NODE_KEY": keys[index].secret.hex()}
        assert not {"env", "expose", "ssh"}.intersection(kwargs)
        assert kwargs["labels"]["hfdask-node"] == str(index)
    assert "secret" not in str(cluster.manifest())
    assert keys[0].secret.hex() not in repr(keys[0])


def test_policy_is_explicit():
    with pytest.raises(ValueError, match="Explicitly"):
        submit_cluster(JobSpec("example", "image", "workload:run"), [])


def test_partial_launch_retains_handles(monkeypatch):
    monkeypatch.setattr(Identity, "public_id", lambda self: self.secret.hex())
    api = MagicMock(spec=HfApi)
    api.run_job.side_effect = [MagicMock(id="first"), ConnectionError("uncertain")]
    with pytest.raises(LaunchError) as caught:
        submit_cluster(
            JobSpec("example", "image", "workload:run", workers=1),
            [Identity(b"a" * 32), Identity(b"b" * 32)],
            public_relays=True,
            api=api,
        )
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
