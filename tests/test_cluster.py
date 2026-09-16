import base64
import json
import os
import time
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from huggingface_hub import HfApi, Volume

from hfdask.cluster import Cluster, LaunchError, WorkerGroup, submit_cluster
from hfdask.jobs import JobSpec
from hfdask.network import issue_credentials


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


def test_live_network_group_starts_before_hfdask_bootstrap():
    if os.environ.get("HFDASK_LIVE_HF") != "1":
        pytest.skip("set HFDASK_LIVE_HF=1 for the paid HF Jobs contract smoke")

    source_path = os.environ.get("HFDASK_LIVE_SOURCE_PATH")
    source_sha256 = os.environ.get("HFDASK_LIVE_SOURCE_SHA256")
    if not source_path or not source_sha256:
        pytest.fail(
            "set HFDASK_LIVE_SOURCE_PATH and HFDASK_LIVE_SOURCE_SHA256 "
            "for a valid jobs-artifacts source bundle"
        )

    api = HfApi()
    namespace = os.environ.get("HFDASK_LIVE_NAMESPACE", "Hanno-Labs")
    terminal_stages = {"COMPLETED", "ERROR", "CANCELED", "DELETED"}
    phases = [
        ("network-group-only", [], False),
        (
            "network-group-with-source-volume",
            [
                Volume(
                    type="bucket",
                    source=f"{namespace}/jobs-artifacts",
                    path=source_path,
                    mount_path="/tmp/hfdask-source",
                    read_only=True,
                )
            ],
            True,
        ),
    ]

    for phase, volumes, use_bootstrap in phases:
        cluster_id = uuid4().hex
        group = f"hfdask-contract-{cluster_id[:16]}"
        jobs = []
        markers_by_job = {}
        observations = {}
        credentials, client_credentials = issue_credentials(["node-0", "node-1"])
        assert client_credentials is None
        try:
            for index, alias in enumerate(("node-0", "node-1")):
                required_markers = ["boot"]
                command = [
                    "python",
                    "-c",
                    "import time; print('boot', flush=True); time.sleep(90)",
                ]
                if use_bootstrap:
                    required_markers = [
                        "hfdask: staging project source",
                        (
                            '{"phase": "cluster_ready", "workers_per_node": [1, 2]}'
                            if index == 0
                            else '{"phase": "cluster_node_ready", "node": 1'
                        ),
                    ]
                    if index == 0:
                        required_markers.extend(
                            [
                                '{"rows": 1000000, "workers": 3, "partitions": 12, "sum": 999999000000}',
                                '{"phase": "workload_complete"}',
                            ]
                        )
                    command = [
                        "python3",
                        "/tmp/hfdask-source/bootstrap.py",
                        '["deploy"]',
                        source_sha256,
                        "python",
                        "-m",
                        "hfdask.runner",
                        "hfdask.runner:run_script",
                        "--workers",
                        "2",
                        "--threads-per-worker",
                        "1",
                        "--memory-limit",
                        "auto",
                        "--kwargs",
                        '{"script": "examples/cpu.py"}',
                        "--cluster",
                        '{"schema": 2, "persistent": false, "job_nodes": 2, "scheduler_worker": true, "startup_timeout": 1200, "node_flavors": ["cpu-basic", "cpu-basic"], "node_tags": [[], []]}',
                        "--node",
                        str(index),
                        "--scheduler-worker",
                    ]
                kwargs = {
                    "image": "ghcr.io/astral-sh/uv:python3.12-bookworm",
                    "command": command,
                    "flavor": "cpu-basic",
                    "timeout": "5m",
                    "namespace": namespace,
                    "network_group": group,
                    "network_aliases": ([alias, "scheduler"] if alias == "node-0" else [alias]),
                    "secrets": credentials[index].job_secrets(),
                    "env": {
                        "PYTHONPATH": "/tmp/hfdask-project:/tmp/hfdask-project/examples",
                        "DASK_DISTRIBUTED__WORKER__DAEMON": "False",
                        "DASK_DISTRIBUTED__WORKER__MULTIPROCESSING_METHOD": "spawn",
                        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                    },
                    "labels": {
                        "hfdask-cluster": cluster_id,
                        "hfdask-node": str(index),
                    },
                }
                if volumes:
                    kwargs["volumes"] = volumes
                job = api.run_job(**kwargs)
                jobs.append(job)
                markers_by_job[job.id] = required_markers

            deadline = time.monotonic() + 30
            while True:
                all_ready = True
                for job in jobs:
                    info = api.inspect_job(job_id=job.id, namespace=namespace)
                    stage_value = info.status.stage
                    stage = str(getattr(stage_value, "value", stage_value))
                    lines = (
                        list(
                            api.fetch_job_logs(
                                job_id=job.id,
                                namespace=namespace,
                                follow=False,
                                tail=200,
                            )
                        )
                        if stage in {"RUNNING", "COMPLETED"}
                        else []
                    )
                    observations[job.id] = {
                        "stage": stage,
                        "message": getattr(info.status, "message", None),
                        "logs": [str(line) for line in lines],
                    }
                    if stage in terminal_stages - {"COMPLETED"}:
                        pytest.fail(f"{phase} ended before boot: {observations}")
                    expected_stage = "COMPLETED" if use_bootstrap else "RUNNING"
                    all_ready = (
                        all_ready
                        and stage == expected_stage
                        and all(
                            any(marker in str(line) for line in lines)
                            for marker in markers_by_job[job.id]
                        )
                    )

                if all_ready:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    pytest.fail(f"{phase} did not boot within 30 seconds: {observations}")
                time.sleep(min(2, remaining))
        finally:
            for job in jobs:
                info = api.inspect_job(job_id=job.id, namespace=namespace)
                stage_value = info.status.stage
                stage = str(getattr(stage_value, "value", stage_value))
                if stage not in terminal_stages:
                    api.cancel_job(job_id=job.id, namespace=namespace)
            if jobs:
                final = api.wait_for_job(
                    [job.id for job in jobs],
                    namespace=namespace,
                    timeout=120,
                    poll_interval=2,
                )
                assert all(
                    str(getattr(info.status.stage, "value", info.status.stage)) in terminal_stages
                    for info in final
                )


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
