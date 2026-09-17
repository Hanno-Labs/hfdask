"""Configuration failures must precede uploads, submissions, and network startup."""

from dataclasses import FrozenInstanceError, is_dataclass, replace
from unittest.mock import MagicMock

import pytest
from huggingface_hub import Volume
from pydantic import ValidationError

from hfdask.client import connect
from hfdask.cluster import Cluster, LaunchPlan, WorkerGroup, submit_cluster
from hfdask.config import ClusterConfig, ConnectionConfig, RunnerClusterConfig
from hfdask.jobs import Job, JobSpec
from hfdask.network import issue_credentials
from hfdask.runner import run


def yaml_config(**overrides):
    return {
        "namespace": "example",
        "environment": {"image": "image"},
        **overrides,
    }


@pytest.mark.parametrize(
    "value,location",
    [
        (None, ()),
        ([], ()),
        ({1: "bad"}, (1,)),
        (yaml_config(unknown=True), ("unknown",)),
        (yaml_config(network={"public_relays": True}), ("network",)),
        (yaml_config(coordinator={"unknown": True}), ("coordinator", "unknown")),
        (yaml_config(workers={"unknown": True}), ("workers", "unknown")),
        (yaml_config(environment={"image": "image", "unknown": True}), ("environment", "unknown")),
        (
            yaml_config(
                mounts=[{"source": "hf://buckets/a/b", "target": "/data", "unknown": True}]
            ),
            ("mounts", 0, "unknown"),
        ),
        (yaml_config(namespace=" \t"), ("namespace",)),
        (yaml_config(workers=None), ("workers",)),
        (yaml_config(mounts={}), ("mounts",)),
        (
            yaml_config(environment={"image": "image", "groups": "deploy"}),
            ("environment", "groups"),
        ),
        (
            yaml_config(environment={"image": "image", "groups": ["--bad"]}),
            ("environment", "groups", 0),
        ),
    ],
)
def test_yaml_shape_and_unknown_keys(value, location):
    with pytest.raises(ValidationError) as caught:
        ClusterConfig.model_validate(value)
    assert location in [error["loc"] for error in caught.value.errors()]


@pytest.mark.parametrize("value", [True, False, 1.5, 1.0, "1", None, 0, 64])
def test_strict_yaml_counts(value):
    with pytest.raises(ValidationError) as caught:
        ClusterConfig.model_validate(yaml_config(workers={"count": value}))
    assert caught.value.errors()[0]["loc"] == ("workers", "count")


def test_independent_defaults():
    first = ClusterConfig.model_validate(yaml_config())
    second = ClusterConfig.model_validate(yaml_config())
    assert first.workers.count == 1
    assert first.workers.flavor == first.coordinator.flavor == "cpu-basic"
    assert first.coordinator.worker is False
    assert first.timeout_seconds == 3600
    first.environment.groups.append("deploy")
    assert second.environment.groups == []


@pytest.mark.parametrize("value", ["0h", "-1h", "1.5h", "1", "1h\n", " 1h", 3600, True])
def test_invalid_duration(value):
    with pytest.raises(ValidationError):
        ClusterConfig.model_validate(yaml_config(timeout=value))


@pytest.mark.parametrize("value,seconds", [("1s", 1), ("30m", 1800), ("2h", 7200), ("1d", 86400)])
def test_duration_units(value, seconds):
    assert ClusterConfig.model_validate(yaml_config(timeout=value)).timeout_seconds == seconds


@pytest.mark.parametrize("field", ["workers", "threads_per_worker"])
@pytest.mark.parametrize("value", [True, False, 1.0, 1.5, "1", 0, -1])
def test_job_counts_are_strict(field, value):
    with pytest.raises(ValidationError):
        JobSpec("example", "image", **{field: value})


def test_worker_threads_are_fixed_at_one():
    with pytest.raises(ValidationError):
        JobSpec("example", "image", threads_per_worker=2)


def test_dataclass_api_and_volume_compatibility():
    volume = Volume(type="bucket", source="example/data", mount_path="/data")
    spec = JobSpec("example", "image", "workload:run", volumes=[volume], bootstrap=("uv",))
    assert is_dataclass(spec)
    assert replace(spec, workers=3).workers == 3
    assert spec.volumes[0] == volume
    assert spec.bootstrap == ("uv",)
    with pytest.raises(FrozenInstanceError):
        spec.workers = 4
    with pytest.raises(ValidationError):
        replace(spec, workers=True)
    assert replace(WorkerGroup("cpu-basic"), count=2).count == 2


@pytest.mark.parametrize(
    "options",
    [
        {"count": True},
        {"count": 1.5},
        {"count": 0},
        {"flavor": " "},
        {"tags": ("GPU_FAKE",)},
        {"tags": ("FLAVOR_fake",)},
        {"tags": ("HAS_GPU",)},
        {"tags": ("",)},
        {"tags": "custom"},
    ],
)
def test_worker_group_validation(options):
    with pytest.raises(ValidationError):
        WorkerGroup(**{"flavor": "cpu-basic", **options})


@pytest.mark.parametrize(
    "options",
    [
        {"startup_timeout": True},
        {"startup_timeout": 1.5},
        {"startup_timeout": 0},
        {"startup_timeout": float("inf")},
        {"scheduler_worker": "false"},
        {"scheduler_flavor": " "},
    ],
)
def test_invalid_launch_options_never_submit(options):
    api = MagicMock()
    with pytest.raises(ValidationError):
        submit_cluster(JobSpec("example", "image", "workload:run"), api=api, **options)
    api.run_job.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": float("nan")},
        {"timeout": float("inf")},
        {"timeout": -1},
        {"timeout": True},
        {"poll_interval": 0},
        {"poll_interval": float("nan")},
        {"poll_interval": float("inf")},
        {"poll_interval": "5"},
    ],
)
def test_wait_validation_before_api_calls(options):
    api = MagicMock()
    job = Job("job", "example", api)
    for wait in (job.wait, Cluster("cluster", [job]).wait, Cluster("cluster", [job]).close):
        with pytest.raises(ValidationError):
            wait(**options)
    api.inspect_job.assert_not_called()
    api.cancel_job.assert_not_called()


def connection(**overrides):
    return {
        "schema": 2,
        "persistent": True,
        "job_nodes": 1,
        "scheduler_worker": True,
        **overrides,
    }


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        connection(schema=True),
        connection(schema=1),
        connection(persistent=1),
        connection(persistent=False),
        connection(job_nodes=True),
        connection(job_nodes=65),
        connection(scheduler_worker="false"),
    ],
)
def test_invalid_manifest_never_starts_tunnel(monkeypatch, value):
    tunnel = MagicMock()
    monkeypatch.setattr("hfdask.client._ssh_tunnel", tunnel)
    credentials, _ = issue_credentials(["client"])
    with pytest.raises((ValidationError, ValueError)):
        with connect({"connection": value}, credentials[0]):
            pytest.fail("Invalid configuration must not connect")
    tunnel.assert_not_called()


def test_connection_accepts_runner_metadata():
    config = ConnectionConfig.model_validate(
        connection(node_flavors=["cpu-basic"], schema_future=3)
    )
    assert config.job_nodes == 1


@pytest.mark.parametrize(
    "workers,options,message",
    [
        (64, {}, "At most 64"),
        (1, {"worker_groups": []}, "worker_groups"),
        (
            1,
            {"worker_groups": [WorkerGroup("cpu-basic")], "worker_flavor": "cpu-basic"},
            "instead of worker_flavor",
        ),
    ],
)
def test_launch_planning_guards_remain(workers, options, message):
    api = MagicMock()
    with pytest.raises(ValueError, match=message):
        submit_cluster(
            JobSpec("example", "image", "workload:run", workers=workers),
            api=api,
            **options,
        )
    api.run_job.assert_not_called()


def test_runner_rejects_counts_before_cluster_start(monkeypatch):
    cluster = MagicMock()
    monkeypatch.setattr("hfdask.runner.LocalCluster", cluster)
    with pytest.raises(ValidationError):
        run("tests.workloads:calculate", workers=True)
    cluster.assert_not_called()


def test_launch_limit_precedes_topology_expansion(monkeypatch):
    def expanded(_):
        pytest.fail("Oversized topology must not be expanded")

    monkeypatch.setattr(LaunchPlan, "node_flavors", property(expanded))
    with pytest.raises(ValidationError, match="At most 64"):
        submit_cluster(
            JobSpec("example", "image", "workload:run", workers=10**12),
            api=MagicMock(),
        )


@pytest.mark.parametrize(
    "options",
    [
        {"node": True},
        {"node": 2},
        {"job_nodes": None},
        {"startup_timeout": 0},
        {"startup_timeout": True},
        {"node_flavors": ["cpu-basic"]},
        {"node_tags": [[]]},
    ],
)
def test_runner_cluster_configuration(options):
    with pytest.raises(ValidationError):
        RunnerClusterConfig.model_validate(
            {
                "schema": 2,
                "node": 0,
                "job_nodes": 2,
                "persistent": False,
                "scheduler_worker": False,
                "startup_timeout": 1200,
                "node_flavors": ["cpu-basic", "cpu-basic"],
                "node_tags": [[], []],
                **options,
            }
        )


@pytest.mark.parametrize("persistent", [False, True])
def test_launch_wire_configuration_validates(persistent):
    plan = LaunchPlan(
        spec=JobSpec("example", "image", "" if persistent else "workload:run", workers=1),
        persistent=persistent,
    )
    for node in range(plan.node_count):
        wire = RunnerClusterConfig.model_validate({**plan.connection, "node": node})
        assert wire.model_dump(by_alias=True, exclude_unset=True) == plan.connection


@pytest.mark.parametrize("limit", ["-1", "garbage"])
def test_invalid_local_memory_precedes_startup(monkeypatch, limit):
    cluster = MagicMock()
    monkeypatch.setattr("hfdask.runner.LocalCluster", cluster)
    with pytest.raises(ValidationError):
        run("tests.workloads:calculate", memory_limit=limit)
    cluster.assert_not_called()
