"""Configuration failures must precede uploads, submissions, and transport startup."""

from dataclasses import FrozenInstanceError, is_dataclass, replace
from unittest.mock import MagicMock

import pytest
from huggingface_hub import Volume
from pydantic import ValidationError

from hfdask.client import connect
from hfdask.cluster import Cluster, Identity, WorkerGroup, submit_cluster
from hfdask.config import (
    ClusterConfig,
    ConnectionConfig,
    MeshConfig,
    RelayConfig,
    RunnerMeshConfig,
    ServiceConfig,
)
from hfdask.jobs import Job, JobSpec
from hfdask.runner import run


def yaml_config(**overrides):
    return {"namespace": "example", "environment": {"image": "image"},
            "network": {"public_relays": True}, **overrides}


@pytest.mark.parametrize("value,location", [
    (None, ()), ([], ()), ({1: "bad"}, (1,)),
    (yaml_config(unknown=True), ("unknown",)),
    (yaml_config(coordinator={"unknown": True}), ("coordinator", "unknown")),
    (yaml_config(workers={"unknown": True}), ("workers", "unknown")),
    (yaml_config(environment={"image": "image", "unknown": True}), ("environment", "unknown")),
    (yaml_config(network={"public_relays": True, "unknown": True}), ("network", "unknown")),
    (yaml_config(mounts=[{"source": "hf://buckets/a/b", "target": "/data", "unknown": True}]),
     ("mounts", 0, "unknown")),
    (yaml_config(namespace=" \t"), ("namespace",)),
    (yaml_config(workers=None), ("workers",)),
    (yaml_config(mounts={}), ("mounts",)),
    (yaml_config(environment={"image": "image", "extras": "inference"}),
     ("environment", "extras")),
    (yaml_config(environment={"image": "image", "extras": ["--bad"]}),
     ("environment", "extras", 0)),
])
def test_yaml_shape_and_unknown_keys(value, location):
    with pytest.raises(ValidationError) as caught:
        ClusterConfig.model_validate(value)
    assert location in [error["loc"] for error in caught.value.errors()]


@pytest.mark.parametrize("value", [True, False, 1.5, 1.0, "1", None, 0, 64])
def test_strict_yaml_counts(value):
    with pytest.raises(ValidationError) as caught:
        ClusterConfig.model_validate(yaml_config(workers={"count": value}))
    assert caught.value.errors()[0]["loc"] == ("workers", "count")


@pytest.mark.parametrize("value", [1, 0, "true", "false", None, False])
def test_yaml_consent_is_explicit_boolean(value):
    with pytest.raises(ValidationError):
        ClusterConfig.model_validate(yaml_config(network={"public_relays": value}))


def test_missing_consent_and_independent_defaults():
    config = yaml_config()
    del config["network"]
    with pytest.raises(ValidationError, match="network"):
        ClusterConfig.model_validate(config)
    first = ClusterConfig.model_validate(yaml_config())
    second = ClusterConfig.model_validate(yaml_config())
    assert first.workers.count == 1
    assert first.workers.flavor == first.coordinator.flavor == "cpu-basic"
    assert first.coordinator.worker is False
    assert first.timeout_seconds == 3600
    first.environment.extras.append("inference")
    assert second.environment.extras == []


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


@pytest.mark.parametrize("options", [
    {"count": True}, {"count": 1.5}, {"count": 0}, {"flavor": " "},
    {"tags": ("GPU_FAKE",)}, {"tags": ("FLAVOR_fake",)}, {"tags": ("HAS_GPU",)},
    {"tags": ("",)}, {"tags": "custom"},
])
def test_worker_group_validation(options):
    with pytest.raises(ValidationError):
        WorkerGroup(**{"flavor": "cpu-basic", **options})


@pytest.mark.parametrize("options", [
    {"startup_timeout": True}, {"startup_timeout": 1.5}, {"startup_timeout": 0},
    {"startup_timeout": float("inf")}, {"scheduler_worker": "false"},
    {"public_relays": "true"}, {"relay_urls": ["http://relay"]},
    {"relay_urls": ["https://relay"]}, {"scheduler_flavor": " "},
])
def test_invalid_launch_options_never_submit(options):
    api = MagicMock()
    with pytest.raises(ValidationError):
        submit_cluster(JobSpec("example", "image", "workload:run"), [], api=api,
                       **{"public_relays": True, **options})
    api.run_job.assert_not_called()


def test_relay_policy():
    assert RelayConfig(relays=["https://relay"]).public_relays is False
    with pytest.raises(ValidationError):
        RelayConfig()
    with pytest.raises(ValidationError):
        RelayConfig(public_relays=True, relays=["https://relay"])


@pytest.mark.parametrize("options", [
    {"timeout": float("nan")}, {"timeout": float("inf")}, {"timeout": -1},
    {"timeout": True}, {"poll_interval": 0}, {"poll_interval": float("nan")},
    {"poll_interval": float("inf")}, {"poll_interval": "5"},
])
def test_wait_validation_before_api_calls(options):
    api = MagicMock()
    job = Job("job", "example", api)
    for wait in (job.wait, Cluster("cluster", [job]).wait, Cluster("cluster", [job]).close):
        with pytest.raises(ValidationError):
            wait(**options)
    api.inspect_job.assert_not_called()
    api.cancel_job.assert_not_called()


def connection(**overrides):
    return {"schema": 1, "persistent": True, "job_nodes": 1, "peers": ["server", "client"],
            "scheduler_worker": True, "public_relays": True, "relays": [], **overrides}


@pytest.mark.parametrize("value", [None, {}, connection(schema=True), connection(persistent=1),
    connection(persistent=False),
    connection(job_nodes=True), connection(job_nodes=65), connection(peers=["client", "client"]),
    connection(peers=["client"]), connection(scheduler_worker="false"),
    connection(public_relays="false"), connection(relays=["http://relay"]),
    connection(public_relays=False)])
def test_invalid_manifest_never_starts_transport(monkeypatch, value):
    serve = MagicMock()
    monkeypatch.setattr("hfdask.client._serve", serve)
    with pytest.raises(ValidationError), connect({"connection": value}, Identity(b"a" * 32)):
        pytest.fail("Invalid configuration must not connect")
    serve.assert_not_called()


def test_connection_accepts_runner_metadata():
    config = ConnectionConfig.model_validate(connection(node_flavors=["cpu-basic"], schema_future=2))
    assert config.job_nodes == 1


@pytest.mark.parametrize("options", [
    {"index": True}, {"base_port": True}, {"base_port": 1023}, {"base_port": 65536},
    {"max_connections": True}, {"max_connections": 0}, {"max_connections": 1.5},
    {"bind_host": "0.0.0.0"}, {"services": (True,)}, {"services": (-1,)},
])
def test_mesh_configuration(options):
    with pytest.raises(ValidationError):
        MeshConfig(**{"index": 0, "services": (0,), "peers": (b"own",),
                      "endpoint_id": b"own", **options})


@pytest.mark.parametrize("workers,keys,options,message", [
    (64, [], {}, "At most 64"),
    (1, [], {}, "one identity per Job"),
    (1, [Identity(b"a" * 32)] * 2, {}, "distinct identity"),
    (1, [], {"worker_groups": []}, "worker_groups"),
    (1, [], {"worker_groups": [WorkerGroup("cpu-basic")], "worker_flavor": "cpu-basic"},
     "instead of worker_flavor"),
])
def test_launch_planning_guards_remain(monkeypatch, workers, keys, options, message):
    monkeypatch.setattr(Identity, "public_id", lambda self: self.secret.hex())
    api = MagicMock()
    with pytest.raises(ValueError, match=message):
        submit_cluster(JobSpec("example", "image", "workload:run", workers=workers), keys,
                       public_relays=True, api=api, **options)
    api.run_job.assert_not_called()


def test_runner_rejects_counts_before_cluster_start(monkeypatch):
    cluster = MagicMock()
    monkeypatch.setattr("hfdask.runner.LocalCluster", cluster)
    with pytest.raises(ValidationError):
        run("tests.workloads:calculate", workers=True)
    cluster.assert_not_called()


@pytest.mark.parametrize("options,message", [
    ({"services": (1,)}, "service"),
    ({"index": 1}, "index"),
    ({"base_port": 65535}, "port"),
    ({"peers": (b"own", b"own")}, "distinct identity"),
    ({"endpoint_id": b"other"}, "endpoint identity"),
])
def test_focused_mesh_invariants(options, message):
    with pytest.raises(ValidationError, match=message):
        MeshConfig(**{"index": 0, "services": (0,), "peers": (b"own",),
                      "endpoint_id": b"own", **options})


@pytest.mark.parametrize("options", [
    {"nodes": True}, {"nodes": 0}, {"node": True}, {"node": -1}, {"node": 2},
    {"ordinal": True}, {"ordinal": -1}, {"ordinal": 16},
])
def test_strict_service_indices(options):
    with pytest.raises(ValidationError):
        ServiceConfig(**{"nodes": 2, "node": 0, "ordinal": 0, **options})


def test_launch_limit_precedes_topology_expansion(monkeypatch):
    from hfdask.cluster import LaunchPlan

    def expanded(_):
        pytest.fail("Oversized topology must not be expanded")

    monkeypatch.setattr(LaunchPlan, "node_flavors", property(expanded))
    with pytest.raises(ValidationError, match="At most 64"):
        submit_cluster(JobSpec("example", "image", "workload:run", workers=10**12), [],
                       public_relays=True, api=MagicMock())


def test_launch_plan_excludes_secret_material(monkeypatch):
    from hfdask.cluster import LaunchPlan

    monkeypatch.setattr(Identity, "public_id", lambda self: self.secret.hex())
    plan = LaunchPlan(spec=JobSpec("example", "image", "workload:run", workers=1),
                      identities=(Identity(b"a" * 32), Identity(b"b" * 32)), public_relays=True)
    assert "identities" not in plan.model_dump()
    assert "secret" not in repr(plan)
    assert "client_identity" not in plan.model_dump()


@pytest.mark.parametrize("options", [
    {"node": True}, {"node": 2}, {"job_nodes": None},
    {"startup_timeout": 0}, {"startup_timeout": True},
    {"peers": ["same", "same"]}, {"hardware_detection": "true"},
    {"node_flavors": ["cpu-basic"]}, {"node_tags": [[]]},
])
def test_runner_mesh_configuration(options):
    with pytest.raises(ValidationError):
        RunnerMeshConfig.model_validate({
            "node": 0, "peers": ["scheduler", "worker"], "public_relays": True,
            "relays": [], "startup_timeout": 1200, "hardware_detection": True,
            "node_flavors": ["cpu-basic", "cpu-basic"], **options,
        })


@pytest.mark.parametrize("persistent", [False, True])
def test_launch_wire_configuration_validates(monkeypatch, persistent):
    from hfdask.cluster import LaunchPlan

    monkeypatch.setattr(Identity, "public_id", lambda self: self.secret.hex())
    plan = LaunchPlan(
        spec=JobSpec("example", "image", "" if persistent else "workload:run", workers=1),
        identities=(Identity(b"a" * 32), Identity(b"b" * 32)),
        client_identity=Identity(b"c" * 32) if persistent else None, public_relays=True,
    )
    for node in range(plan.node_count):
        wire = RunnerMeshConfig.model_validate({**plan.connection, "node": node})
        assert wire.model_dump(by_alias=True, exclude_unset=True) == plan.connection


def test_invalid_runner_mesh_never_binds_endpoint(monkeypatch):
    import asyncio
    import json
    from argparse import Namespace
    from unittest.mock import AsyncMock

    from hfdask.runner import run_mesh

    bind = AsyncMock()
    monkeypatch.setattr("iroh.Endpoint.bind", bind)
    monkeypatch.setenv("HFDASK_NODE_KEY", "not consumed for invalid config")
    args = Namespace(entrypoint="workload:run", workers=1, threads_per_worker=1,
                     memory_limit="auto", node=2, mesh=json.dumps({
                         "peers": ["server"], "public_relays": True,
                         "relays": [], "startup_timeout": 1200,
                     }))
    with pytest.raises(ValidationError, match="roster"):
        asyncio.run(run_mesh(args, {}))
    bind.assert_not_called()


def test_runner_mesh_accepts_legacy_and_persistent_metadata():
    legacy = RunnerMeshConfig.model_validate({
        "node": 0, "peers": ["scheduler", "worker"], "public_relays": True,
        "relays": [], "startup_timeout": 1200,
    })
    assert legacy.nodes == 2
    assert "job_nodes" not in legacy.model_dump(exclude_unset=True)
    assert "node" not in legacy.model_dump()
    persistent = RunnerMeshConfig.model_validate(connection(
        node=0, startup_timeout=1200, hardware_detection=True, node_flavors=["cpu-basic"]))
    assert persistent.nodes == 1


@pytest.mark.parametrize("limit", ["-1", "garbage"])
def test_invalid_local_memory_precedes_startup(monkeypatch, limit):
    cluster = MagicMock()
    monkeypatch.setattr("hfdask.runner.LocalCluster", cluster)
    with pytest.raises(ValidationError):
        run("tests.workloads:calculate", memory_limit=limit)
    cluster.assert_not_called()
