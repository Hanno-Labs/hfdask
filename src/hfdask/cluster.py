"""Multi-job submission with an explicit Iroh relay policy and node identities."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Self
from uuid import uuid4

from huggingface_hub import HfApi
from pydantic import AfterValidator, Field, model_validator
from pydantic.dataclasses import dataclass as validated_dataclass

from .config import (
    CONFIG,
    LaunchConfig,
    PersistentWorkloadConfig,
    PositiveCount,
    Text,
    WaitConfig,
    require_distinct_identities,
)
from .jobs import TERMINAL, Job, JobFailed, JobSpec


def require_custom_tag(value: str) -> str:
    if not value or value.startswith(("GPU_", "FLAVOR_", "HAS_GPU")):
        raise ValueError("Custom tags must be nonempty and not impersonate hardware tags")
    return value


CustomTag = Annotated[str, AfterValidator(require_custom_tag)]


@validated_dataclass(frozen=True, config=CONFIG)
class WorkerGroup:
    """A count of remote HF machines of one flavor (not Dask processes)."""

    flavor: Text
    count: PositiveCount = 1
    tags: tuple[CustomTag, ...] = ()


@validated_dataclass(frozen=True, config=CONFIG)
class Identity:
    """A caller-provided 32-byte Iroh secret. Never serialize this into a manifest."""

    secret: Annotated[bytes, Field(min_length=32, max_length=32)] = field(repr=False)

    def public_id(self) -> str:
        import iroh
        return iroh.SecretKey.from_bytes(self.secret).public().to_bytes().hex()


class LaunchPlan(LaunchConfig):
    """Validated launch inputs and their resolved topology, before any submission."""

    spec: JobSpec
    identities: tuple[Identity, ...] = Field(repr=False, exclude=True)
    client_identity: Identity | None = Field(default=None, repr=False, exclude=True)
    worker_groups: Annotated[tuple[WorkerGroup, ...], Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def require_one_worker_placement_policy(self) -> Self:
        if self.worker_groups is not None and self.worker_flavor is not None:
            raise ValueError("Use nonempty worker_groups instead of worker_flavor")
        return self

    @model_validator(mode="after")
    def require_job_limit(self) -> Self:
        # Count before expanding flavors: untrusted counts must not allocate huge lists.
        if self.node_count > 64:
            raise ValueError("At most 64 Jobs per cluster")
        return self

    @model_validator(mode="after")
    def require_identity_per_job(self) -> Self:
        if len(self.identities) != self.node_count:
            raise ValueError("Provide one identity per Job")
        return self

    @model_validator(mode="after")
    def require_distinct_job_and_client_identities(self) -> Self:
        require_distinct_identities(tuple(self.peers))
        return self

    @model_validator(mode="after")
    def require_workload_for_launch_mode(self) -> Self:
        if self.client_identity is not None:
            PersistentWorkloadConfig(entrypoint=self.spec.entrypoint, kwargs=self.spec.kwargs)
        else:
            self.spec.command()
        return self

    @property
    def node_count(self) -> int:
        return 1 + (sum(group.count for group in self.worker_groups)
                    if self.worker_groups is not None
                    else self.spec.workers - int(self.scheduler_worker))

    @property
    def node_flavors(self) -> list[str]:
        remote = ([group.flavor for group in self.worker_groups for _ in range(group.count)]
                  if self.worker_groups is not None else
                  [self.worker_flavor or self.spec.flavor]
                  * (self.spec.workers - int(self.scheduler_worker)))
        return [self.scheduler_flavor or self.spec.flavor, *remote]

    @property
    def peers(self) -> list[str]:
        identities = self.identities + ((self.client_identity,) if self.client_identity else ())
        return [identity.public_id() for identity in identities]

    @property
    def connection(self) -> dict[str, Any]:
        tags = ([list(group.tags) for group in self.worker_groups for _ in range(group.count)]
                if self.worker_groups is not None else [[] for _ in self.node_flavors[1:]])
        config = {"peers": self.peers, "relays": self.relays,
                  "public_relays": self.public_relays, "startup_timeout": self.startup_timeout,
                  "node_flavors": self.node_flavors, "node_tags": [[], *tags],
                  "hardware_detection": True}
        if self.client_identity is not None:
            config.update(schema=1, persistent=True, job_nodes=len(self.node_flavors),
                          scheduler_worker=self.scheduler_worker)
        return config

    @property
    def command(self) -> list[str]:
        spec = (replace(self.spec, entrypoint="hfdask.runner:main")
                if self.client_identity is not None else self.spec)
        return spec.command()


@dataclass
class Cluster:
    """Known job handles; public metadata can be saved for recovery/cancellation."""

    id: str
    jobs: list[Job]
    connection: dict[str, Any] | None = None

    def manifest(self) -> dict[str, Any]:
        result: dict[str, Any] = {"cluster_id": self.id, "jobs": [
            {"namespace": job.namespace, "id": job.id} for job in self.jobs]}
        if self.connection is not None:
            result["connection"] = self.connection
        return result

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any], *, api: HfApi | None = None) -> Cluster:
        """Restore handles for later explicit shutdown; does not submit new Jobs."""
        client = api if api is not None else HfApi()
        return cls(manifest["cluster_id"],
                   [Job(job["id"], job["namespace"], client) for job in manifest["jobs"]],
                   manifest.get("connection"))

    def cancel(self) -> list[str]:
        """Attempt every cancellation; return IDs whose cancellation was not acknowledged.

        The caller must inspect terminal states before considering capacity released.
        """
        failures = []
        for job in self.jobs:
            try:
                job.cancel()
            except Exception:  # noqa: BLE001 - attempt all owned jobs, return failures to caller.
                failures.append(job.id)
        return failures

    def close(self, *, timeout: float = 120, poll_interval: float = 5) -> None:
        """Cancel all known jobs and verify termination, or raise with handles intact."""
        self._timing(timeout, poll_interval)
        self.cancel()
        deadline = time.monotonic() + timeout
        while any(job.status() not in TERMINAL for job in self.jobs):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Cluster {self.id} cleanup is unverified; retain manifest")
            time.sleep(min(remaining, poll_interval))

    @staticmethod
    def _timing(timeout: float, poll_interval: float) -> None:
        WaitConfig(timeout=timeout, poll_interval=poll_interval)

    def require_scheduler_job(self) -> Job:
        if not self.jobs:
            raise ValueError("Cluster has no scheduler job")
        return self.jobs[0]

    def wait(self, *, timeout: float = 3600, poll_interval: float = 30) -> str:
        """Wait for driver completion, then release and verify remaining jobs.

        A local deadline or API failure leaves handles available; it never retries
        submissions. A remote driver failure triggers cleanup before JobFailed.
        """
        self._timing(timeout, poll_interval)
        scheduler = self.require_scheduler_job()
        deadline = time.monotonic() + timeout
        while True:
            stage = scheduler.status()
            if stage in TERMINAL:
                self.close()
                if stage != "COMPLETED":
                    raise JobFailed(f"Cluster {self.id} driver ended with {stage}")
                return stage
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Cluster {self.id} still active; not canceled")
            time.sleep(min(remaining, poll_interval))


class LaunchError(RuntimeError):
    def __init__(self, cluster: Cluster) -> None:
        self.cluster = cluster
        super().__init__(f"Cluster {cluster.id} submission failed. Known jobs retained; "
                         "inspect cluster labels for an ambiguous submission before retrying.")


def submit_cluster(
    spec: JobSpec,
    identities: Sequence[Identity],
    *,
    public_relays: bool = False,
    relay_urls: Sequence[str] = (),
    startup_timeout: int = 1200,
    scheduler_worker: bool = False,
    scheduler_flavor: str | None = None,
    worker_flavor: str | None = None,
    worker_groups: Sequence[WorkerGroup] | None = None,
    api: HfApi | None = None,
    on_submitted: Callable[[Cluster], None] | None = None,
    _client_identity: Identity | None = None,
) -> Cluster:
    """Launch spec.workers total workers, optionally colocating one with the driver.

    The caller explicitly supplies keys and permits public n0 discovery/relay
    service use. Custom relays still use n0 discovery with this initial backend.
    Each image must contain hfdask and the workload, or bootstrap them. No exposed HF ports.
    Hardware overrides default independently to spec.flavor. A colocated worker
    shares the scheduler Job's hardware; worker_flavor applies to remote Jobs.
    """
    plan = LaunchPlan(spec=spec, identities=tuple(identities), client_identity=_client_identity,
                      scheduler_flavor=scheduler_flavor, worker_flavor=worker_flavor,
                      worker_groups=None if worker_groups is None else tuple(worker_groups),
                      public_relays=public_relays, relays=list(relay_urls),
                      startup_timeout=startup_timeout, scheduler_worker=scheduler_worker)
    client = api if api is not None else HfApi()
    cluster = Cluster(uuid4().hex, [])
    config = plan.connection
    if plan.client_identity is not None:
        cluster.connection = config
    configuration = json.dumps(config)
    command = plan.command
    environment_kwargs: dict[str, Any] = {"env": spec.env} if spec.env else {}
    try:
        for index, identity in enumerate(plan.identities):
            info = client.run_job(
                image=spec.image,
                command=command + ["--mesh", configuration, "--node", str(index)]
                + (["--scheduler-worker"] if scheduler_worker else []),
                flavor=plan.node_flavors[index],
                namespace=spec.namespace, timeout=spec.timeout,
                volumes=spec.volumes,
                **environment_kwargs,
                secrets={"HFDASK_NODE_KEY": identity.secret.hex()},
                labels={"hfdask-cluster": cluster.id, "hfdask-node": str(index)},
            )
            cluster.jobs.append(Job(info.id, spec.namespace, client))
            if on_submitted is not None:
                on_submitted(cluster)
    except (Exception, KeyboardInterrupt) as error:
        raise LaunchError(cluster) from error
    return cluster


def boot_cluster(spec: JobSpec, identities: Sequence[Identity], *,
                 client_identity: Identity, **options: Any) -> Cluster:
    """Submit a persistent cluster; connect waits for readiness, close releases Jobs."""
    PersistentWorkloadConfig(entrypoint=spec.entrypoint, kwargs=spec.kwargs)
    return submit_cluster(spec, identities,
                          _client_identity=client_identity, **options)
