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
    """A homogeneous group of remote HF machines, not Dask processes.

    Args:
        flavor: HF hardware flavor for every machine in this group.
        count: Positive machine count. Each GPU machine may start multiple Dask workers.
        tags: Custom categorical tags added to each worker's metadata. Tags must be
            nonempty and cannot start with `GPU_`, `FLAVOR_`, or `HAS_GPU`, which are
            reserved for hardware detection.

    Explicit groups replace homogeneous remote worker placement. A scheduler
    worker, if enabled, is additional to these groups. Inputs are strictly validated.
    """

    flavor: Text
    count: PositiveCount = 1
    tags: tuple[CustomTag, ...] = ()


@validated_dataclass(frozen=True, config=CONFIG)
class Identity:
    """A caller-provided Iroh identity; never serialize its secret into a manifest.

    Args:
        secret: Exactly 32 private bytes. Generate independently for each Job and
            persistent client, for example with `secrets.token_bytes(32)`.

    Keep persistent client keys separately from public recovery manifests. Job
    keys are passed through HF Job secrets, not ordinary environment configuration.
    """

    secret: Annotated[bytes, Field(min_length=32, max_length=32)] = field(repr=False)

    def public_id(self) -> str:
        """Derive the public endpoint ID as hexadecimal without exposing the secret."""
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
        return 1 + (
            sum(group.count for group in self.worker_groups)
            if self.worker_groups is not None
            else self.spec.workers - int(self.scheduler_worker)
        )

    @property
    def node_flavors(self) -> list[str]:
        remote = (
            [group.flavor for group in self.worker_groups for _ in range(group.count)]
            if self.worker_groups is not None
            else [self.worker_flavor or self.spec.flavor]
            * (self.spec.workers - int(self.scheduler_worker))
        )
        return [self.scheduler_flavor or self.spec.flavor, *remote]

    @property
    def peers(self) -> list[str]:
        identities = self.identities + ((self.client_identity,) if self.client_identity else ())
        return [identity.public_id() for identity in identities]

    @property
    def connection(self) -> dict[str, Any]:
        tags = (
            [list(group.tags) for group in self.worker_groups for _ in range(group.count)]
            if self.worker_groups is not None
            else [[] for _ in self.node_flavors[1:]]
        )
        config = {
            "peers": self.peers,
            "relays": self.relays,
            "public_relays": self.public_relays,
            "startup_timeout": self.startup_timeout,
            "node_flavors": self.node_flavors,
            "node_tags": [[], *tags],
            "hardware_detection": True,
        }
        if self.client_identity is not None:
            config.update(
                schema=1,
                persistent=True,
                job_nodes=len(self.node_flavors),
                scheduler_worker=self.scheduler_worker,
            )
        return config

    @property
    def command(self) -> list[str]:
        spec = (
            replace(self.spec, entrypoint="hfdask.runner:main")
            if self.client_identity is not None
            else self.spec
        )
        return spec.command()


@dataclass
class Cluster:
    """Known Job handles and public metadata for recovery and explicit cleanup.

    Args:
        id: Cluster label used to reconcile Jobs after ambiguous submissions.
        jobs: Known handles in launch order, scheduler first; may be a partial launch.
        connection: Public connection configuration for persistent clusters, otherwise `None`.

    Save `manifest` after each submission when recovery matters. A manifest has no
    private keys and is neither a task checkpoint nor sufficient client credentials.
    """

    id: str
    jobs: list[Job]
    connection: dict[str, Any] | None = None

    def manifest(self) -> dict[str, Any]:
        """Return JSON-serializable recovery handles and any public connection metadata."""
        result: dict[str, Any] = {
            "cluster_id": self.id,
            "jobs": [{"namespace": job.namespace, "id": job.id} for job in self.jobs],
        }
        if self.connection is not None:
            result["connection"] = self.connection
        return result

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any], *, api: HfApi | None = None) -> Cluster:
        """Restore handles for later explicit shutdown without submitting new Jobs.

        Args:
            manifest: Previously saved `manifest` mapping.
            api: Optional API client used by all restored Job handles.

        Returns:
            A cluster handle; this does not inspect liveness or reconnect a Dask client.
        """
        client = api if api is not None else HfApi()
        return cls(
            manifest["cluster_id"],
            [Job(job["id"], job["namespace"], client) for job in manifest["jobs"]],
            manifest.get("connection"),
        )

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
        """Cancel all known Jobs and poll until every one is terminal.

        Args:
            timeout: Positive polling budget in seconds after cancellation requests.
            poll_interval: Positive delay in seconds between status polls.

        Raises:
            TimeoutError: If termination remains unverified; retain the manifest.
            ValueError: If timing values fail validation.

        Status inspection errors propagate with handles intact. Only known Jobs
        are checked; reconcile ambiguous submissions by cluster labels separately.
        """
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

        Args:
            timeout: Positive local driver-wait budget in seconds.
            poll_interval: Positive delay in seconds between driver status polls.

        Returns:
            `"COMPLETED"` after the driver succeeds and known Jobs are verified terminal.

        Raises:
            JobFailed: If the driver fails and subsequent cleanup succeeds.
            TimeoutError: If the driver deadline or the separate cleanup deadline expires.
            ValueError: If timing is invalid or no scheduler Job is known.

        A local deadline or API failure leaves handles available; it never retries
        submissions. Driver termination triggers `close` with its own default
        timing, so total waiting may exceed `timeout`. Cleanup errors propagate
        before a driver failure can be reported as `JobFailed`.
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
    """Submission or its callback failed; `cluster` retains all known Job handles.

    No automatic rollback or resubmission is attempted. Save the partial manifest,
    reconcile potentially ambiguous submissions by cluster labels, and explicitly
    release capacity. The original failure is available as the exception cause.
    """

    def __init__(self, cluster: Cluster) -> None:
        self.cluster = cluster
        super().__init__(
            f"Cluster {cluster.id} submission failed. Known jobs retained; "
            "inspect cluster labels for an ambiguous submission before retrying."
        )


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
    """Launch a batch driver and worker machines as separate paid HF Jobs.

    Args:
        spec: Workload specification. Without explicit groups, `spec.workers` counts
            worker machines, including a colocated scheduler worker if enabled.
        identities: One distinct identity per Job, in scheduler-first order; at most
            64 Jobs. These keys are supplied by the caller, not generated here.
        public_relays: Required explicit consent to public n0 discovery and relay use.
        relay_urls: Optional custom relay URLs; these still use n0 discovery.
        startup_timeout: Positive worker-readiness timeout in seconds.
        scheduler_worker: Also run workers on the scheduler Job's hardware.
        scheduler_flavor: Scheduler hardware override; defaults to `spec.flavor`.
        worker_flavor: Homogeneous remote worker hardware override; defaults to `spec.flavor`.
        worker_groups: Explicit remote machine groups instead of `worker_flavor` and
            the remote count derived from `spec.workers`. A scheduler worker is additional.
        api: Optional HF API client using the submitter's credentials.
        on_submitted: Callback after each acknowledged Job, receiving the growing
            cluster handle. Use it to save recovery manifests incrementally.

    Returns:
        Known Job handles without waiting for readiness or workload completion.

    Raises:
        ValueError: If launch configuration, identity roster, or workload is invalid.
        LaunchError: If submission or its callback fails, including interruption;
            the exception retains the partial cluster for recovery.

    Each image must contain hfdask and the workload, or bootstrap them. No HF ports
    are exposed. One GPU machine may host multiple Dask worker processes. Never
    retry an ambiguous submission blindly; no automatic rollback is performed.
    """
    plan = LaunchPlan(
        spec=spec,
        identities=tuple(identities),
        client_identity=_client_identity,
        scheduler_flavor=scheduler_flavor,
        worker_flavor=worker_flavor,
        worker_groups=None if worker_groups is None else tuple(worker_groups),
        public_relays=public_relays,
        relays=list(relay_urls),
        startup_timeout=startup_timeout,
        scheduler_worker=scheduler_worker,
    )
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
                command=command
                + ["--mesh", configuration, "--node", str(index)]
                + (["--scheduler-worker"] if scheduler_worker else []),
                flavor=plan.node_flavors[index],
                namespace=spec.namespace,
                timeout=spec.timeout,
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


def boot_cluster(
    spec: JobSpec,
    identities: Sequence[Identity],
    *,
    client_identity: Identity,
    public_relays: bool = False,
    relay_urls: Sequence[str] = (),
    startup_timeout: int = 1200,
    scheduler_worker: bool = False,
    scheduler_flavor: str | None = None,
    worker_flavor: str | None = None,
    worker_groups: Sequence[WorkerGroup] | None = None,
    api: HfApi | None = None,
    on_submitted: Callable[[Cluster], None] | None = None,
) -> Cluster:
    """Submit a persistent cluster for later connections, not a batch workload.

    Args:
        spec: Job configuration with empty `entrypoint` and `kwargs`.
        identities: Distinct Job identities in scheduler-first order.
        client_identity: Separate identity authorized to connect to this cluster.
            Retain its private key separately from the public manifest.
        public_relays: Explicit consent to public discovery and relay use.
        relay_urls: Optional custom relays; public n0 discovery is still used.
        startup_timeout: Positive worker-readiness timeout in seconds.
        scheduler_worker: Enable workers on the scheduler machine as well.
        scheduler_flavor: Scheduler hardware override, defaulting to `spec.flavor`.
        worker_flavor: Homogeneous remote hardware override, defaulting to `spec.flavor`.
        worker_groups: Explicit remote groups instead of homogeneous placement.
        api: Optional HF API client.
        on_submitted: Callback for saving the growing cluster's recovery manifest.

    Returns:
        A `Cluster` with public connection metadata. Use `hfdask.client.connect`
        with its manifest and the client identity to wait for readiness and run tasks.

    Raises:
        ValueError: If the persistent workload or launch configuration is invalid.
        LaunchError: If submission or the callback fails; known handles are retained.

    Placement and identity-count rules match `submit_cluster`. Disconnecting a
    client does not release paid Jobs; explicitly call `Cluster.close` and retain
    recovery handles until termination is verified.
    """
    PersistentWorkloadConfig(entrypoint=spec.entrypoint, kwargs=spec.kwargs)
    return submit_cluster(
        spec,
        identities,
        _client_identity=client_identity,
        public_relays=public_relays,
        relay_urls=relay_urls,
        startup_timeout=startup_timeout,
        scheduler_worker=scheduler_worker,
        scheduler_flavor=scheduler_flavor,
        worker_flavor=worker_flavor,
        worker_groups=worker_groups,
        api=api,
        on_submitted=on_submitted,
    )
