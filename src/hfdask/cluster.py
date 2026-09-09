"""Multi-job submission with an explicit Iroh relay policy and node identities."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any
from uuid import uuid4

from huggingface_hub import HfApi

from .jobs import TERMINAL, Job, JobFailed, JobSpec


@dataclass(frozen=True)
class WorkerGroup:
    """A count of remote HF machines of one flavor (not Dask processes)."""

    flavor: str
    count: int = 1
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.count, int) or isinstance(self.count, bool):
            raise TypeError("Worker group count must be an integer")
        if not self.flavor.strip() or isinstance(self.count, bool) or self.count < 1:
            raise ValueError("Worker groups require a flavor and positive count")
        if any(not tag or tag.startswith(("GPU_", "FLAVOR_", "HAS_GPU")) for tag in self.tags):
            raise ValueError("Custom tags must be nonempty and not impersonate hardware tags")


@dataclass(frozen=True)
class Identity:
    """A caller-provided 32-byte Iroh secret. Never serialize this into a manifest."""

    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if len(self.secret) != 32:
            raise ValueError("Iroh identities require a 32-byte secret")

    def public_id(self) -> str:
        import iroh
        return iroh.SecretKey.from_bytes(self.secret).public().to_bytes().hex()


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
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")

    def wait(self, *, timeout: float = 3600, poll_interval: float = 30) -> str:
        """Wait for driver completion, then release and verify remaining jobs.

        A local deadline or API failure leaves handles available; it never retries
        submissions. A remote driver failure triggers cleanup before JobFailed.
        """
        self._timing(timeout, poll_interval)
        if not self.jobs:
            raise ValueError("Cluster has no scheduler job")
        deadline = time.monotonic() + timeout
        while True:
            stage = self.jobs[0].status()
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
    scheduler_hardware = spec.flavor if scheduler_flavor is None else scheduler_flavor
    worker_hardware = spec.flavor if worker_flavor is None else worker_flavor
    if not scheduler_hardware.strip() or not worker_hardware.strip():
        raise ValueError("Hardware flavors must not be empty")
    if not public_relays and not relay_urls:
        raise ValueError("Explicitly permit public relays or supply custom relay_urls")
    if public_relays and relay_urls:
        raise ValueError("Choose public or custom relays, not both")
    if any(not url.startswith("https://") for url in relay_urls):
        raise ValueError("Custom relays must use HTTPS")
    if worker_groups is not None:
        if not worker_groups or worker_flavor is not None:
            raise ValueError("Use nonempty worker_groups instead of worker_flavor")
        remote_flavors = [group.flavor for group in worker_groups for _ in range(group.count)]
        node_tags = [[], *[list(group.tags) for group in worker_groups for _ in range(group.count)]]
    else:
        remote_flavors = [worker_hardware] * (spec.workers - int(scheduler_worker))
        node_tags = [[] for _ in range(1 + len(remote_flavors))]
    node_flavors = [scheduler_hardware, *remote_flavors]
    node_count = len(node_flavors)
    if node_count > 64:
        raise ValueError("At most 64 Jobs per cluster")
    if len(identities) != node_count or startup_timeout < 1:
        raise ValueError("Provide one identity per Job and a positive deadline")
    ids = [identity.public_id() for identity in identities]
    if _client_identity is not None:
        ids.append(_client_identity.public_id())
    if len(set(ids)) != len(ids):
        raise ValueError("Every job needs a distinct identity")
    client = api if api is not None else HfApi()
    cluster = Cluster(uuid4().hex, [])
    config = {"peers": ids, "relays": list(relay_urls),
                                "public_relays": public_relays,
                                "startup_timeout": startup_timeout,
                                "node_flavors": node_flavors,
                                "node_tags": node_tags,
                                "hardware_detection": True}
    if _client_identity is not None:
        config.update(schema=1, persistent=True, job_nodes=node_count,
                      scheduler_worker=scheduler_worker)
        cluster.connection = config
    configuration = json.dumps(config)
    command = spec.command()
    environment_kwargs: dict[str, Any] = {"env": spec.env} if spec.env else {}
    try:
        for index, identity in enumerate(identities):
            info = client.run_job(
                image=spec.image,
                command=command + ["--mesh", configuration, "--node", str(index)]
                + (["--scheduler-worker"] if scheduler_worker else []),
                flavor=node_flavors[index],
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
    if spec.entrypoint or spec.kwargs:
        raise ValueError("Persistent clusters do not take an entrypoint or workload kwargs")
    return submit_cluster(replace(spec, entrypoint="hfdask.runner:main"), identities,
                          _client_identity=client_identity, **options)
