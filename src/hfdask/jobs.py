"""Single-Job submission and reconnectable lifecycle handles.

`submit` runs a workload on one HF machine with a local Dask cluster. For
multiple machines, use `hfdask.cluster.submit_cluster` instead. Credentials
stay in the submitting process; images must contain the workload and hfdask
or install them through the configured bootstrap command.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Annotated, Any

from huggingface_hub import HfApi, Volume
from pydantic import AfterValidator
from pydantic.dataclasses import dataclass as validated_dataclass

from .config import (
    CONFIG,
    DaskMemoryLimit,
    Duration,
    OneThread,
    PositiveCount,
    RunConfig,
    Text,
    WaitConfig,
    require_entrypoint,
)

TERMINAL = frozenset({"COMPLETED", "ERROR", "CANCELED", "DELETED"})


def require_optional_entrypoint(value: str) -> str:
    return require_entrypoint(value) if value else value


def require_json_kwargs(value: dict[str, Any]) -> dict[str, Any]:
    # Preserve stdlib JSON semantics rather than coercing arbitrary workload objects.
    json.dumps(value, allow_nan=False)
    return value


def require_absolute_volume(value: Volume) -> Volume:
    if not value.mount_path.startswith("/"):
        raise ValueError("volume mount_path must be absolute")
    return value


OptionalEntrypoint = Annotated[str, AfterValidator(require_optional_entrypoint)]
WorkloadKwargs = Annotated[dict[str, Any], AfterValidator(require_json_kwargs)]
AbsoluteVolume = Annotated[Volume, AfterValidator(require_absolute_volume)]


@validated_dataclass(frozen=True, config=CONFIG)
class JobSpec:
    """Validated, frozen workload specification; construction does not submit a Job.

    Args:
        namespace: HF user or organization that owns the paid Jobs.
        image: Container image containing hfdask and the workload, or a bootstrap runtime.
        entrypoint: Importable `module:function` called with a Dask client as its first
            argument. Leave empty only for `hfdask.cluster.boot_cluster`.
        flavor: HF hardware flavor; also the default for multi-Job launch overrides.
        workers: Worker-machine count for homogeneous mesh launches, including a
            colocated scheduler worker if enabled. Single-Job submission detects
            local CPU cores instead. Explicit worker groups determine remote counts.
        threads_per_worker: Fixed at `1`; every usable CPU core gets one worker process.
        memory_limit: Dask memory limit, such as `"auto"` or `"1GiB"`. `"0"` disables
            the local limit; hardware-budgeted mesh workers require a positive limit.
        timeout: Remote Job lifetime as a positive integer duration, such as `"1h"`.
        kwargs: JSON-serializable keyword arguments passed to the workload.
        volumes: HF volumes with absolute mount paths; grant only required access.
        bootstrap: Command prefix prepended to the Python runner invocation.
        env: Non-secret environment variables supplied to each Job.

    Counts and strings are validated strictly without coercion. Schema failures
    raise `pydantic.ValidationError`; workload kwargs use standard-library JSON
    serialization and reject non-finite numbers. Use `dataclasses.replace` to
    derive a new specification.
    """

    namespace: Text
    image: Text
    entrypoint: OptionalEntrypoint = ""
    flavor: Text = "cpu-basic"
    workers: PositiveCount = 2
    threads_per_worker: OneThread = 1
    memory_limit: DaskMemoryLimit = "auto"
    timeout: Duration = "1h"
    kwargs: WorkloadKwargs = field(default_factory=dict)
    volumes: list[AbsoluteVolume] = field(default_factory=list)
    bootstrap: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def command(self) -> list[str]:
        """Build runner arguments without launching; require a nonempty entrypoint."""
        RunConfig(
            entrypoint=self.entrypoint,
            workers=self.workers,
            threads_per_worker=self.threads_per_worker,
        )
        return [
            *self.bootstrap,
            "python",
            "-m",
            "hfdask.runner",
            self.entrypoint,
            "--workers",
            str(self.workers),
            "--threads-per-worker",
            str(self.threads_per_worker),
            "--memory-limit",
            self.memory_limit,
            "--kwargs",
            json.dumps(self.kwargs, allow_nan=False),
        ]


class JobFailed(RuntimeError):  # noqa: N818 -- Preserve the public exception name.
    """The remote job reached a non-success terminal state."""


@dataclass
class Job:
    """Reconnectable handle for an existing HF Job; construction does not submit.

    Args:
        id: Job ID returned by HF submission or saved in a recovery manifest.
        namespace: HF user or organization that owns the Job.
        api: HF API client; defaults to the submitting process's configured credentials.

    A local wait timeout does not cancel the Job. Use `cancel` to request
    cancellation, then inspect `status` to verify a terminal state.
    """

    id: str
    namespace: str
    api: HfApi = field(default_factory=HfApi, repr=False)

    def status(self) -> str:
        """Inspect the current HF stage, such as `RUNNING` or `COMPLETED`."""
        stage = self.api.inspect_job(job_id=self.id, namespace=self.namespace).status.stage
        return str(getattr(stage, "value", stage))

    def cancel(self) -> None:
        """Request cancellation; a successful response does not verify termination."""
        self.api.cancel_job(job_id=self.id, namespace=self.namespace)

    def wait(self, *, timeout: float = 3600, poll_interval: float = 30) -> str:
        """Poll for success without canceling on a local deadline or API failure.

        Args:
            timeout: Positive local waiting budget in seconds, not the HF Job lifetime.
            poll_interval: Positive delay in seconds between status requests.

        Returns:
            `"COMPLETED"` when the remote workload succeeds.

        Raises:
            JobFailed: If the Job reaches another terminal state.
            TimeoutError: If the local deadline expires; the Job is not canceled.
            ValueError: If timing values fail validation.
        """
        WaitConfig(timeout=timeout, poll_interval=poll_interval)
        deadline = time.monotonic() + timeout
        while True:
            stage = self.status()
            if stage == "COMPLETED":
                return stage
            if stage in TERMINAL:
                raise JobFailed(f"Job {self.namespace}/{self.id} ended with {stage}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Job {self.namespace}/{self.id} remains {stage}; not canceled")
            time.sleep(min(poll_interval, remaining))


def submit(spec: JobSpec, *, api: HfApi | None = None) -> Job:
    """Submit one paid HF Job running a local Dask cluster.

    Args:
        spec: Workload, hardware, timeout, mounts, and environment to submit.
        api: Optional HF API client; otherwise use locally configured credentials.

    Returns:
        A reconnectable `Job` handle. Submission does not wait for readiness or completion.

    Raises:
        ValueError: If the runner configuration is invalid, including an empty entrypoint.

    HF API errors propagate. Never blindly retry an ambiguous submission response:
    the remote Job may exist even when no handle was returned.
    """
    client = api if api is not None else HfApi()
    environment_kwargs: dict[str, Any] = {"env": spec.env} if spec.env else {}
    info = client.run_job(
        image=spec.image,
        command=spec.command(),
        flavor=spec.flavor,
        namespace=spec.namespace,
        timeout=spec.timeout,
        volumes=spec.volumes,
        **environment_kwargs,
    )
    return Job(id=info.id, namespace=spec.namespace, api=client)
