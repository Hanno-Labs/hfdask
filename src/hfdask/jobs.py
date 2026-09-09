"""Client-side lifecycle. Credentials stay in the submitting process."""

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
    """A batch job; the image must contain hfdask and the workload module."""

    namespace: Text
    image: Text
    entrypoint: OptionalEntrypoint = ""
    flavor: Text = "cpu-basic"
    workers: PositiveCount = 2
    threads_per_worker: PositiveCount = 1
    memory_limit: DaskMemoryLimit = "auto"
    timeout: Duration = "1h"
    kwargs: WorkloadKwargs = field(default_factory=dict)
    volumes: list[AbsoluteVolume] = field(default_factory=list)
    bootstrap: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def command(self) -> list[str]:
        RunConfig(entrypoint=self.entrypoint, workers=self.workers,
                  threads_per_worker=self.threads_per_worker)
        return [*self.bootstrap,
            "python", "-m", "hfdask.runner", self.entrypoint,
            "--workers", str(self.workers),
            "--threads-per-worker", str(self.threads_per_worker),
            "--memory-limit", self.memory_limit,
            "--kwargs", json.dumps(self.kwargs, allow_nan=False),
        ]


class JobFailed(RuntimeError):
    """The remote job reached a non-success terminal state."""


@dataclass
class Job:
    """Reconnectable job handle. A local wait timeout does not cancel the job."""

    id: str
    namespace: str
    api: HfApi = field(default_factory=HfApi, repr=False)

    def status(self) -> str:
        stage = self.api.inspect_job(job_id=self.id, namespace=self.namespace).status.stage
        return str(getattr(stage, "value", stage))

    def cancel(self) -> None:
        self.api.cancel_job(job_id=self.id, namespace=self.namespace)

    def wait(self, *, timeout: float = 3600, poll_interval: float = 30) -> str:
        """Wait for success; failures raise JobFailed, local deadlines TimeoutError."""
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
    """Submit once. Never blindly retry a submission with an ambiguous response."""
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
