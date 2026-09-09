"""Client-side lifecycle. Credentials stay in the submitting process."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from huggingface_hub import HfApi, Volume

_ENTRYPOINT = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*\Z")
TERMINAL = frozenset({"COMPLETED", "ERROR", "CANCELED", "DELETED"})


@dataclass(frozen=True)
class JobSpec:
    """A batch job; the image must contain hfdask and the workload module."""

    namespace: str
    image: str
    entrypoint: str = ""
    flavor: str = "cpu-basic"
    workers: int = 2
    threads_per_worker: int = 1
    memory_limit: str = "auto"
    timeout: str = "1h"
    kwargs: dict[str, Any] = field(default_factory=dict)
    volumes: list[Volume] = field(default_factory=list)
    bootstrap: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.namespace.strip() or not self.image.strip():
            raise ValueError("namespace and image are required")
        if self.entrypoint and not _ENTRYPOINT.fullmatch(self.entrypoint):
            raise ValueError("entrypoint must be module:function")
        if self.workers < 1 or self.threads_per_worker < 1:
            raise ValueError("worker and thread counts must be positive")
        json.dumps(self.kwargs, allow_nan=False)
        for volume in self.volumes:
            if not volume.mount_path.startswith("/"):
                raise ValueError("volume mount_path must be absolute")

    def command(self) -> list[str]:
        if not self.entrypoint:
            raise ValueError("Batch submissions require an entrypoint")
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
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be finite and positive")
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
