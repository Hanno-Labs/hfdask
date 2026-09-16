"""External Dask clients authenticated through an HF SSH forward and mTLS."""

from __future__ import annotations

import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from .config import ConnectConfig, ConnectionConfig
from .jobs import TERMINAL
from .network import SCHEDULER_PORT, TLSCredentials


def _scheduler_handle(manifest: dict[str, Any]) -> tuple[str, str]:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs or not isinstance(jobs[0], dict):
        raise ValueError("Persistent manifest has no scheduler Job handle")
    namespace, job_id = jobs[0].get("namespace"), jobs[0].get("id")
    if not isinstance(namespace, str) or not namespace or not isinstance(job_id, str) or not job_id:
        raise ValueError("Persistent manifest has an invalid scheduler Job handle")
    return namespace, job_id


def _wait_for_running(
    api: HfApi,
    namespace: str,
    job_id: str,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while True:
        status = api.inspect_job(job_id=job_id, namespace=namespace).status.stage
        stage = str(getattr(status, "value", status))
        if stage == "RUNNING":
            return
        if stage in TERMINAL:
            raise RuntimeError(f"Scheduler Job {namespace}/{job_id} ended with {stage}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Scheduler Job {namespace}/{job_id} remains {stage}")
        time.sleep(min(1, remaining))


@contextmanager
def _ssh_tunnel(
    job_id: str,
    *,
    local_port: int,
    timeout: float,
    identity_file: Path | None,
) -> Iterator[None]:
    executable = shutil.which("ssh")
    if executable is None:
        raise RuntimeError("OpenSSH is required for persistent hfdask clients")
    command = [
        executable,
        "-N",
        "-L",
        f"127.0.0.1:{local_port}:127.0.0.1:{SCHEDULER_PORT}",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]
    if identity_file is not None:
        command.extend(["-i", str(identity_file)])
    command.append(f"{job_id}@ssh.hf.jobs")
    with tempfile.TemporaryFile(mode="w+t") as error_output:
        process = subprocess.Popen(  # noqa: S603 - fixed executable and validated arguments.
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=error_output,
            text=True,
        )
        try:
            deadline = time.monotonic() + timeout
            while True:
                status = process.poll()
                if status is not None:
                    error_output.seek(0)
                    detail = error_output.read(4000).strip()
                    raise ConnectionError(
                        f"HF SSH tunnel exited with status {status}: {detail or 'no diagnostic'}"
                    )
                try:
                    with socket.create_connection(("127.0.0.1", local_port), timeout=0.2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("HF SSH tunnel did not open its local port")
                    time.sleep(0.1)
            yield
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


@contextmanager
def connect(
    manifest: dict[str, Any],
    credentials: TLSCredentials,
    *,
    timeout: float = 1200,
    local_port: int = 8786,
    identity_file: Path | None = None,
    api: HfApi | None = None,
) -> Iterator[Any]:
    """Connect to a persistent cluster without taking ownership of its HF Jobs.

    The scheduler Job must have been created by `boot_cluster`, which enables HF
    SSH only on that Job. This context waits for the Job, opens a loopback-only SSH
    forward, authenticates to Dask with the separately retained client certificate,
    and verifies the coordinator's published worker topology. Closing it stops only
    the local client and tunnel; call `Cluster.close` to release paid Jobs.
    """
    from distributed import Client

    from .runner import wait_published_topology

    connection = ConnectionConfig.model_validate(manifest.get("connection"))
    ConnectConfig(timeout=timeout, connection=connection, local_port=local_port)
    namespace, job_id = _scheduler_handle(manifest)
    client_api = api if api is not None else HfApi()
    _wait_for_running(client_api, namespace, job_id, timeout)
    with (
        _ssh_tunnel(
            job_id,
            local_port=local_port,
            timeout=timeout,
            identity_file=identity_file,
        ),
        credentials.security() as security,
        Client(
            f"tls://127.0.0.1:{local_port}",
            security=security,
            timeout=timeout,
            set_as_default=False,
        ) as client,
    ):
        wait_published_topology(client, connection.job_nodes, timeout)
        yield client
