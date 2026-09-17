"""In-Job Dask driver for local runs and HF network-group clusters."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import runpy
import sys
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path
from typing import Any, cast

from distributed import Client, LocalCluster
from distributed.security import Security

from .config import RunConfig, RunnerClusterConfig

TOPOLOGY_METADATA = "hfdask-topology"


def require_callable_entrypoint(entrypoint: str) -> Callable[..., Any]:
    module_name, name = entrypoint.split(":")
    function = getattr(importlib.import_module(module_name), name)
    if not callable(function):
        raise TypeError("entrypoint must be callable")
    return cast(Callable[..., Any], function)


def run(
    entrypoint: str,
    *,
    workers: int = 2,
    threads_per_worker: int = 1,
    memory_limit: str = "auto",
    kwargs: dict[str, Any] | None = None,
) -> object:
    """Run a callable on a local, process-based Dask cluster inside one machine."""
    RunConfig(
        entrypoint=entrypoint,
        workers=workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    )
    from .hardware import available_cpu_cores, detect

    cores = available_cpu_cores(detect())
    local_workers = cores - 1
    if local_workers < 1:
        raise RuntimeError("A single-Job cluster needs at least two available CPU cores")
    function = require_callable_entrypoint(entrypoint)
    print(
        json.dumps({"phase": "cluster_start", "cpu_cores": cores, "workers": local_workers}),
        flush=True,
    )
    with (
        LocalCluster(
            n_workers=local_workers,
            threads_per_worker=1,
            processes=True,
            host="127.0.0.1",
            dashboard_address=None,
            worker_dashboard_address=None,
            memory_limit=memory_limit,
        ) as cluster,
        Client(cluster, set_as_default=False) as client,
    ):
        client.wait_for_workers(local_workers, timeout=120)
        print(json.dumps({"phase": "cluster_ready", "workers": local_workers}), flush=True)
        result = function(client, **(kwargs or {}))
    print(json.dumps({"phase": "workload_complete"}), flush=True)
    return result


def run_script(client: Client, script: str) -> None:
    """Execute ordinary Python with this cluster as Dask's default scheduler."""
    path = Path(script).resolve(strict=True)
    argv, search_path = sys.argv, sys.path.copy()
    try:
        sys.argv = [str(path)]
        sys.path.insert(0, str(path.parent))
        with client.as_current():
            try:
                runpy.run_path(str(path), run_name="__main__")
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise RuntimeError(f"Script exited with status {error.code}") from error
    finally:
        sys.argv = argv
        sys.path[:] = search_path


def wait_topology(
    client: Client,
    job_nodes: int,
    scheduler_workers: int,
    timeout: float,
) -> tuple[int, ...]:
    """Wait until every remote Job has registered its complete detected worker set."""
    deadline = time.monotonic() + timeout
    last_observed: dict[int, int] = {}
    while True:
        infos = [worker.get("hfdask", {}) for worker in client.scheduler_info()["workers"].values()]
        unexpected = {
            info.get("node")
            for info in infos
            if not isinstance(info.get("node"), int) or not 0 <= info["node"] < job_nodes
        }
        if unexpected:
            raise RuntimeError(f"Worker topology contains unexpected nodes: {unexpected}")
        observed = {
            node: sum(info.get("node") == node for info in infos) for node in range(job_nodes)
        }
        expected: dict[int, int] = {0: scheduler_workers}
        consistent = observed[0] == scheduler_workers
        for node in range(1, job_nodes):
            advertised = {info.get("workers_on_node") for info in infos if info.get("node") == node}
            if len(advertised) != 1 or not all(
                isinstance(count, int) and count > 0 for count in advertised
            ):
                consistent = False
                continue
            expected[node] = advertised.pop()
            consistent = consistent and observed[node] == expected[node]
        if consistent and len(expected) == job_nodes:
            return tuple(expected[node] for node in range(job_nodes))
        last_observed = observed
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Worker topology incomplete: observed={last_observed}, expected_nodes={job_nodes}"
            )
        time.sleep(0.2)


def wait_published_topology(client: Client, job_nodes: int, timeout: float) -> tuple[int, ...]:
    """Wait for the coordinator's verified topology marker in scheduler metadata."""
    deadline = time.monotonic() + timeout
    while True:
        value = client.get_metadata(TOPOLOGY_METADATA, default=None)
        if (
            isinstance(value, (list, tuple))
            and len(value) == job_nodes
            and all(isinstance(count, int) and count >= 0 for count in value)
        ):
            return tuple(value)
        if time.monotonic() >= deadline:
            raise TimeoutError("Coordinator did not publish a complete worker topology")
        time.sleep(0.2)


def close_nannies(client: Client) -> None:
    """Ask each nanny to stop its worker instead of closing worker processes first."""
    workers = list(client.scheduler_info()["workers"])
    scheduler = cast(Any, client.scheduler)
    client.sync(
        scheduler.broadcast,
        msg={"op": "terminate", "reason": "hfdask-workload-complete", "reply": False},
        workers=workers,
        nanny=True,
    )


async def wait_for_scheduler(address: str, security: Security, timeout: float) -> None:
    """Authenticate to the scheduler with retries because HF aliases precede readiness."""
    deadline = time.monotonic() + timeout
    last_error: OSError | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Dask scheduler did not become ready at {address}") from last_error
        try:
            client = await Client(
                address,
                asynchronous=True,
                security=security,
                timeout=min(5, remaining),
                set_as_default=False,
            )
        except OSError as error:
            last_error = error
            await asyncio.sleep(min(0.5, remaining))
        else:
            await client.close()
            return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrypoint")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--memory-limit", default="auto")
    parser.add_argument("--kwargs", default="{}")
    parser.add_argument("--cluster", help="Public HF network-group cluster configuration")
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument(
        "--scheduler-worker",
        action="store_true",
        help="Run per-core workers after reserving one core for the scheduler",
    )
    args = parser.parse_args()
    kwargs = json.loads(args.kwargs)
    if not isinstance(kwargs, dict):
        parser.error("--kwargs must contain a JSON object")
    if args.cluster:
        asyncio.run(run_network_group(args, kwargs))
    else:
        run(
            args.entrypoint,
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
            memory_limit=args.memory_limit,
            kwargs=kwargs,
        )


async def run_network_group(args: argparse.Namespace, kwargs: dict[str, Any]) -> None:
    from distributed import Nanny, Scheduler

    from .hardware import detect, worker_metadata, worker_profiles
    from .network import (
        SCHEDULER_PORT,
        TLSCredentials,
        nanny_port,
        network_hostname,
        node_alias,
        scheduler_address,
        worker_port,
    )

    RunConfig(
        entrypoint=args.entrypoint,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit,
    )
    inputs = RunnerClusterConfig.model_validate({**json.loads(args.cluster), "node": args.node})
    config = inputs.model_dump(by_alias=True, exclude_unset=True)
    credentials = TLSCredentials.from_environment()
    profiles = (
        worker_profiles(
            detect(),
            config["node_flavors"][args.node],
            args.memory_limit,
            reserve_scheduler_core=args.node == 0 and args.scheduler_worker,
        )
        if args.node or args.scheduler_worker
        else []
    )
    scheduler_workers = len(profiles) if args.node == 0 else 0
    node_host = network_hostname(node_alias(args.node))
    address = scheduler_address()

    async with AsyncExitStack() as stack:
        security = stack.enter_context(credentials.security())
        scheduler = None
        if args.node == 0:
            scheduler = await stack.enter_async_context(
                Scheduler(
                    host="0.0.0.0",
                    port=SCHEDULER_PORT,
                    protocol="tls",
                    security=security,
                    contact_address=address,
                    dashboard_address=None,
                )
            )
        else:
            await wait_for_scheduler(address, security, config["startup_timeout"])

        nannies = []
        for ordinal, profile in enumerate(profiles):
            profile = profile.copy()
            profile.update(node=args.node, workers_on_node=len(profiles))
            profile["tags"] = profile["tags"] + config["node_tags"][args.node]
            gpu = profile["gpu"]
            port = worker_port(ordinal)
            nanny = await stack.enter_async_context(
                Nanny(
                    address,
                    host=node_host,
                    worker_port=port,
                    port=nanny_port(ordinal),
                    protocol="tls",
                    security=security,
                    contact_address=f"tls://{node_host}:{port}",
                    name=f"node-{args.node}-worker-{ordinal}",
                    nthreads=profile["nthreads"],
                    memory_limit=profile["memory_limit"],
                    resources=profile["resources"],
                    dashboard_address=None,
                    env={"CUDA_VISIBLE_DEVICES": gpu["uuid"] if gpu else ""},
                    startup_information={"hfdask": partial(worker_metadata, profile=profile)},
                    death_timeout=config["startup_timeout"],
                )
            )
            nannies.append(nanny)
        print(
            json.dumps(
                {
                    "phase": "cluster_node_ready",
                    "node": args.node,
                    "worker_processes": len(profiles),
                }
            ),
            flush=True,
        )
        if args.node:
            await asyncio.gather(*(nanny.finished() for nanny in nannies))
            return

        assert scheduler is not None

        def coordinator() -> None:
            with Client(
                "tls://127.0.0.1:8786",
                security=security,
                timeout=config["startup_timeout"],
                set_as_default=False,
            ) as client:
                topology = wait_topology(
                    client,
                    config["job_nodes"],
                    scheduler_workers,
                    config["startup_timeout"],
                )
                client.set_metadata(TOPOLOGY_METADATA, list(topology))
                print(
                    json.dumps({"phase": "cluster_ready", "workers_per_node": topology}), flush=True
                )
                if config["persistent"]:
                    return
                require_callable_entrypoint(args.entrypoint)(client, **kwargs)
                close_nannies(client)

        await asyncio.to_thread(coordinator)
        if config["persistent"]:
            print(json.dumps({"phase": "accepting_clients"}), flush=True)
            await scheduler.finished()
        else:
            print(json.dumps({"phase": "workload_complete"}), flush=True)


if __name__ == "__main__":
    main()
