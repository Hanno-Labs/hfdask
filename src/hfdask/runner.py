"""In-job driver. No HF credential or externally reachable scheduler required."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import runpy
import sys
import time
from collections.abc import Callable
from contextlib import AsyncExitStack
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import iroh

from distributed import Client, LocalCluster

from .config import RunConfig, RunnerMeshConfig


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
    """Call module:function(client, **kwargs) and close the cluster on every exit."""
    RunConfig(
        entrypoint=entrypoint,
        workers=workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
    )
    function = require_callable_entrypoint(entrypoint)
    print(json.dumps({"phase": "cluster_start", "workers": workers}), flush=True)
    with (
        LocalCluster(
            n_workers=workers,
            threads_per_worker=threads_per_worker,
            processes=True,
            host="127.0.0.1",
            dashboard_address=None,
            worker_dashboard_address=None,
            memory_limit=memory_limit,
        ) as cluster,
        Client(cluster, set_as_default=False) as client,
    ):
        client.wait_for_workers(workers, timeout=120)
        print(json.dumps({"phase": "cluster_ready", "workers": workers}), flush=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrypoint")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--memory-limit", default="auto")
    parser.add_argument("--kwargs", default="{}")
    parser.add_argument("--mesh", help="Public cluster configuration for the P2P backend")
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument(
        "--scheduler-worker",
        action="store_true",
        help="Run one of the total workers in the scheduler Job",
    )
    args = parser.parse_args()
    kwargs = json.loads(args.kwargs)
    if not isinstance(kwargs, dict):
        parser.error("--kwargs must contain a JSON object")
    if args.mesh:
        asyncio.run(run_mesh(args, kwargs))
    else:
        run(
            args.entrypoint,
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
            memory_limit=args.memory_limit,
            kwargs=kwargs,
        )


async def run_mesh(args: argparse.Namespace, kwargs: dict[str, Any]) -> None:
    import iroh
    from distributed import Scheduler, Worker

    from .network import ALPN, Mesh

    RunConfig(
        entrypoint=args.entrypoint,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        memory_limit=args.memory_limit,
    )
    inputs = RunnerMeshConfig.model_validate({**json.loads(args.mesh), "node": args.node})
    config = inputs.model_dump(by_alias=True, exclude_unset=True)
    # FFI annotates BaseEventLoop but uses the standard AbstractEventLoop interface.
    iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())  # ty: ignore[invalid-argument-type]
    secret = bytes.fromhex(os.environ.pop("HFDASK_NODE_KEY"))
    relay_mode = (
        iroh.RelayMode.custom_from_urls(config["relays"])
        if config["relays"]
        else iroh.RelayMode.default_mode()
    )
    endpoint = await iroh.Endpoint.bind(
        iroh.EndpointOptions(
            preset=iroh.preset_n0(), secret_key=secret, alpns=[ALPN], relay_mode=relay_mode
        )
    )
    try:
        await asyncio.wait_for(endpoint.online(), timeout=120)
        peers = [
            iroh.EndpointAddr(iroh.EndpointId.from_bytes(bytes.fromhex(peer)), None, [])
            for peer in config["peers"]
        ]
        if config.get("hardware_detection"):
            await run_detected(args, config, endpoint, peers, kwargs)
            return
        services = list(range(len(peers))) + ([0] if args.scheduler_worker else [])
        async with Mesh(endpoint, peers, args.node, services=services):
            print(json.dumps({"phase": "mesh_ready", "node": args.node}), flush=True)
            if args.node:
                async with Worker(
                    "tcp://127.0.0.1:21000",
                    host="127.0.0.1",
                    port=21000 + args.node,
                    nthreads=args.threads_per_worker,
                    memory_limit=args.memory_limit,
                    dashboard_address=None,
                    death_timeout=config["startup_timeout"],
                ) as worker:
                    await worker.finished()
            else:
                async with AsyncExitStack() as stack:
                    scheduler = await stack.enter_async_context(
                        Scheduler(host="127.0.0.1", port=21000, dashboard_address=None)
                    )
                    if args.scheduler_worker:
                        await stack.enter_async_context(
                            Worker(
                                scheduler.address,
                                host="127.0.0.1",
                                port=21000 + len(peers),
                                nthreads=args.threads_per_worker,
                                memory_limit=args.memory_limit,
                                dashboard_address=None,
                            )
                        )

                    def workload() -> None:
                        with Client(scheduler.address, set_as_default=False) as client:
                            client.wait_for_workers(args.workers, timeout=config["startup_timeout"])
                            require_callable_entrypoint(args.entrypoint)(client, **kwargs)
                            # Ask workers to close; HF timeouts bound survivors on disconnect.
                            client.retire_workers(close_workers=True)

                    await asyncio.to_thread(workload)
                    print(json.dumps({"phase": "workload_complete"}), flush=True)
    finally:
        await endpoint.close()


def wait_topology(client: Client, nodes: set[int], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        infos = [worker.get("hfdask", {}) for worker in client.scheduler_info()["workers"].values()]
        counts = {node: sum(info.get("node") == node for info in infos) for node in nodes}
        expected = {info["node"]: info["workers_on_node"] for info in infos if "node" in info}
        if nodes == set(expected) and all(counts[node] == expected[node] for node in nodes):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Worker topology incomplete: observed={counts}, expected={expected}"
            )
        time.sleep(0.2)


async def run_detected(
    args: argparse.Namespace,
    config: dict[str, Any],
    endpoint: iroh.Endpoint,
    peers: list[iroh.EndpointAddr],
    kwargs: dict[str, Any],
) -> None:
    from distributed import Nanny, Scheduler

    from .hardware import (
        detect,
        nanny_service,
        service_owners,
        worker_metadata,
        worker_profiles,
        worker_service,
    )
    from .network import Mesh

    nodes = config.get("job_nodes", len(peers))
    worker_nodes = set(range(1, nodes)) | ({0} if args.scheduler_worker else set())
    profiles = (
        worker_profiles(
            detect(), config["node_flavors"][args.node], args.threads_per_worker, args.memory_limit
        )
        if args.node in worker_nodes
        else []
    )
    async with Mesh(endpoint, peers, args.node, services=service_owners(nodes)):  # noqa: SIM117
        async with AsyncExitStack() as stack:
            if args.node == 0:
                scheduler = await stack.enter_async_context(
                    Scheduler(host="127.0.0.1", port=21000, dashboard_address=None)
                )
            nannies = []
            for ordinal, profile in enumerate(profiles):
                profile = profile.copy()
                profile.update(node=args.node, workers_on_node=len(profiles))
                profile["tags"] = profile["tags"] + config.get("node_tags", [[]] * nodes)[args.node]
                gpu = profile["gpu"]
                nanny = await stack.enter_async_context(
                    Nanny(
                        "tcp://127.0.0.1:21000",
                        host="127.0.0.1",
                        worker_port=21000 + worker_service(nodes, args.node, ordinal),
                        port=21000 + nanny_service(nodes, args.node, ordinal),
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
                    {"phase": "mesh_ready", "node": args.node, "worker_processes": len(profiles)}
                ),
                flush=True,
            )
            if args.node:
                await asyncio.gather(*(nanny.finished() for nanny in nannies))
            else:
                if config.get("persistent"):
                    print(json.dumps({"phase": "accepting_clients"}), flush=True)
                    await scheduler.finished()
                    return

                def workload() -> None:
                    with Client("tcp://127.0.0.1:21000", set_as_default=False) as client:
                        wait_topology(client, worker_nodes, config["startup_timeout"])
                        require_callable_entrypoint(args.entrypoint)(client, **kwargs)
                        client.retire_workers(close_workers=True)

                await asyncio.to_thread(workload)
                print(json.dumps({"phase": "workload_complete"}), flush=True)


if __name__ == "__main__":
    main()
