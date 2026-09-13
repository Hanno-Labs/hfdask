"""External synchronous Dask clients over the cluster's encrypted mesh."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .cluster import Identity
from .config import ConnectConfig, ConnectionConfig

_connection_lock = threading.Lock()


async def _serve(
    config: ConnectionConfig,
    identity: Identity,
    ready: concurrent.futures.Future[tuple[int, ...]],
    stop: threading.Event,
    timeout: float,
) -> None:
    import iroh

    from .hardware import service_owners
    from .network import ALPN, Mesh, fetch_worker_counts

    # FFI annotates BaseEventLoop but uses the standard AbstractEventLoop interface.
    iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())  # ty: ignore[invalid-argument-type]
    mode = (
        iroh.RelayMode.custom_from_urls(config.relays)
        if config.relays
        else iroh.RelayMode.default_mode()
    )
    endpoint = await iroh.Endpoint.bind(
        iroh.EndpointOptions(
            preset=iroh.preset_n0(), secret_key=identity.secret, alpns=[ALPN], relay_mode=mode
        )
    )
    try:
        await asyncio.wait_for(endpoint.online(), timeout=timeout)
        peers = [
            iroh.EndpointAddr(iroh.EndpointId.from_bytes(bytes.fromhex(peer)), None, [])
            for peer in config.peers
        ]
        nodes = config.job_nodes
        workers_per_node = await fetch_worker_counts(endpoint, peers[0], timeout)
        if len(workers_per_node) != nodes:
            raise ConnectionError("Live worker topology does not match the manifest")
        async with Mesh(
            endpoint,
            peers,
            nodes,
            services=service_owners(workers_per_node),
            workers_per_node=workers_per_node,
            max_connections=max(256, 4 * sum(workers_per_node)),
        ):
            ready.set_result(workers_per_node)
            while not stop.is_set():
                await asyncio.sleep(0.1)
    finally:
        await endpoint.close()


@contextmanager
def connect(
    manifest: dict[str, Any], identity: Identity, *, timeout: float = 1200
) -> Iterator[Any]:
    """Connect to a persistent cluster without taking ownership of its HF Jobs.

    Args:
        manifest: Public `hfdask.cluster.Cluster.manifest` from `boot_cluster`.
        identity: Private client identity supplied at boot, not a Job node identity.
        timeout: Positive timeout in seconds used for mesh startup, Dask connection,
            and worker readiness. These phases do not share one total deadline.

    Yields:
        A synchronous Dask client after the expected worker topology is ready.
        It is not installed as Dask's default client.

    Raises:
        ValueError: If the manifest, identity, or timeout fails validation.
        TimeoutError: If mesh startup or worker readiness exceeds its deadline.
        RuntimeError: If this process already has a mesh client or cleanup cannot finish.

    One mesh connection per host: fixed loopback ports start at 21000. Keep the
    context open while using futures. Exiting closes the client and local mesh,
    never the HF Jobs; explicitly call `hfdask.cluster.Cluster.close` to release them.
    """
    from distributed import Client

    from .runner import wait_topology

    connection = ConnectionConfig.model_validate(manifest.get("connection"))
    ConnectConfig(timeout=timeout, connection=connection, public_id=identity.public_id())

    if not _connection_lock.acquire(blocking=False):
        raise RuntimeError("A mesh client is already connected in this process")
    ready: concurrent.futures.Future[tuple[int, ...]] = concurrent.futures.Future()
    stopped = threading.Event()
    loop = asyncio.new_event_loop()
    task: asyncio.Task[None] | None = None

    def serve() -> None:
        nonlocal task
        asyncio.set_event_loop(loop)
        task = loop.create_task(_serve(connection, identity, ready, stopped, timeout))
        try:
            loop.run_until_complete(task)
        except BaseException as error:  # noqa: BLE001 - propagate startup failures across threads.
            if not ready.done():
                ready.set_exception(error)
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    thread = threading.Thread(target=serve, name="hfdask-client-mesh", daemon=True)
    try:
        thread.start()
        workers_per_node = ready.result(timeout=timeout + 5)
        with Client("tcp://127.0.0.1:21000", timeout=timeout, set_as_default=False) as client:
            wait_topology(client, workers_per_node, timeout)
            yield client
    finally:
        stopped.set()
        if thread.ident is not None:
            thread.join(timeout=5)
            if thread.is_alive() and not loop.is_closed() and task is not None:
                loop.call_soon_threadsafe(task.cancel)
                thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("Mesh cleanup did not finish; restart this client process")
        _connection_lock.release()
