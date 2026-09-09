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


async def _serve(config: ConnectionConfig, identity: Identity,
                 ready: concurrent.futures.Future[None], stop: threading.Event,
                 timeout: float) -> None:
    import iroh

    from .hardware import service_owners
    from .network import ALPN, Mesh

    iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())  # type: ignore[arg-type]
    mode = (iroh.RelayMode.custom_from_urls(config.relays)
            if config.relays else iroh.RelayMode.default_mode())
    endpoint = await iroh.Endpoint.bind(iroh.EndpointOptions(
        preset=iroh.preset_n0(), secret_key=identity.secret, alpns=[ALPN], relay_mode=mode))
    try:
        await asyncio.wait_for(endpoint.online(), timeout=timeout)
        peers = [iroh.EndpointAddr(iroh.EndpointId.from_bytes(bytes.fromhex(peer)), None, [])
                 for peer in config.peers]
        nodes = config.job_nodes
        async with Mesh(endpoint, peers, nodes, services=service_owners(nodes)):
            ready.set_result(None)
            while not stop.is_set():
                await asyncio.sleep(0.1)
    finally:
        await endpoint.close()


@contextmanager
def connect(manifest: dict[str, Any], identity: Identity, *,
            timeout: float = 1200) -> Iterator[Any]:
    """Yield a normal Dask Client. Disconnecting never cancels the HF Jobs.

    One mesh connection per host: fixed loopback ports start at 21000. Keep this
    context open while using futures. The caller owns explicit cluster shutdown.
    """
    from distributed import Client

    from .runner import wait_topology

    connection = ConnectionConfig.model_validate(manifest.get("connection"))
    ConnectConfig(timeout=timeout, connection=connection, public_id=identity.public_id())
    nodes = connection.job_nodes

    if not _connection_lock.acquire(blocking=False):
        raise RuntimeError("A mesh client is already connected in this process")
    ready: concurrent.futures.Future[None] = concurrent.futures.Future()
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
        ready.result(timeout=timeout + 5)
        with Client("tcp://127.0.0.1:21000", timeout=timeout,
                    set_as_default=False) as client:  # type: ignore[no-untyped-call]
            worker_nodes = set(range(1, nodes)) | ({0} if connection.scheduler_worker else set())
            wait_topology(client, worker_nodes, timeout)
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
