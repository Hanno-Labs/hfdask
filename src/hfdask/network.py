"""Authenticated Iroh QUIC links presented to Dask as loopback TCP peers.

Encryption, peer identity verification, NAT traversal, and relay transport are
provided by Iroh. This module only bridges bounded byte streams. Each remote
identity can access only this node's registered Dask services, not arbitrary TCP.
"""

from __future__ import annotations

import asyncio
import logging
import resource
from collections.abc import Coroutine, Sequence
from typing import Any, Self

import iroh

from .config import MeshConfig, WorkerTopologyConfig

ALPN = b"hfdask/1"
CHUNK_SIZE = 65536
logger = logging.getLogger(__name__)


def encode_worker_counts(workers_per_node: tuple[int, ...]) -> bytes:
    counts = WorkerTopologyConfig(workers_per_node=workers_per_node).workers_per_node
    if len(counts) > 65535 or any(count > 2**32 - 1 for count in counts):
        raise ValueError("Worker topology is too large for the mesh protocol")
    return len(counts).to_bytes(2, "big") + b"".join(count.to_bytes(4, "big") for count in counts)


async def read_worker_counts(stream: iroh.BiStream) -> tuple[int, ...]:
    size = int.from_bytes(await stream.recv().read_exact(2), "big")
    if size == 0:
        raise ConnectionError("Worker topology cannot be empty")
    counts = []
    for _ in range(size):
        counts.append(int.from_bytes(await stream.recv().read_exact(4), "big"))
    return WorkerTopologyConfig(workers_per_node=tuple(counts)).workers_per_node


async def exchange_worker_counts(
    endpoint: iroh.Endpoint,
    peers: Sequence[iroh.EndpointAddr],
    index: int,
    local_count: int,
    nodes: int,
    timeout: float,
) -> tuple[int, ...]:
    """Exchange detected per-node worker counts before allocating mesh services."""
    if not 0 <= index < nodes <= len(peers):
        raise ValueError("Worker-count exchange does not match the Job roster")
    encode_worker_counts((local_count,))
    job_peers = peers[:nodes]
    scheduler_id = job_peers[0].id().to_bytes()

    async with asyncio.timeout(timeout):
        if index:
            connection = await endpoint.connect(job_peers[0], ALPN)
            try:
                require_expected_peer(connection.remote_id().to_bytes(), scheduler_id)
                stream = await connection.open_bi()
                await stream.send().write_all(b"W" + local_count.to_bytes(4, "big"))
                if await stream.recv().read_exact(1) != b"K":
                    raise ConnectionError("Scheduler rejected worker topology")
                counts = await read_worker_counts(stream)
                if len(counts) != nodes or counts[index] != local_count:
                    raise ConnectionError("Scheduler returned inconsistent worker topology")
                await stream.send().write_all(b"A")
                await stream.send().finish()
                return counts
            finally:
                connection.close(0, b"worker topology exchanged")

        counts: list[int | None] = [local_count, *([None] * (nodes - 1))]
        streams: list[iroh.BiStream] = []
        connections: list[Any] = []
        job_ids = {peer.id().to_bytes(): node for node, peer in enumerate(job_peers)}
        try:
            while any(count is None for count in counts):
                incoming = await endpoint.accept_next()
                if incoming is None:
                    raise ConnectionError("Endpoint closed during worker-count exchange")
                connection = await (await incoming.accept()).connect()
                remote_id = connection.remote_id().to_bytes()
                node = job_ids.get(remote_id)
                if node in (None, 0) or counts[node] is not None:
                    connection.close(1, b"unexpected worker topology reporter")
                    continue
                stream = await connection.accept_bi()
                if await stream.recv().read_exact(1) != b"W":
                    connection.close(1, b"invalid worker topology preamble")
                    continue
                counts[node] = int.from_bytes(await stream.recv().read_exact(4), "big")
                streams.append(stream)
                connections.append(connection)
            result = tuple(count for count in counts if count is not None)
            payload = encode_worker_counts(result)
            for stream in streams:
                await stream.send().write_all(b"K" + payload)
                await stream.send().finish()
                if await stream.recv().read_exact(1) != b"A":
                    raise ConnectionError("Worker did not acknowledge topology")
            return result
        finally:
            for connection in connections:
                connection.close(0, b"worker topology exchanged")


async def fetch_worker_counts(
    endpoint: iroh.Endpoint, scheduler: iroh.EndpointAddr, timeout: float
) -> tuple[int, ...]:
    """Fetch the live worker topology when an external client joins later."""
    async with asyncio.timeout(timeout):
        connection = await endpoint.connect(scheduler, ALPN)
        try:
            require_expected_peer(connection.remote_id().to_bytes(), scheduler.id().to_bytes())
            stream = await connection.open_bi()
            await stream.send().write_all(b"C")
            if await stream.recv().read_exact(1) != b"K":
                raise ConnectionError("Scheduler did not provide worker topology")
            counts = await read_worker_counts(stream)
            await stream.send().write_all(b"A")
            await stream.send().finish()
            return counts
        finally:
            connection.close(0, b"worker topology fetched")


def require_expected_peer(actual: bytes, expected: bytes) -> None:
    if actual != expected:
        raise PermissionError("Unexpected peer identity")


def authorized(peer_id: bytes, roster: Sequence[bytes]) -> bool:
    return peer_id in roster


async def bridge(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, stream: iroh.BiStream
) -> None:
    """Preserve backpressure and half-close; never buffer the whole payload."""

    async def upload() -> None:
        while data := await reader.read(CHUNK_SIZE):
            await stream.send().write_all(data)
        await stream.send().finish()
        await stream.send().stopped()

    async def download() -> None:
        while data := await stream.recv().read(CHUNK_SIZE):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()

    tasks = [asyncio.create_task(upload()), asyncio.create_task(download())]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        writer.close()
        await writer.wait_closed()


class Mesh:
    """One owned endpoint per job, with identical peer-port aliases on each job.

    Dask on peer i listens/advertises 127.0.0.1:(base_port+i). On all other
    machines that address is a proxy to peer i's authenticated Iroh endpoint.
    The caller owns endpoint creation and its key/relay policy.
    Optional services maps consecutive port offsets to owning node indices,
    allowing a node to host both scheduler and worker without sharing listeners.
    """

    def __init__(
        self,
        endpoint: iroh.Endpoint,
        peers: Sequence[iroh.EndpointAddr],
        index: int,
        *,
        base_port: int = 21000,
        max_connections: int = 256,
        bind_host: str = "127.0.0.1",
        services: Sequence[int] | None = None,
        workers_per_node: tuple[int, ...] | None = None,
    ) -> None:
        owners = tuple(range(len(peers))) if services is None else tuple(services)
        ids = [peer.id().to_bytes() for peer in peers]
        MeshConfig(
            index=index,
            base_port=base_port,
            max_connections=max_connections,
            bind_host=bind_host,
            services=owners,
            peers=tuple(ids),
            endpoint_id=endpoint.id().to_bytes(),
        )
        self.endpoint = endpoint
        self.peers = tuple(peers)
        self.ids = ids
        self.index = index
        self.base_port = base_port
        self.max_connections = max_connections
        self.bind_host = bind_host
        self.services = owners
        self.workers_per_node = (
            None
            if workers_per_node is None
            else WorkerTopologyConfig(workers_per_node=workers_per_node).workers_per_node
        )
        self.servers: list[asyncio.Server] = []
        self.tasks: set[asyncio.Task[None]] = set()
        self.accept_task: asyncio.Task[None] | None = None

    def spawn(self, coroutine: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)

        def finished(done: asyncio.Task[None]) -> None:
            self.tasks.discard(done)
            if not done.cancelled() and (error := done.exception()) is not None:
                logger.warning("mesh stream failed: %s", type(error).__name__)

        task.add_done_callback(finished)

    def require_file_descriptor_budget(self) -> None:
        proxy_count = sum(owner != self.index for owner in self.services)
        required = proxy_count + 2 * self.max_connections + 64
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit == resource.RLIM_INFINITY or soft_limit >= required:
            return
        if hard_limit != resource.RLIM_INFINITY and hard_limit < required:
            raise RuntimeError("File descriptor hard limit is too low for the configured mesh")
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard_limit))
        except (OSError, ValueError) as error:
            raise RuntimeError(
                "Could not raise the file descriptor limit for the configured mesh"
            ) from error

    def require_owned_service(self, service: int) -> None:
        if not 0 <= service < len(self.services) or self.services[service] != self.index:
            raise PermissionError("Service is not owned by this node")

    async def __aenter__(self) -> Self:
        self.require_file_descriptor_budget()
        try:
            for index, owner in enumerate(self.services):
                if owner == self.index:
                    continue

                def accepted(
                    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, target: int = index
                ) -> None:
                    if len(self.tasks) >= self.max_connections:
                        writer.close()
                    else:
                        self.spawn(self.outgoing(target, reader, writer))

                self.servers.append(
                    await asyncio.start_server(accepted, self.bind_host, self.base_port + index)
                )
            self.accept_task = asyncio.create_task(self.accept())
            return self
        except BaseException:
            await self.close()
            raise

    async def outgoing(
        self, index: int, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        connection = None
        try:
            async with asyncio.timeout(30):
                owner = self.services[index]
                connection = await self.endpoint.connect(self.peers[owner], ALPN)
                require_expected_peer(connection.remote_id().to_bytes(), self.ids[owner])
                stream = await connection.open_bi()
                await stream.send().write_all(b"D" + index.to_bytes(2, "big"))
                if await stream.recv().read_exact(1) != b"K":
                    raise ConnectionError("Peer did not open its Dask listener")
            logger.warning("mesh link authenticated node=%s target=%s", self.index, index)
            await bridge(reader, writer, stream)
        finally:
            writer.close()
            await writer.wait_closed()
            if connection is not None:
                connection.close(0, b"stream closed")

    async def accept(self) -> None:
        while incoming := await self.endpoint.accept_next():
            if len(self.tasks) >= self.max_connections:
                await incoming.refuse()
            else:
                self.spawn(self.incoming(incoming))

    async def incoming(self, incoming: iroh.Incoming) -> None:
        connection = None
        try:
            async with asyncio.timeout(30):
                connection = await (await incoming.accept()).connect()
                if not authorized(connection.remote_id().to_bytes(), self.ids):
                    connection.close(1, b"not a cluster member")
                    return
                stream = await connection.accept_bi()
                preamble = await stream.recv().read_exact(1)
                if preamble == b"C":
                    if self.workers_per_node is None:
                        raise ConnectionError("Worker topology is unavailable")
                    await stream.send().write_all(
                        b"K" + encode_worker_counts(self.workers_per_node)
                    )
                    await stream.send().finish()
                    if await stream.recv().read_exact(1) != b"A":
                        raise ConnectionError("Client did not acknowledge worker topology")
                    return
                if preamble != b"D":
                    raise ConnectionError("Invalid stream preamble")
                service = int.from_bytes(await stream.recv().read_exact(2), "big")
                self.require_owned_service(service)
                reader, writer = await asyncio.open_connection(
                    self.bind_host, self.base_port + service
                )
            try:
                await stream.send().write_all(b"K")
                await bridge(reader, writer, stream)
            finally:
                writer.close()
                await writer.wait_closed()
        finally:
            if connection is not None:
                connection.close(0, b"stream closed")

    async def close(self) -> None:
        for server in self.servers:
            server.close()
        await asyncio.gather(*(server.wait_closed() for server in self.servers))
        if self.accept_task is not None:
            self.accept_task.cancel()
            await asyncio.gather(self.accept_task, return_exceptions=True)
        tasks = list(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def __aexit__(self, *args: object) -> None:
        await self.close()
