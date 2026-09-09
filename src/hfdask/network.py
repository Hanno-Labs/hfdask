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

from .config import MeshConfig

ALPN = b"hfdask/1"
CHUNK_SIZE = 65536
logger = logging.getLogger(__name__)


def require_expected_peer(actual: bytes, expected: bytes) -> None:
    if actual != expected:
        raise PermissionError("Unexpected peer identity")


def authorized(peer_id: bytes, roster: Sequence[bytes]) -> bool:
    return peer_id in roster


async def bridge(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                 stream: iroh.BiStream) -> None:
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

    def __init__(self, endpoint: iroh.Endpoint, peers: Sequence[iroh.EndpointAddr],
                 index: int, *, base_port: int = 21000, max_connections: int = 256,
                 bind_host: str = "127.0.0.1",
                 services: Sequence[int] | None = None) -> None:
        owners = tuple(range(len(peers))) if services is None else tuple(services)
        ids = [peer.id().to_bytes() for peer in peers]
        MeshConfig(index=index, base_port=base_port, max_connections=max_connections,
                   bind_host=bind_host, services=owners, peers=tuple(ids),
                   endpoint_id=endpoint.id().to_bytes())
        self.endpoint = endpoint
        self.peers = tuple(peers)
        self.ids = ids
        self.index = index
        self.base_port = base_port
        self.max_connections = max_connections
        self.bind_host = bind_host
        self.services = owners
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
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit != resource.RLIM_INFINITY and proxy_count + 2 * self.max_connections + 64 > soft_limit:
            raise RuntimeError("File descriptor limit is too low for the configured mesh")

    def require_owned_service(self, service: int) -> None:
        if not 0 <= service < len(self.services) or self.services[service] != self.index:
            raise PermissionError("Service is not owned by this node")

    async def __aenter__(self) -> Self:
        self.require_file_descriptor_budget()
        try:
            for index, owner in enumerate(self.services):
                if owner == self.index:
                    continue
                def accepted(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                             target: int = index) -> None:
                    if len(self.tasks) >= self.max_connections:
                        writer.close()
                    else:
                        self.spawn(self.outgoing(target, reader, writer))
                self.servers.append(await asyncio.start_server(
                    accepted, self.bind_host, self.base_port + index))
            self.accept_task = asyncio.create_task(self.accept())
            return self
        except BaseException:
            await self.close()
            raise

    async def outgoing(self, index: int, reader: asyncio.StreamReader,
                       writer: asyncio.StreamWriter) -> None:
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
                if await stream.recv().read_exact(1) != b"D":
                    raise ConnectionError("Invalid stream preamble")
                service = int.from_bytes(await stream.recv().read_exact(2), "big")
                self.require_owned_service(service)
                reader, writer = await asyncio.open_connection(
                    self.bind_host, self.base_port + service)
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
