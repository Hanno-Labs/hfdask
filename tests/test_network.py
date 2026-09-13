import asyncio
import resource
from unittest.mock import AsyncMock, MagicMock

import pytest

from hfdask.network import (
    Mesh,
    authorized,
    bridge,
    encode_worker_counts,
    read_worker_counts,
)


def peer(value):
    result = MagicMock()
    result.id.return_value.to_bytes.return_value = value
    return result


def test_allowlist():
    assert authorized(b"known", [b"known"])
    assert not authorized(b"stranger", [b"known"])


def test_worker_topology_round_trip():
    async def check():
        stream = MagicMock()
        stream.recv.return_value.read_exact = AsyncMock(
            side_effect=[(3).to_bytes(2, "big"), *(n.to_bytes(4, "big") for n in (0, 2, 17))]
        )
        assert await read_worker_counts(stream) == (0, 2, 17)

    assert encode_worker_counts((0, 2, 17)) == (
        b"\x00\x03\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x11"
    )
    asyncio.run(check())


def test_scheduler_serves_live_worker_topology(monkeypatch):
    async def check():
        mesh = Mesh(
            peer(b"own"),
            [peer(b"own"), peer(b"client")],
            0,
            services=[0],
            workers_per_node=(0, 2),
        )
        stream = MagicMock()
        stream.recv.return_value.read_exact = AsyncMock(side_effect=[b"C", b"A"])
        stream.send.return_value.write_all = AsyncMock()
        stream.send.return_value.finish = AsyncMock()
        connection = MagicMock()
        connection.remote_id.return_value.to_bytes.return_value = b"client"
        connection.accept_bi = AsyncMock(return_value=stream)
        pending = MagicMock()
        pending.connect = AsyncMock(return_value=connection)
        incoming = MagicMock()
        incoming.accept = AsyncMock(return_value=pending)
        opened = AsyncMock()
        monkeypatch.setattr(asyncio, "open_connection", opened)
        await mesh.incoming(incoming)
        stream.send.return_value.write_all.assert_awaited_once_with(
            b"K" + encode_worker_counts((0, 2))
        )
        opened.assert_not_called()

    asyncio.run(check())


def test_roster_must_match_endpoint():
    with pytest.raises(ValueError):
        Mesh(peer(b"wrong"), [peer(b"known")], 0)


def test_reject_before_dask_connection(monkeypatch):
    async def check():
        mesh = Mesh(peer(b"own"), [peer(b"own"), peer(b"other")], 0)
        connection = MagicMock()
        connection.remote_id.return_value.to_bytes.return_value = b"attacker"
        pending = MagicMock()
        pending.connect = AsyncMock(return_value=connection)
        incoming = MagicMock()
        incoming.accept = AsyncMock(return_value=pending)
        opened = AsyncMock()
        monkeypatch.setattr(asyncio, "open_connection", opened)
        await mesh.incoming(incoming)
        opened.assert_not_called()
        connection.close.assert_any_call(1, b"not a cluster member")

    asyncio.run(check())


def test_bridge_half_close():
    async def check():
        reader = asyncio.StreamReader()
        reader.feed_data(b"request")
        reader.feed_eof()
        writer = MagicMock()
        writer.drain = AsyncMock()
        writer.wait_closed = AsyncMock()
        stream = MagicMock()
        stream.send.return_value.write_all = AsyncMock()
        stream.send.return_value.finish = AsyncMock()
        stream.send.return_value.stopped = AsyncMock()
        stream.recv.return_value.read = AsyncMock(side_effect=[b"reply", b""])
        await bridge(reader, writer, stream)
        stream.send.return_value.write_all.assert_awaited_once_with(b"request")
        stream.send.return_value.finish.assert_awaited_once()
        writer.write.assert_called_once_with(b"reply")
        writer.close.assert_called_once()

    asyncio.run(check())


def test_non_loopback_rejected():
    with pytest.raises(ValueError, match="loopback"):
        Mesh(peer(b"own"), [peer(b"own")], 0, bind_host="0.0.0.0")


def test_mesh_raises_soft_file_descriptor_limit(monkeypatch):
    changed = MagicMock()
    monkeypatch.setattr("hfdask.network.resource.getrlimit", lambda _: (100, 1000))
    monkeypatch.setattr("hfdask.network.resource.setrlimit", changed)
    mesh = Mesh(peer(b"own"), [peer(b"own"), peer(b"other")], 0, services=[0, 1, 0])
    mesh.require_file_descriptor_budget()
    changed.assert_called_once_with(resource.RLIMIT_NOFILE, (577, 1000))


def test_mesh_rejects_insufficient_hard_file_descriptor_limit(monkeypatch):
    monkeypatch.setattr("hfdask.network.resource.getrlimit", lambda _: (100, 500))
    with pytest.raises(RuntimeError, match="hard limit"):
        Mesh(
            peer(b"own"), [peer(b"own"), peer(b"other")], 0, services=[0, 1, 0]
        ).require_file_descriptor_budget()


@pytest.mark.parametrize("service", [1, 3, 65535])
def test_service_rejected_before_local_connection(monkeypatch, service):
    async def check():
        mesh = Mesh(peer(b"own"), [peer(b"own"), peer(b"other")], 0, services=[0, 1, 0])
        stream = MagicMock()
        stream.recv.return_value.read_exact = AsyncMock(
            side_effect=[b"D", service.to_bytes(2, "big")]
        )
        connection = MagicMock()
        connection.remote_id.return_value.to_bytes.return_value = b"other"
        connection.accept_bi = AsyncMock(return_value=stream)
        pending = MagicMock()
        pending.connect = AsyncMock(return_value=connection)
        incoming = MagicMock()
        incoming.accept = AsyncMock(return_value=pending)
        opened = AsyncMock()
        monkeypatch.setattr(asyncio, "open_connection", opened)
        with pytest.raises(PermissionError, match="Service"):
            await mesh.incoming(incoming)
        opened.assert_not_called()

    asyncio.run(check())
