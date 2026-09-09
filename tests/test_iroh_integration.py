"""Explicit opt-in: creates disposable endpoint keys, no discovery or relays."""

import asyncio
import os

import iroh
import pytest

from hfdask.network import ALPN, Mesh


@pytest.mark.skipif(os.environ.get("HFDASK_TEST_KEYS") != "1",
                    reason="requires explicit temporary-key test opt-in")
@pytest.mark.parametrize("services,target", [(None, 1), ([0, 1, 1], 2), ([1], 0)])
def test_real_quic_and_unknown_peer(services, target):
    async def check():
        iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
        endpoints = []
        accepted = []
        server = None
        try:
            for _ in range(3):
                endpoints.append(await iroh.Endpoint.bind(iroh.EndpointOptions(
                    preset=iroh.preset_minimal(), bind_addr="127.0.0.1:0", alpns=[ALPN])))
            roster = [endpoint.addr() for endpoint in endpoints[:2]]
            async def echo(reader, writer):
                accepted.append(True)
                data = await reader.read()
                writer.write(data[::-1])
                await writer.drain()
                writer.close()
                await writer.wait_closed()
            server = await asyncio.start_server(echo, "127.0.0.3", 23100 + target)
            async with Mesh(endpoints[0], roster, 0, base_port=23100,
                            bind_host="127.0.0.2", services=services), Mesh(
                                endpoints[1], roster, 1, base_port=23100,
                                bind_host="127.0.0.3", services=services):
                reader, writer = await asyncio.open_connection("127.0.0.2", 23100 + target)
                payload = bytes(range(256)) * 4096
                writer.write(payload)
                await writer.drain()
                writer.write_eof()
                assert await asyncio.wait_for(reader.read(), 20) == payload[::-1]
                writer.close()
                await writer.wait_closed()
                assert len(accepted) == 1
                attacker = await endpoints[2].connect(roster[1], ALPN)
                await asyncio.wait_for(attacker.closed(), 5)
                assert len(accepted) == 1
        finally:
            if server:
                server.close()
                await server.wait_closed()
            await asyncio.gather(*(endpoint.close() for endpoint in endpoints))
    asyncio.run(check())
