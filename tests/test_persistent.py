import asyncio
import json
from unittest.mock import MagicMock

import pytest
from huggingface_hub import HfApi

from hfdask.cluster import Cluster, Identity, boot_cluster
from hfdask.jobs import JobSpec


def test_boot_and_restore(monkeypatch):
    monkeypatch.setattr(Identity, "public_id", lambda self: "public-" + str(self.secret[0]))
    api = MagicMock(spec=HfApi)
    api.run_job.side_effect = [MagicMock(id="scheduler"), MagicMock(id="worker")]
    nodes = [Identity(b"a" * 32), Identity(b"b" * 32)]
    external = Identity(b"c" * 32)
    snapshots = []
    cluster = boot_cluster(
        JobSpec("example", "image", workers=2),
        nodes,
        client_identity=external,
        public_relays=True,
        scheduler_worker=True,
        api=api,
        on_submitted=lambda c: snapshots.append(c.manifest()),
    )
    manifest = json.loads(json.dumps(cluster.manifest()))
    assert manifest["connection"]["job_nodes"] == 2
    assert manifest["connection"]["peers"] == ["public-97", "public-98", "public-99"]
    assert external.secret.hex() not in json.dumps(manifest)
    assert len(snapshots[0]["jobs"]) == 1
    for index, call in enumerate(api.run_job.call_args_list):
        assert call.kwargs["secrets"] == {"HFDASK_NODE_KEY": nodes[index].secret.hex()}
    restored = Cluster.from_manifest(manifest, api=api)
    api.inspect_job.return_value.status.stage = "CANCELED"
    restored.close(timeout=0)
    assert api.cancel_job.call_count == 2
    assert api.run_job.call_count == 2


def test_boot_rejects_workload():
    with pytest.raises(ValueError, match="entrypoint"):
        boot_cluster(
            JobSpec("example", "image", "workload:run"), [], client_identity=Identity(b"a" * 32)
        )


def test_batch_requires_entrypoint():
    with pytest.raises(ValueError, match="entrypoint"):
        JobSpec("example", "image").command()


def test_client_connect_disconnect_keeps_scheduler(monkeypatch):
    """Real Dask lifecycle with transport stubbed; QUIC has separate integration tests."""
    from distributed import Client, LocalCluster

    from hfdask.client import connect

    async def transport(config, identity, ready, stop, timeout):
        ready.set_result((1,))
        while not stop.is_set():
            await asyncio.sleep(0.01)

    monkeypatch.setattr("hfdask.client._serve", transport)
    monkeypatch.setattr(Identity, "public_id", lambda self: "client")
    manifest = {
        "connection": {
            "schema": 1,
            "persistent": True,
            "job_nodes": 1,
            "peers": ["server", "client"],
            "scheduler_worker": True,
            "public_relays": True,
            "relays": [],
        }
    }
    with LocalCluster(
        n_workers=1,
        threads_per_worker=1,
        processes=False,
        host="127.0.0.1",
        scheduler_port=21000,
        dashboard_address=None,
        startup_information={"hfdask": lambda worker: {"node": 0, "workers_on_node": 1}},
    ) as cluster:
        for _ in range(2):
            with connect(manifest, Identity(b"a" * 32), timeout=5) as client:
                assert client.submit(abs, -17).result() == 17
                with pytest.raises(RuntimeError, match="already connected"):  # noqa: SIM117
                    with connect(manifest, Identity(b"a" * 32), timeout=5):
                        pass
        with Client(cluster) as client:
            assert client.submit(sum, [1, 2, 3]).result() == 6
        assert cluster.status.name == "running"


def test_client_rejects_wrong_identity(monkeypatch):
    from hfdask.client import connect

    monkeypatch.setattr(Identity, "public_id", lambda self: "wrong")
    with pytest.raises(ValueError, match="identity"):  # noqa: SIM117
        with connect(
            {
                "connection": {
                    "schema": 1,
                    "persistent": True,
                    "job_nodes": 1,
                    "peers": ["server", "client"],
                    "scheduler_worker": True,
                    "public_relays": True,
                    "relays": [],
                }
            },
            Identity(b"a" * 32),
        ):
            pass


def persistent_server():
    from argparse import Namespace
    from unittest.mock import patch

    from hfdask.runner import run_detected

    class NoMesh:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    config = {
        "job_nodes": 1,
        "node_flavors": ["cpu-basic"],
        "node_tags": [[]],
        "startup_timeout": 20,
        "persistent": True,
    }
    args = Namespace(
        node=0,
        scheduler_worker=True,
        threads_per_worker=1,
        memory_limit="256MiB",
        entrypoint="missing:must_not_run",
    )

    async def topology(*args, **kwargs):
        return (1,)

    inventory = {"cpu_threads": 2, "ram_bytes": 512 * 1024**2, "gpus": []}
    with (
        patch("hfdask.network.Mesh", NoMesh),
        patch("hfdask.network.exchange_worker_counts", topology),
        patch("hfdask.hardware.detect", return_value=inventory),
    ):
        asyncio.run(run_detected(args, config, None, [None, None], {}))


def test_persistent_runner_survives_separate_clients():
    import multiprocessing
    import subprocess
    import sys
    import time

    server = multiprocessing.get_context("spawn").Process(target=persistent_server)
    server.start()
    program = """
from distributed import Client
from hfdask.runner import wait_topology
with Client('tcp://127.0.0.1:21000', timeout=20) as client:
    wait_topology(client, (1,), 20)
    assert client.submit(abs, -29).result() == 29
"""
    try:
        for _ in range(2):
            subprocess.run([sys.executable, "-c", program], check=True, timeout=45)
            time.sleep(0.2)
            assert server.is_alive()
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from distributed import Client; "
                    "c=Client('tcp://127.0.0.1:21000'); c.shutdown(); c.close()"
                ),
            ],
            check=True,
            timeout=30,
        )
        server.join(15)
        assert server.exitcode == 0
    finally:
        if server.is_alive():
            server.terminate()
            server.join(5)


def encrypted_server(ready):
    import iroh
    from distributed import Scheduler, Worker

    from hfdask.network import ALPN, Mesh

    async def serve():
        iroh.iroh_ffi.uniffi_set_event_loop(asyncio.get_running_loop())
        endpoint = await iroh.Endpoint.bind(
            iroh.EndpointOptions(
                preset=iroh.preset_minimal(),
                secret_key=b"a" * 32,
                bind_addr="127.0.0.1:23111",
                alpns=[ALPN],
            )
        )
        peers = [
            endpoint.addr(),
            iroh.EndpointAddr(iroh.SecretKey.from_bytes(b"b" * 32).public(), None, []),
        ]
        try:
            async with Mesh(  # noqa: SIM117
                endpoint,
                peers,
                0,
                services=[0, 0, 0],
                workers_per_node=(1,),
                bind_host="127.0.0.2",
            ):
                async with Scheduler(
                    host="127.0.0.2", port=21000, dashboard_address=None
                ) as scheduler:
                    async with Worker(
                        scheduler.address,
                        host="127.0.0.2",
                        port=21001,
                        contact_address="tcp://127.0.0.1:21001",
                        nthreads=1,
                        dashboard_address=None,
                        startup_information={"hfdask": lambda w: {"node": 0, "workers_on_node": 1}},
                    ):
                        ready.set()
                        await scheduler.finished()
        finally:
            await endpoint.close()

    asyncio.run(serve())


def test_real_encrypted_background_client(monkeypatch):
    """Known disposable loopback keys; no public discovery, relays, or HF Jobs."""
    import multiprocessing

    import iroh

    from hfdask.client import connect

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    server = context.Process(target=encrypted_server, args=(ready,))
    server.start()
    original_addr = iroh.EndpointAddr
    original_bind = iroh.Endpoint.bind

    async def local_bind(options):
        return await original_bind(
            iroh.EndpointOptions(
                preset=iroh.preset_minimal(),
                secret_key=options.secret_key,
                bind_addr="127.0.0.1:0",
                alpns=options.alpns,
            )
        )

    async def online(endpoint):
        pass

    # Only discovery changes: use the server's known loopback address.
    monkeypatch.setattr(iroh.Endpoint, "bind", local_bind)
    monkeypatch.setattr(iroh.Endpoint, "online", online)
    monkeypatch.setattr(
        iroh,
        "EndpointAddr",
        lambda key, relay, addresses: original_addr(
            key,
            None,
            ["127.0.0.1:23111"]
            if key.to_bytes() == iroh.SecretKey.from_bytes(b"a" * 32).public().to_bytes()
            else [],
        ),
    )
    manifest = {
        "connection": {
            "schema": 1,
            "persistent": True,
            "job_nodes": 1,
            "peers": [Identity(b"a" * 32).public_id(), Identity(b"b" * 32).public_id()],
            "scheduler_worker": True,
            "public_relays": True,
            "relays": [],
        }
    }
    try:
        assert ready.wait(20)
        for _ in range(2):
            with connect(manifest, Identity(b"b" * 32), timeout=10) as client:
                assert client.submit(abs, -37).result(timeout=10) == 37
            assert server.is_alive()
    finally:
        if server.is_alive():
            server.terminate()
        server.join(5)
