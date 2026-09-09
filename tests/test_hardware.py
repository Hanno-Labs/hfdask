from unittest.mock import MagicMock

import pytest

from hfdask.hardware import (
    GIB,
    _limits,
    gpu_inventory,
    nanny_service,
    service_owners,
    worker_profiles,
    worker_service,
)
from hfdask.routing import submit_on, workers_with
from hfdask.runner import wait_topology


def test_cgroup_limits(tmp_path):
    (tmp_path / "cpu.max").write_text("150000 100000")
    (tmp_path / "memory.max").write_text(str(2 * GIB))
    assert _limits(tmp_path) == ([1.5], [2 * GIB])


def test_profiles_partition_host_budget():
    gpus = [{"uuid": f"GPU-{i}", "name": "NVIDIA A100-SXM4-80GB",
             "vram_bytes": 80 * GIB} for i in range(2)]
    profiles = worker_profiles({"cpu_threads": 8, "ram_bytes": 100 * GIB, "gpus": gpus},
                               "a100x2", 8)
    assert len(profiles) == 2
    for profile in profiles:
        assert profile["resources"] == {"CPU_THREADS": 4, "RAM_GIB": 40,
                                        "GPU": 1, "GPU_VRAM_GIB": 72}
        assert "GPU_MODEL_A100" in profile["tags"]
        assert "HAS_GPU" not in profile["resources"]
    assert profiles[0]["gpu"]["uuid"] != profiles[1]["gpu"]["uuid"]


def test_cpu_profile_memory_cap():
    profile, = worker_profiles({"cpu_threads": 2, "ram_bytes": 8 * GIB, "gpus": []},
                               "cpu-basic", 1, "1 GiB")
    assert profile["resources"]["RAM_GIB"] == 1
    assert profile["resources"]["CPU_THREADS"] == 1
    assert profile["resources"]["GPU"] == 0
    assert profile["tags"] == ["FLAVOR_cpu-basic"]


def test_gpu_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b")
    monkeypatch.setattr("shutil.which", lambda _: "nvidia-smi")
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: MagicMock(
        stdout="0, GPU-a, NVIDIA T4, 15360, [N/A]\n1, GPU-b, NVIDIA T4, 15360, [N/A]\n"))
    assert [gpu["uuid"] for gpu in gpu_inventory()] == ["GPU-b"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-unknown")
    with pytest.raises(RuntimeError, match="visibility"):
        gpu_inventory()


def test_service_layout():
    owners = service_owners(6)
    seen = {0}
    for node in range(6):
        for ordinal in range(16):
            for service in (worker_service(6, node, ordinal), nanny_service(6, node, ordinal)):
                assert service not in seen
                assert owners[service] == node
                seen.add(service)


def test_category_routing():
    client = MagicMock()
    client.scheduler_info.return_value = {"workers": {
        "cpu": {"hfdask": {"tags": ["FLAVOR_cpu-basic"]}},
        "gpu": {"hfdask": {"tags": ["HAS_GPU", "GPU_MODEL_A100"]}},
    }}
    assert workers_with(client, tags={"HAS_GPU", "GPU_MODEL_A100"}) == ["gpu"]
    submit_on(client, abs, -2, tags={"HAS_GPU"}, resources={"GPU": 1})
    assert client.submit.call_args.kwargs["workers"] == ["gpu"]
    assert client.submit.call_args.kwargs["allow_other_workers"] is False
    with pytest.raises(ValueError, match="No workers"):
        workers_with(client, tags={"unknown"})


def test_topology_requires_every_gpu_worker():
    client = MagicMock()
    info = {"node": 1, "workers_on_node": 2}
    client.scheduler_info.return_value = {"workers": {"first": {"hfdask": info}}}
    with pytest.raises(TimeoutError):
        wait_topology(client, {1}, 0)
    client.scheduler_info.return_value["workers"]["second"] = {"hfdask": info}
    wait_topology(client, {1}, 0)


def test_real_nanny_metadata_and_routing():
    import asyncio
    from functools import partial

    from distributed import Client, Nanny, Scheduler

    from hfdask.hardware import worker_metadata

    async def check():
        profile = {"tags": ["FLAVOR_test"], "node": 1, "workers_on_node": 1}
        async with (
            Scheduler(host="127.0.0.1", dashboard_address=None) as scheduler,
            Nanny(scheduler.address, host="127.0.0.1", nthreads=1,
                             memory_limit="512 MiB", resources={"CPU_THREADS": 1},
                             env={"CUDA_VISIBLE_DEVICES": ""},
                             startup_information={"hfdask": partial(worker_metadata, profile=profile)},
                             dashboard_address=None),
        ):
            def compute():
                with Client(scheduler.address, set_as_default=False) as client:
                    wait_topology(client, {1}, 10)
                    assert submit_on(client, abs, -7, tags={"FLAVOR_test"},
                                     resources={"CPU_THREADS": 1}).result(timeout=10) == 7
            await asyncio.to_thread(compute)
    asyncio.run(check())
