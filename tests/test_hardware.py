from unittest.mock import MagicMock

import pytest

from hfdask.hardware import (
    GIB,
    _limits,
    gpu_inventory,
    worker_profiles,
)
from hfdask.routing import submit_on, workers_with
from hfdask.runner import wait_topology


def test_cgroup_limits(tmp_path):
    (tmp_path / "cpu.max").write_text("150000 100000")
    (tmp_path / "memory.max").write_text(str(2 * GIB))
    assert _limits(tmp_path) == ([1.5], [2 * GIB])


def test_profiles_partition_host_budget():
    gpus = [
        {"uuid": f"GPU-{i}", "name": "NVIDIA A100-SXM4-80GB", "vram_bytes": 80 * GIB}
        for i in range(2)
    ]
    profiles = worker_profiles({"cpu_threads": 8, "ram_bytes": 100 * GIB, "gpus": gpus}, "a100x2")
    assert len(profiles) == 8
    for profile in profiles[:2]:
        assert profile["resources"] == {
            "CPU_THREADS": 1,
            "RAM_GIB": 10,
            "GPU": 1,
            "GPU_VRAM_GIB": 72,
        }
        assert "GPU_MODEL_A100" in profile["tags"]
        assert "HAS_GPU" not in profile["resources"]
        assert profile["nthreads"] == 1
    for profile in profiles[2:]:
        assert profile["resources"] == {
            "CPU_THREADS": 1,
            "RAM_GIB": 10,
            "GPU": 0,
            "GPU_VRAM_GIB": 0,
        }
        assert profile["gpu"] is None
    assert profiles[0]["gpu"]["uuid"] != profiles[1]["gpu"]["uuid"]


def test_scheduler_reserves_exactly_one_core():
    inventory = {"cpu_threads": 4.75, "ram_bytes": 8 * GIB, "gpus": []}
    assert len(worker_profiles(inventory, "cpu-basic")) == 4
    assert len(worker_profiles(inventory, "cpu-basic", reserve_scheduler_core=True)) == 3
    one_core = {"cpu_threads": 1, "ram_bytes": 8 * GIB, "gpus": []}
    assert worker_profiles(one_core, "cpu-basic", reserve_scheduler_core=True) == []


def test_only_complete_cpu_cores_become_workers():
    partial_core = {"cpu_threads": 0.75, "ram_bytes": 8 * GIB, "gpus": []}
    with pytest.raises(RuntimeError, match="No complete CPU core"):
        worker_profiles(partial_core, "cpu-basic")
    almost_two = {"cpu_threads": 1.9, "ram_bytes": 8 * GIB, "gpus": []}
    assert len(worker_profiles(almost_two, "cpu-basic")) == 1


def test_cpu_profile_memory_cap():
    profiles = worker_profiles(
        {"cpu_threads": 2, "ram_bytes": 8 * GIB, "gpus": []}, "cpu-basic", "1 GiB"
    )
    assert len(profiles) == 2
    for profile in profiles:
        assert profile["resources"]["RAM_GIB"] == 1
        assert profile["resources"]["CPU_THREADS"] == 1
        assert profile["resources"]["GPU"] == 0
        assert profile["tags"] == ["FLAVOR_cpu-basic"]


@pytest.mark.parametrize("memory", ["0", "-1 GiB", "bad"])
def test_profile_input_contract(memory):
    with pytest.raises(ValueError):
        worker_profiles({"cpu_threads": 2, "ram_bytes": 8 * GIB, "gpus": []}, "cpu-basic", memory)


def test_profile_gpu_capacity_boundary():
    with pytest.raises(ValueError, match="Visible GPUs exceed"):
        worker_profiles({"cpu_threads": 2, "ram_bytes": 8 * GIB, "gpus": [{}] * 3}, "gpu")


def test_routing_validates_before_scheduler_lookup():
    client = MagicMock()
    with pytest.raises(TypeError, match="collection"):
        workers_with(client, tags="HAS_GPU")
    with pytest.raises(ValueError):
        workers_with(client, tags=[1])
    for placement in ({"workers": ["other"]}, {"allow_other_workers": True}):
        with pytest.raises(ValueError, match="owns worker placement"):
            submit_on(client, abs, -1, tags=(tag for tag in ["HAS_GPU"]), **placement)
    client.scheduler_info.assert_not_called()
    client.submit.assert_not_called()


def test_gpu_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-b")
    monkeypatch.setattr("shutil.which", lambda _: "nvidia-smi")
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **kw: MagicMock(
            stdout="0, GPU-a, NVIDIA T4, 15360, [N/A]\n1, GPU-b, NVIDIA T4, 15360, [N/A]\n"
        ),
    )
    assert [gpu["uuid"] for gpu in gpu_inventory()] == ["GPU-b"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-unknown")
    with pytest.raises(RuntimeError, match="visibility"):
        gpu_inventory()


def test_category_routing():
    client = MagicMock()
    client.scheduler_info.return_value = {
        "workers": {
            "cpu": {"hfdask": {"tags": ["FLAVOR_cpu-basic"]}},
            "gpu": {"hfdask": {"tags": ["HAS_GPU", "GPU_MODEL_A100"]}},
        }
    }
    assert workers_with(client, tags={"HAS_GPU", "GPU_MODEL_A100"}) == ["gpu"]
    submit_on(client, abs, -2, tags={"HAS_GPU"}, resources={"GPU": 1})
    assert client.submit.call_args.kwargs["workers"] == ["gpu"]
    assert client.submit.call_args.kwargs["allow_other_workers"] is False
    with pytest.raises(ValueError, match="No workers"):
        workers_with(client, tags={"unknown"})


def test_topology_requires_every_detected_worker():
    client = MagicMock()
    info = {"node": 1, "workers_on_node": 2}
    client.scheduler_info.return_value = {"workers": {"first": {"hfdask": info}}}
    with pytest.raises(TimeoutError):
        wait_topology(client, 2, 0, 0)
    client.scheduler_info.return_value["workers"]["second"] = {"hfdask": info}
    assert wait_topology(client, 2, 0, 0) == (0, 2)


def test_real_nanny_metadata_and_routing():
    import asyncio
    from functools import partial

    from distributed import Client, Nanny, Scheduler

    from hfdask.hardware import worker_metadata

    async def check():
        profile = {"tags": ["FLAVOR_test"], "node": 1, "workers_on_node": 1}
        async with (
            Scheduler(host="127.0.0.1", dashboard_address=None) as scheduler,
            Nanny(
                scheduler.address,
                host="127.0.0.1",
                nthreads=1,
                memory_limit="512 MiB",
                resources={"CPU_THREADS": 1},
                env={"CUDA_VISIBLE_DEVICES": ""},
                startup_information={"hfdask": partial(worker_metadata, profile=profile)},
                dashboard_address=None,
            ),
        ):

            def compute():
                with Client(scheduler.address, set_as_default=False) as client:
                    wait_topology(client, 2, 0, 10)
                    assert (
                        submit_on(
                            client, abs, -7, tags={"FLAVOR_test"}, resources={"CPU_THREADS": 1}
                        ).result(timeout=10)
                        == 7
                    )

            await asyncio.to_thread(compute)

    asyncio.run(check())
