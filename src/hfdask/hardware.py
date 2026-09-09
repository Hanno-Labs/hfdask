"""Runtime hardware inventory and conservative per-process Dask reservations."""

from __future__ import annotations

import csv
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import ServiceConfig, WorkerConfig

MAX_WORKERS_PER_NODE = 16
GIB = 1024 ** 3


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError):
        return ""


def _limits(root: Path) -> tuple[list[float], list[int]]:
    """Read v2 limits at the mounted root and visible ancestors of our cgroup."""
    directories = {root}
    for line in _read(Path("/proc/self/cgroup")).splitlines():
        if line.startswith("0::"):
            current = root / line[3:].lstrip("/")
            if ".." not in current.parts:
                while current != root and root in current.parents:
                    directories.add(current)
                    current = current.parent
    cpus, memory = [], []
    for directory in directories:
        quota = _read(directory / "cpu.max").split()
        if len(quota) == 2 and quota[0] != "max" and int(quota[1]) > 0:
            cpus.append(int(quota[0]) / int(quota[1]))
        ram = _read(directory / "memory.max")
        if ram and ram != "max":
            memory.append(int(ram))
    # Common cgroup v1 mount layout.
    v1_quota = _read(root / "cpu/cpu.cfs_quota_us")
    period = _read(root / "cpu/cpu.cfs_period_us")
    if v1_quota and period and int(v1_quota) > 0 and int(period) > 0:
        cpus.append(int(v1_quota) / int(period))
    ram = _read(root / "memory/memory.limit_in_bytes")
    if ram:
        memory.append(int(ram))
    return cpus, memory


def gpu_inventory() -> list[dict[str, Any]]:
    """Discover NVIDIA GPUs and honor inherited device visibility; fail on ambiguity."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip() in {"", "-1"}:
        return []
    executable = shutil.which("nvidia-smi")
    if executable is None:
        if visible:
            raise RuntimeError("GPU visibility is set but nvidia-smi is unavailable")
        return []
    output = subprocess.run(
        [executable, "--query-gpu=index,uuid,name,memory.total,mig.mode.current",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout
    inventory: list[dict[str, Any]] = []
    for row in csv.reader(output.splitlines()):
        index, uuid, name, memory, mig = (field.strip() for field in row)
        if mig.lower() == "enabled":
            raise RuntimeError("MIG partitions are not yet supported; refusing pooled GPU capacity")
        inventory.append({"index": index, "uuid": uuid, "name": name,
                          "vram_bytes": int(float(memory) * 1024 ** 2)})
    if visible is None:
        return inventory
    selected = []
    for token in visible.split(","):
        token = token.strip()
        matches = [gpu for gpu in inventory if gpu["index"] == token
                   or (token.startswith("GPU-") and gpu["uuid"].startswith(token))]
        if len(matches) != 1 or matches[0] in selected:
            raise RuntimeError("Unsupported or ambiguous CUDA visibility (including MIG)")
        selected.append(matches[0])
    return selected


def detect() -> dict[str, Any]:
    cpu_count = float(len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                      else (os.cpu_count() or 1))
    ram = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    cpu_limits, ram_limits = _limits(Path("/sys/fs/cgroup"))
    cpus = min([cpu_count, *cpu_limits])
    memory = min([ram, *ram_limits])
    if cpus <= 0 or memory <= 0:
        raise RuntimeError("Detected nonpositive hardware capacity")
    return {"cpu_threads": cpus, "ram_bytes": memory, "gpus": gpu_inventory()}


def require_supported_gpu_count(gpus: list[dict[str, Any]]) -> None:
    if len(gpus) > MAX_WORKERS_PER_NODE:
        raise ValueError("At most 16 GPU workers per Job are supported")


def worker_profiles(inventory: dict[str, Any], flavor: str, threads: int,
                    memory_limit: str = "auto") -> list[dict[str, Any]]:
    from dask.utils import parse_bytes

    WorkerConfig(threads=threads, memory_limit=memory_limit)
    gpus = inventory["gpus"]
    require_supported_gpu_count(gpus)
    count = max(1, len(gpus))
    ram = int(inventory["ram_bytes"] * 0.8 / count)
    if memory_limit != "auto":
        requested = int(parse_bytes(memory_limit))
        ram = min(ram, requested)
    cpu = min(float(threads), inventory["cpu_threads"] / count)
    profiles = []
    for index in range(count):
        gpu = gpus[index] if gpus else None
        resources = {"CPU_THREADS": cpu, "RAM_GIB": ram / GIB,
                     "GPU": float(gpu is not None),
                     "GPU_VRAM_GIB": gpu["vram_bytes"] * 0.9 / GIB if gpu else 0.0}
        tags = [f"FLAVOR_{flavor}"]
        if gpu:
            normalized = re.sub(r"[^A-Z0-9]+", "_", gpu["name"].upper()).strip("_")
            family = re.search(r"\b(A100|H100|H200|B100|B200|V100|T4|L4|L40S?|A10)\b",
                               gpu["name"].upper())
            model = family.group(1) if family else normalized.removeprefix("NVIDIA_")
            tags += ["HAS_GPU", "GPU_VENDOR_NVIDIA", f"GPU_MODEL_{model}"]
        profiles.append({"resources": resources, "tags": tags, "flavor": flavor,
                         "gpu": gpu, "node_inventory": inventory,
                         "memory_limit": ram, "nthreads": max(1, math.ceil(cpu))})
    return profiles


def service_owners(nodes: int) -> list[int]:
    # Preserve primary remote worker ports, with reserved slots for additional GPUs.
    return list(range(nodes)) + [node for node in range(nodes)
                                for _ in range(MAX_WORKERS_PER_NODE * 2)]


def worker_service(nodes: int, node: int, ordinal: int) -> int:
    ServiceConfig(nodes=nodes, node=node, ordinal=ordinal)
    return (node if node and ordinal == 0 else
            nodes + node * MAX_WORKERS_PER_NODE * 2 + ordinal * 2)


def nanny_service(nodes: int, node: int, ordinal: int) -> int:
    ServiceConfig(nodes=nodes, node=node, ordinal=ordinal)
    return nodes + node * MAX_WORKERS_PER_NODE * 2 + ordinal * 2 + 1


def worker_metadata(worker: Any, *, profile: dict[str, Any]) -> dict[str, Any]:
    result = dict(profile)
    result["visible_cuda"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    result["nthreads"] = worker.state.nthreads
    result["memory_limit"] = worker.memory_manager.memory_limit
    return result
