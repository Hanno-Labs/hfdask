"""A small CPU workload with an explicit output path."""

import hashlib
import json
import os
import random
import socket
from pathlib import Path

from distributed import get_worker


def square(value: int) -> int:
    return value * value


def run(client, count: int = 100, output: str = "/output/result.json") -> None:
    futures = client.map(square, range(count))
    values = client.gather(futures)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"count": len(values), "sum": sum(values)}) + "\n")


def make_payload(size: int) -> tuple[bytes, str, str]:
    return random.Random(23).randbytes(size), get_worker().address, socket.gethostname()


def consume_payload(payload: tuple[bytes, str, str]) -> dict:
    data, producer, producer_host = payload
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "producer": producer, "consumer": get_worker().address,
            "visible_cuda": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "producer_host": producer_host, "consumer_host": socket.gethostname()}


def proof(client, output: str = "/output/result.json", size: int = 4 * 1024 * 1024,
          hardware: bool = False) -> None:
    """Force a real dependency transfer; the client never gathers the payload."""
    workers = sorted(client.scheduler_info()["workers"])
    if len(workers) != 2:
        raise RuntimeError("This proof requires exactly two workers")
    if hardware:
        from hfdask.routing import workers_with
        workers = [workers_with(client, tags={"FLAVOR_cpu-performance"})[0],
                   workers_with(client, tags={"HAS_GPU"})[0]]
    source = client.submit(make_payload, size, workers=[workers[0]],
                           allow_other_workers=False, pure=False)
    result = client.submit(consume_payload, source, workers=[workers[1]],
                           resources={"GPU": 1} if hardware else None,
                           allow_other_workers=False, pure=False).result(timeout=120)
    expected = hashlib.sha256(random.Random(23).randbytes(size)).hexdigest()
    if (result["sha256"] != expected or result["bytes"] != size
            or result["producer"] != workers[0] or result["consumer"] != workers[1]
            or result["producer_host"] == result["consumer_host"]):
        raise AssertionError(result)
    result["status"] = "passed"
    if hardware:
        inventory = {address: worker["hfdask"] for address, worker
                     in client.scheduler_info()["workers"].items()}
        gpu = inventory[workers[1]]["gpu"]
        if result["visible_cuda"] != gpu["uuid"] or inventory[workers[0]]["gpu"] is not None:
            raise AssertionError("GPU process isolation or category placement failed")
        result["hardware"] = inventory
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, sort_keys=True) + "\n")
    print(json.dumps({"phase": "proof_passed", **result}), flush=True)
