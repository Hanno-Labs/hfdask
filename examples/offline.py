"""Offline vLLM replicas; Dask owns placement, vLLM owns batching within a replica."""

import hashlib
import json
from pathlib import Path
from typing import Any

from distributed import Client, as_completed, get_worker

from hfdask.routing import workers_with


def read_batch(path: str) -> list[dict[str, str]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row.get("id"), str)
                       or not isinstance(row.get("prompt"), str) for row in rows):
        raise ValueError("Each nonempty JSONL shard needs string id and prompt fields")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate IDs within shard")
    return rows


def generate(rows: list[dict[str, str]], model: str, max_tokens: int) -> list[dict[str, str]]:
    from vllm import LLM, SamplingParams

    worker = get_worker()
    engine = getattr(worker, "offline_engine", None)
    if engine is None:
        print(json.dumps({"phase": "loading_model", "worker": worker.address}), flush=True)
        engine = LLM(model=model, tensor_parallel_size=1, gpu_memory_utilization=0.85,
                     max_model_len=4096, trust_remote_code=False, seed=23)
        worker.offline_engine = engine
        worker.offline_model = model
    elif worker.offline_model != model:
        raise ValueError("A worker cannot switch models in this example")
    outputs = engine.generate([row["prompt"] for row in rows],
                              SamplingParams(temperature=0, max_tokens=max_tokens),
                              use_tqdm=False)
    if len(outputs) != len(rows):
        raise ValueError("vLLM output count does not match input")
    return [{"id": row["id"], "prompt": row["prompt"], "text": output.outputs[0].text}
            for row, output in zip(rows, outputs, strict=True)]


def save_batch(rows: list[dict[str, str]], output: str, name: str) -> dict[str, Any]:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path = directory / name
    path.write_bytes(payload)
    if path.read_bytes() != payload:
        raise OSError("Output readback mismatch")
    manifest = {"file": name, "rows": len(rows), "sha256": hashlib.sha256(payload).hexdigest()}
    # Manifest last: a missing manifest means this shard must be rerun.
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    return manifest


def run(client: Client, input: str = "/input", output: str = "/output",
        model: str = "/model", max_tokens: int = 128) -> None:
    gpu_workers = workers_with(client, tags={"GPU_MODEL_H200"})
    cpu_workers = workers_with(client, tags={"FLAVOR_cpu-basic"})
    if len(gpu_workers) != 4 or len(cpu_workers) != 1:
        raise ValueError("Expected one CPU coordinator and four H200 workers")
    shards = iter(sorted(Path(input).glob("*.jsonl")))
    pending = as_completed()
    assignments: dict[Any, tuple[str, str]] = {}
    completed = 0

    def enqueue(worker: str) -> bool:
        path = next(shards, None)
        if path is None:
            return False
        rows = client.submit(read_batch, str(path), workers=cpu_workers,
                             allow_other_workers=False, pure=False)
        future = client.submit(generate, rows, model, max_tokens, workers=[worker],
                               allow_other_workers=False, resources={"GPU": 1}, pure=False)
        assignments[future] = (worker, path.name)
        pending.add(future)
        return True

    for worker in gpu_workers:
        enqueue(worker)
    for future in pending:
        worker, name = assignments.pop(future)
        saved = client.submit(save_batch, future, output, name, workers=cpu_workers,
                              allow_other_workers=False, pure=False).result()
        completed += 1
        print(json.dumps({"phase": "batch_saved", "completed": completed, **saved}), flush=True)
        future.release()
        enqueue(worker)
    if completed == 0:
        raise ValueError("No input JSONL shards found")
    print(json.dumps({"phase": "complete", "shards": completed}), flush=True)
