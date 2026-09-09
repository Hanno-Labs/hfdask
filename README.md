# hfdask

A Python library for running Dask workloads on Hugging Face Jobs.

Define a fixed cluster by hardware flavor and machine count, then run a Python
function with a Dask `Client`. Clusters support CPU and GPU workers, mounted
storage, automatic hardware resources, and authenticated encrypted connections.

## Install

```sh
pip install -e '.[p2p]'
```

Install hfdask and your workload module in the container image used by the Jobs.
Authenticate the submitting machine with `hf auth login`.

## Write a workload

```python
# workload.py
import json
from pathlib import Path

def square(value):
    return value * value

def run(client, output="/output/results.json"):
    results = client.gather(client.map(square, range(100)))
    Path(output).write_text(json.dumps(results))
```

The entrypoint receives a Dask client followed by JSON-serializable keyword
arguments from `JobSpec.kwargs`. Wait for tasks and save results before returning.

## Create a cluster

```python
import json
import iroh
from pathlib import Path
from huggingface_hub import Volume
from hfdask import JobSpec, WorkerGroup
from hfdask.cluster import Identity, submit_cluster

identities = [Identity(iroh.SecretKey.generate().to_bytes()) for _ in range(3)]
spec = JobSpec(
    namespace="your-namespace",
    image="your-registry/workload@sha256:...",
    entrypoint="workload:run",
    flavor="cpu-basic",
    threads_per_worker=2,
    timeout="1h",
    volumes=[Volume(type="bucket", source="your-namespace/results",
                    mount_path="/output", read_only=False)],
)
cluster = submit_cluster(
    spec, identities,
    scheduler_worker=True,
    worker_groups=[WorkerGroup("cpu-performance", count=2)],
    public_relays=True,
    on_submitted=lambda c: Path("cluster.json").write_text(json.dumps(c.manifest())),
)
try:
    cluster.wait(timeout=3600)
finally:
    cluster.close()
```

This starts three Jobs: a scheduler with a colocated CPU worker, plus two remote
CPU worker machines. Supply one distinct identity per Job and keep keys secret.

`WorkerGroup.count` counts remote machines. With groups, `scheduler_worker=True`
adds a worker on the scheduler machine and `JobSpec.workers` is superseded.
Set `scheduler_flavor` to override scheduler hardware. Without groups,
`JobSpec.workers` sets the total worker count and `worker_flavor` selects remote
hardware. Both overrides default to `JobSpec.flavor`.

## Offline vLLM inference

The [CPU coordinator + four H200 example](examples/README.md) includes a
workload, Dockerfile, mounts, and submission code. Each H200 keeps one model
replica loaded. Dask distributes batches and the CPU coordinator saves output.

Its topology is:

```python
cluster = submit_cluster(
    spec, five_identities,
    scheduler_flavor="cpu-basic",
    scheduler_worker=True,
    worker_groups=[WorkerGroup("h200", count=4, tags=("inference",))],
    public_relays=True,
)
```

## Resources and routing

Multi-job clusters detect hardware at startup. CPU machines run one worker;
NVIDIA machines run one worker per visible GPU, each with its own
`CUDA_VISIBLE_DEVICES` assignment.

| Resource | Per-worker capacity |
|---|---|
| `CPU_THREADS` | CPU quota/affinity share, capped by configured threads |
| `RAM_GIB` | 80% of node RAM divided among workers, capped by the memory limit |
| `GPU` | 1 for a GPU worker, 0 for a CPU worker |
| `GPU_VRAM_GIB` | 90% of the assigned GPU's total VRAM |

These are scheduling reservation budgets. Tasks request the capacity they need:

```python
from hfdask.routing import submit_on

future = submit_on(
    client, predict, batch,
    tags={"GPU_MODEL_H200", "inference"},
    resources={"GPU": 1, "RAM_GIB": 4},
)
result = future.result()
```

Tags include `HAS_GPU`, `GPU_VENDOR_NVIDIA`, model-family tags such as
`GPU_MODEL_H200`, flavor tags such as `FLAVOR_cpu-basic`, and custom group tags.
`submit_on` enforces placement and raises when no registered worker matches.
`workers_with(client, tags={...})` returns addresses for native Dask APIs.

NVIDIA detection requires `nvidia-smi`; MIG partitions are unsupported.
Clusters support up to 64 Jobs and 16 GPU workers per machine.

## Storage and lifecycle

Pass Hugging Face `Volume` objects through `JobSpec.volumes` to mount models,
datasets, or buckets on every Job. Mount inputs read-only and outputs on a
writable bucket. Pin model/data revisions and image digests for reproducibility.

Persist `cluster.manifest()` during submission. A partial launch raises
`LaunchError`, whose `cluster` contains known Job handles. Reconcile ambiguous
submissions by cluster labels before retrying. `cluster.close()` cancels Jobs
and verifies termination. A local wait timeout leaves Jobs running until
explicit cleanup or their remote timeout.

Write long workloads as independently recoverable shards. Job-local disks and
scheduler state are ephemeral; automatic whole-cluster resume is not implemented.
Retrieve outputs and verify their manifests after completion.

## Transport

Iroh connects Jobs over authenticated encrypted QUIC. Dask services bind to
loopback and nodes accept peers from their fixed identity roster.
`public_relays=True` enables n0 discovery and public relay fallback.
Alternatively, supply `relay_urls=["https://your-relay.example"]`; discovery
still uses n0. Discovery and relay services can observe connection metadata.

Run trusted workload code and images. HF credentials stay in the submitting
process; each node receives its own Iroh key through Job secrets.

## Single-machine workloads

`submit(JobSpec(...))` creates one HF Job containing a scheduler and
`JobSpec.workers` worker processes. Its entrypoint uses the same workload contract.

## Development

```sh
uv sync --extra p2p --group dev
uv run pytest
uv run ruff check .
uv run mypy src
```

Run the opt-in encrypted transport tests with:

```sh
HFDASK_TEST_KEYS=1 uv run pytest tests/test_iroh_integration.py
```

See [examples](examples/README.md) for cross-Job verification details.

## License

See [LICENSE](LICENSE).
