# hfdask

Run ordinary Dask scripts on Hugging Face Jobs with a YAML cluster definition.
Clusters support CPU and GPU workers, mounted storage, automatic hardware
resources, and authenticated encrypted connections.

## Run a script with the CLI

Authenticate the submitting machine with `hf auth login`. From this repository's
root, configure the namespace and bucket mounts in
[`examples/inference.yaml`](examples/inference.yaml), stage the model and prompts
as described in [examples](examples/README.md), then run:

```sh
uv lock
uv run --extra p2p hfdask run --cluster examples/inference.yaml examples/job.py
```

This launches a `cpu-basic` coordinator and four `h200` worker Jobs. Submission
reserves paid HF Jobs; verify hardware access and budget first. The coordinator
runs the scheduler and ordinary Python script with the cluster's Dask client
made current, so `.compute()` uses the remote workers. There is no Dask worker
on the coordinator: only the GPUs execute Dask tasks in this example.
No `run(client, ...)` wrapper, inline script dependencies, or custom Dockerfile
is needed.

## Dependencies and bootstrap

Dependencies come from `pyproject.toml` and `uv.lock`. PyYAML is a base dependency;
the `inference` extra adds `dask[dataframe]` and vLLM (Linux x86_64 only). Keep the
local command on `--extra p2p`, **not** `--extra inference`: the latter can install
the large GPU stack locally on Linux x86_64. YAML's `environment.extras:
[inference]` selects the remote environment; hfdask adds its own `p2p` extra there.
Each Job runs `uv sync --locked --no-dev` with those extras before starting.

The YAML explicitly uses the stock Python + uv image
`ghcr.io/astral-sh/uv:python3.12-bookworm-slim`. CUDA user-space libraries come
from the installed wheels; compatible NVIDIA drivers must come from the GPU
host. This is offline Python inference, not a vLLM serving container. Pin a tested
image digest for production; the four-H200 inference path has **not** been GPU-validated.

External projects must declare `hfdask[p2p]` in their dependencies (or a selected
project extra) and regenerate `uv.lock`. Their workload dependencies and any
YAML-selected extras must also be declared in that project.

## Cluster configuration

[`examples/inference.yaml`](examples/inference.yaml) defines the run:

| Field | Purpose |
|---|---|
| `namespace` | HF namespace that owns the Jobs |
| `coordinator.flavor` | Hardware for the scheduler and script |
| `workers.flavor`, `workers.count` | Remote worker hardware and number of machines |
| `environment.image`, `environment.extras` | Bootstrap image and locked project extras |
| `mounts` | Storage sources, mount targets, and read-only settings |
| `timeout` | Remote Job timeout |
| `network.public_relays` | Explicit opt-in to public discovery and relay fallback |

Mount models and datasets read-only and outputs on a writable bucket. Use a
fresh output prefix for every run, verify bucket ownership/privacy, and pin
model/data revisions and image digests for reproducibility.

Worker machines detect hardware at startup. CPU machines run one worker;
NVIDIA machines run one worker per visible GPU, each with its own
`CUDA_VISIBLE_DEVICES` assignment. NVIDIA detection requires `nvidia-smi`;
MIG partitions are unsupported. Clusters support up to 64 Jobs and 16 GPU
workers per machine.

## Source shipping

Run from a Git project containing `pyproject.toml`, `uv.lock`, and the
project-relative script. The CLI ships current working-tree contents, including
eligible untracked files, rather than just committed `HEAD`. Ignored files and
recognized secret paths are excluded; symlinks are rejected. Review what is
included: filename filtering cannot detect every secret. Keep data and model
weights in mounts. Source limits are **512 KiB compressed, 8 MiB uncompressed,
and 2,000 files**, including the lockfile.

## Cleanup and recovery

The CLI generates node identities and records public recovery handles in
`.hfdask/run-*.json` (override with `--manifest PATH`, choosing an unused path).
It waits for completion and verifies Job cleanup; failures and interrupts
attempt to close known Jobs. If cleanup is unverified or the local process is
lost, retain the manifest and inspect every recorded Job, including partial
launches.

With HF credentials available, use the recorded manifest to release known Jobs
(replace the path below with the manifest printed by your run):

```python
import json
from pathlib import Path
from hfdask.cluster import Cluster

Cluster.from_manifest(json.loads(Path(".hfdask/run-<id>.json").read_text())).close()
```

`Cluster.close()` cancels Jobs and verifies termination. Calling Dask's shutdown
does not itself verify release of all HF Jobs. Reconcile ambiguous submissions
by cluster labels before retrying, and do not delete the manifest until cleanup
is verified. The manifest contains no private node keys and is not a task-resume
checkpoint or a persistent-client credential.

## Execution limits

The [CPU coordinator + four H200 example](examples/README.md) pairs
[`examples/job.py`](examples/job.py) with [`examples/inference.yaml`](examples/inference.yaml).
It reads Parquet prompts, applies `map_partitions(infer)`, and lazily caches one
vLLM model per GPU worker. The vLLM import is module-level; model construction,
not the import, is deferred to worker execution.

`predictions.compute()` gathers the **entire result into CPU coordinator RAM**,
then writes `/output/results.parquet` once at the end. There are no per-shard
durable checkpoints, output manifests, or resume/skip logic. The final write is
not an atomic recovery protocol. Retrieve and validate outputs after success.

Job-local disks and scheduler state are ephemeral; automatic whole-cluster
resume is not implemented. Long workloads should write independently
recoverable shards if they need durable progress.

## Security

Iroh connects Jobs over authenticated encrypted QUIC. Dask services bind to
loopback and nodes accept peers from their fixed identity roster. The CLI
currently requires explicit `network.public_relays: true`, enabling n0 discovery
and public relay fallback. Discovery and relay services can observe connection
metadata.

Run only trusted workload code and images: Dask tasks execute arbitrary Python
and can access mounted data. HF credentials stay in the submitting process;
each node receives its own Iroh key through Job secrets. Keep credentials out
of the source bundle and grant mounts only the access the workload needs.

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

## License

See [LICENSE](LICENSE).
