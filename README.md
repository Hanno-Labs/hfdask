# hfdask

Run ordinary Dask programs across Hugging Face Jobs from one YAML cluster definition.
hfdask ships the current Git working tree, starts a coordinator Job and worker Jobs,
connects them through an authenticated encrypted mesh, and cleans up the paid Jobs
when the program finishes.

> **Status:** pre-release (`0.1.0`). Development and examples currently run from
> this checkout. Public repository visibility, release tags, and package publication
> are intentionally deferred until the implementation and documentation are ready.

## Why hfdask

Hugging Face Jobs already provides CPU and GPU machines plus native model, dataset,
Space, and bucket mounts. Dask already provides distributed DataFrames, arrays, and
task graphs. hfdask supplies the missing cluster lifecycle between them:

- ordinary Dask scripts, without a framework-specific `run(client, ...)` wrapper;
- heterogeneous CPU and GPU worker machines;
- one single-threaded worker process per complete CPU core;
- locked project dependencies and Git-aware source shipping;
- authenticated, encrypted connections without a public Dask scheduler;
- recovery manifests and verified Job cleanup.

```text
submitting machine
    hfdask CLI
        │  source archive + Job definitions
        ▼
coordinator HF Job                 worker HF Job(s)
script + Dask scheduler  ◀─Iroh─▶  one Dask worker per CPU core
optional CPU workers               optional exclusive GPU assignments
```

## CPU-only quickstart

[`examples/cpu.py`](examples/cpu.py) is a normal, unannotated Dask DataFrame
program. It creates four partitions per live worker, so the Dask scheduler can use
the full CPU pool. The script imports Dask and pandas, not hfdask.

1. Authenticate the submitting machine with `hf auth login`.
2. Set your HF namespace in [`examples/cpu.yaml`](examples/cpu.yaml).
3. Review the two `cpu-basic` Jobs and the 15-minute timeout, then run:

```sh
uv sync --locked --no-dev
uv run --no-sync hfdask run --cluster examples/cpu.yaml examples/cpu.py
```

The coordinator hosts the script and scheduler. Because `coordinator.worker: true`,
it reserves one CPU core for the scheduler and starts workers on its remaining
cores. The separate worker Job uses every complete CPU core. Neither the YAML nor
the DataFrame graph contains Hugging Face-specific task annotations.

For heterogeneous CPU → GPU → CPU execution with mounted Hub data and worker-local
vLLM engines, see the [AG News inference example](examples/README.md#gpu-inference).

## Cluster configuration

The CLI reads a strict YAML definition before staging source or submitting Jobs:

| Field | Purpose |
|---|---|
| `namespace` | HF user or organization that owns the paid Jobs |
| `coordinator.flavor` | Hardware for the scheduler and script |
| `coordinator.worker` | Run workers beside the scheduler, reserving one core for it |
| `workers.flavor`, `workers.count` | Remote worker hardware and machine count |
| `environment.image` | Bootstrap image containing Python and uv |
| `environment.extras` | Locked project extras installed in every Job |
| `mounts` | Hub repositories or buckets mounted into every Job |
| `timeout` | HF Job lifetime such as `15m` or `2h` |
| `network.public_relays` | Explicit consent to public discovery and relay fallback |

Unknown keys and coerced scalar types are rejected. Counts must be YAML integers,
booleans must be YAML booleans, and `workers.count` must be between 1 and 63.
Clusters support at most 64 Jobs including the coordinator.

Mount sources use HF CLI paths:

- `hf://models/namespace/repo`
- `hf://datasets/namespace/repo`
- `hf://spaces/namespace/repo`
- `hf://buckets/namespace/bucket`

Repository mounts are read-only and may pin a branch, tag, or commit with
`revision`. Bucket mounts do not accept revisions; set `read_only: false` explicitly
for outputs. Keep permissions narrow and use a fresh output prefix for each run.

## Workers and task placement

Each machine detects its CPU affinity and cgroup quota at startup. Worker-only
machines start one process for every complete available core. A coordinator with
`worker: true` reserves exactly one core for the scheduler and uses the rest; a
one-core coordinator therefore contributes no worker.

Every worker has one Dask thread. Visible NVIDIA GPUs are assigned exclusively to
the first worker processes, one GPU per process, while remaining processes are
CPU-only. GPU discovery requires `nvidia-smi`; MIG partitions are not supported.
Detected worker counts are exchanged before the mesh allocates Dask service ports.

Unannotated Dask tasks are eligible for every worker. Use ordinary Dask resources
for numeric reservations such as `resources={"GPU": 1}`. For categorical placement,
`hfdask.routing.workers_with` and `submit_on` select workers by detected hardware or
custom group tags without consuming artificial resource slots.

## Dependencies and bootstrap

The base package includes distributed Dask, the HF client, PyYAML, Pydantic, and
Iroh. The `dataframe` extra adds Dask DataFrame dependencies. The `inference` extra
adds Dask DataFrame plus vLLM on Linux x86-64.

The CLI ships `pyproject.toml` and `uv.lock`; every Job runs
`uv sync --locked --no-dev` with the YAML-selected extras. The submitting machine
does not need remote workload dependencies such as pandas, PyTorch, or vLLM.

For now, the examples run from this repository itself. An external project must
declare a resolvable `hfdask` dependency, declare all workload dependencies or
selected extras, and regenerate its lockfile. The public installation command will
be documented when distribution is enabled.

## Source shipping

Run the CLI from a Git project containing `pyproject.toml`, `uv.lock`, and the
project-relative script. hfdask snapshots the current working tree, including
eligible untracked files—not just committed `HEAD`.

Ignored files and recognized secret paths are excluded, and symlinks are rejected.
Filename filtering cannot identify every secret, so review the working tree before
submission. Keep data and model weights in mounts. The source limits are 8 MiB
compressed, 8 MiB uncompressed, and 2,000 files.

The launcher creates or reuses the namespace's private `jobs-artifacts` bucket,
uploads the archive under a unique prefix, mounts it read-only, and verifies its
checksum before extraction. The CLI prints the artifact URI. Artifacts remain after
the run and can incur storage charges until removed.

## Cleanup and recovery

The CLI writes public recovery handles to `.hfdask/run-*.json`, or to an unused path
selected with `--manifest`. It saves the growing manifest after every acknowledged
submission, waits for the driver, and verifies termination of known Jobs. Failures
and interrupts attempt cleanup without hiding an unverified result.

If the submitting process is lost or cleanup cannot be verified, keep the manifest
and inspect every recorded Job. With HF credentials available:

```python
import json
from pathlib import Path

from hfdask.cluster import Cluster

manifest = json.loads(Path(".hfdask/run-<id>.json").read_text())
Cluster.from_manifest(manifest).close()
```

`Cluster.close()` cancels known Jobs and polls them to a terminal state. The
manifest contains no private node keys, is not a workload checkpoint, and does not
resolve an ambiguous submission that returned no handle; reconcile those Jobs by
their cluster labels before retrying.

## Security and limitations

Iroh provides authenticated encrypted QUIC, NAT traversal, and relay transport.
Dask services bind only to loopback, and each node admits identities from its fixed
cluster roster. `network.public_relays: true` is required explicitly; discovery and
relay services can observe connection metadata.

Run only trusted workloads and images. Dask tasks execute arbitrary Python and can
access their mounted data. HF credentials remain on the submitting machine; Jobs
receive distinct Iroh keys through Job secrets.

Job-local disks and scheduler state are ephemeral. Automatic whole-cluster resume
is not implemented. Long workloads should write independently recoverable shards
to durable storage. The example output writes are not an atomic checkpoint protocol.

## API documentation

Generate the pdoc reference locally:

```sh
uv sync --locked --extra docs
mise run docs
open docs/hfdask.html
```

The generated `docs/` directory is ignored build output. The tracked
[`pdoc-templates`](pdoc-templates/) directory is source configuration. The reference
documents the public package and implementation modules but intentionally excludes
the standalone pre-installation `hfdask.bootstrap` script.

Start with `hfdask.jobs` for a single Job, `hfdask.cluster` for multi-Job lifecycle,
`hfdask.client` for persistent connections, and `hfdask.routing` for placement.

## Development

```sh
uv sync --locked --group dev
mise run check-format
mise run lint
mise run test
mise run docs
mise run build
```

CI runs formatting, Ruff, ty, the test suite, pdoc generation, and distribution
builds on pull requests and pushes to `main`. The opt-in encrypted transport test is:

```sh
HFDASK_TEST_KEYS=1 uv run pytest tests/test_iroh_integration.py
```

## License

[Apache-2.0](LICENSE).
