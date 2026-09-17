# hfdask

Run ordinary Dask programs across Hugging Face Jobs from one YAML cluster definition.
hfdask ships the current Git working tree, starts a coordinator Job and worker Jobs,
places them in one Hugging Face network group, authenticates every Dask connection
with mTLS, and cleans up the paid Jobs when the program finishes.

> **Status:** pre-release (`0.2.0`). APIs and configuration may change before the
> first stable release.

## Why hfdask

We like working in the Hugging Face ecosystem and find ourselves using it for more
and more of our work. [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)
runs scripts and containers on managed, on-demand CPUs and GPUs alongside native
access to models, datasets, Spaces, and buckets.

Much of that work combines CPU-heavy data transformation with GPU-heavy inference.
We like expressing these pipelines in Dask: ordinary Python task graphs make it easy
to parallelize transformations, preserve dependencies between stages, and schedule
each task on suitable hardware. What was missing was a way to run that same Dask
programming model across Hugging Face Jobs.

hfdask connects the two. It turns a YAML cluster definition into a temporary Dask
cluster on Hugging Face Jobs, runs an ordinary Dask script, and cleans up the Jobs
when the program finishes. This lets us:

- use CPU and GPU Jobs together in one Dask task graph;
- keep data transformation, inference, and aggregation in one program;
- use native Hugging Face model, dataset, Space, and bucket mounts;
- ship a locked project environment and the current Git working tree;
- use HF network groups for private discovery and reachability while Dask mTLS
  authenticates and encrypts scheduler, worker, and client traffic;
- recover and clean up paid Jobs when a run fails or is interrupted.

```text
submitting machine
    hfdask CLI
        │  source archive + Job definitions
        ▼
coordinator HF Job                 worker HF Job(s)
script + Dask scheduler  ◀─mTLS─▶  one Dask worker per CPU core
optional CPU workers               optional exclusive GPU assignments
```

## CPU-only quickstart

Create a Git-backed uv project. Add the libraries used by the Dask program as
normal project dependencies, then add hfdask to the deployment dependency group
selected by the cluster YAML.

```sh
mkdir hfdask-quickstart
cd hfdask-quickstart
git init
uv init --bare --python 3.12
uv python pin 3.12
uv add "dask[dataframe,distributed]>=2025.1,<2027" pandas
uv add --group deploy hfdask
curl -fL https://raw.githubusercontent.com/Hanno-Labs/hfdask/v0.2.0/examples/cpu.py -o cpu.py
curl -fL https://raw.githubusercontent.com/Hanno-Labs/hfdask/v0.2.0/examples/cpu.yaml -o cluster.yaml
```

[`examples/cpu.py`](https://github.com/Hanno-Labs/hfdask/blob/v0.2.0/examples/cpu.py) is a normal, unannotated Dask DataFrame
program. It creates four partitions per live worker, so the Dask scheduler can use
the full CPU pool. The script imports Dask and pandas, not hfdask.

1. Authenticate the submitting machine with `uv run --group deploy hf auth login`.
2. Set your HF namespace in `cluster.yaml`.
3. Review the two `cpu-basic` Jobs and the 15-minute timeout, then run:

```sh
uv run --group deploy hfdask run --cluster cluster.yaml cpu.py
```

The coordinator hosts the script and scheduler. Because `coordinator.worker: true`,
it reserves one CPU core for the scheduler and starts workers on its remaining
cores. The separate worker Job uses every complete CPU core. Neither the YAML nor
the DataFrame graph contains Hugging Face-specific task annotations.

For heterogeneous CPU → GPU → CPU execution with mounted Hub data and worker-local
vLLM engines, see the [AG News inference example](https://github.com/Hanno-Labs/hfdask/blob/v0.2.0/examples/README.md#gpu-inference).

## Cluster configuration

The CLI reads a strict YAML definition before staging source or submitting Jobs:

| Field | Purpose |
|---|---|
| `namespace` | HF user or organization that owns the paid Jobs |
| `coordinator.flavor` | Hardware for the scheduler and script |
| `coordinator.worker` | Run workers beside the scheduler, reserving one core for it |
| `workers.flavor`, `workers.count` | Remote worker hardware and machine count |
| `environment.image` | Bootstrap image containing Python and uv |
| `environment.groups` | Locked dependency groups installed in every Job |
| `mounts` | Hub repositories or buckets mounted into every Job |
| `timeout` | HF Job lifetime such as `15m` or `2h` |

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
Each Job advertises its detected worker count through Dask metadata. The coordinator
waits for the exact per-node topology before running the workload.

Unannotated Dask tasks are eligible for every worker. Use ordinary Dask resources
for numeric reservations such as `resources={"GPU": 1}`. For categorical placement,
`hfdask.routing.workers_with` and `submit_on` select workers by detected hardware or
custom group tags without consuming artificial resource slots.

## Dependencies and bootstrap

Declare the libraries imported by your Dask program as normal project dependencies,
independent of where the program will run:

```sh
uv add "dask[dataframe,distributed]>=2025.1,<2027" pandas
```

Add hfdask to a deployment dependency group, then list that group under
`environment.groups` in the cluster YAML. Dependency groups stay local to the
project instead of becoming published package extras:

```sh
uv add --group deploy hfdask
```

```yaml
environment:
  groups: [deploy]
```

Enable the same group when submitting so the hfdask CLI is available locally:

```sh
uv run --group deploy hfdask run --cluster cluster.yaml cpu.py
```

hfdask ships the project's `pyproject.toml` and `uv.lock` to every Job and runs
`uv sync --locked --no-dev --group deploy`. That installs the base workload
together with hfdask's remote runner in one locked environment.

An isolated `uvx hfdask` invocation cannot replace the dependency group because it
does not add the remote runner to the submitted project's lockfile.

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
manifest contains no private credentials, is not a workload checkpoint, and does not
resolve an ambiguous submission that returned no handle; reconcile those Jobs by
their cluster labels before retrying.

## Persistent clusters

`hfdask.cluster.boot_cluster` creates a cluster without an embedded workload. It
enables HF SSH only on the scheduler Job and returns a separate client mTLS identity.
Save that identity outside the public recovery manifest, then reconnect through a
loopback-only SSH port forward:

```python
from pathlib import Path

from hfdask.client import connect
from hfdask.cluster import boot_cluster
from hfdask.jobs import JobSpec
from hfdask.network import TLSCredentials

cluster = boot_cluster(JobSpec("your-namespace", "your-image", workers=2))
assert cluster.client_credentials is not None
cluster.client_credentials.save(Path("client-credentials.json"))

credentials = TLSCredentials.load(Path("client-credentials.json"))
with connect(cluster.manifest(), credentials) as client:
    print(client.submit(sum, [1, 2, 3]).result())
```

The SSH connection is authorized by the public keys registered on the HF account;
the forwarded Dask connection independently requires the cluster's client
certificate. Disconnecting does not stop the Jobs. Retain the recovery manifest and
call `Cluster.close()` when the cluster is no longer needed.

## Security and limitations

HF network groups supply private routes and DNS aliases, not workload identity.
hfdask creates a fresh certificate authority per cluster, sends each Job a distinct
leaf certificate and private key through HF Job secrets, and discards the CA private
key after issuance. Dask requires CA-authenticated TLS for scheduler, worker, nanny,
embedded driver, and persistent-client connections. No Dask port is publicly exposed.

Run only trusted workloads and images. Dask tasks execute arbitrary Python and can
access their mounted data. HF credentials remain on the submitting machine; Jobs
receive only their own mTLS identity through Job secrets.

Job-local disks and scheduler state are ephemeral. Automatic whole-cluster resume
is not implemented. Long workloads should write independently recoverable shards
to durable storage. The example output writes are not an atomic checkpoint protocol.

## API documentation

From a repository checkout, generate the pdoc reference locally:

```sh
uv sync --locked --group docs
mise run docs
open docs/hfdask.html
```

The generated `docs/` directory is ignored build output. The tracked
[`pdoc-templates`](https://github.com/Hanno-Labs/hfdask/tree/v0.2.0/pdoc-templates) directory is source configuration. The reference
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
builds on pull requests and pushes to `main`. The suite includes a real local Dask
mTLS handshake and rejects clients signed by another cluster CA.

## License

[Apache-2.0](https://github.com/Hanno-Labs/hfdask/blob/v0.2.0/LICENSE).
