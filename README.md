# hfdask

Run ordinary Dask scripts on Hugging Face Jobs with a YAML cluster definition.
Clusters support CPU and GPU workers, mounted storage, automatic hardware
resources, and authenticated encrypted connections.

## Run a script with the CLI

Authenticate the submitting machine with `hf auth login`. From this repository's
root, configure the namespace and a fresh, seeded output bucket prefix in
[`examples/inference.yaml`](examples/inference.yaml), as described in
[examples](examples/README.md). Models and datasets are mounted directly from the Hub;
no input bucket or manual input staging is needed.
Then run:

```sh
uv sync --no-dev
uv run --no-sync hfdask run --cluster examples/inference.yaml examples/job.py
```

This launches two Jobs: a `cpu-basic` coordinator and one `l4x1` GPU worker
Job. Submission reserves paid HF Jobs; verify hardware access and budget first.
The coordinator runs the scheduler and ordinary Python script with the cluster's
Dask client made current. The example sets `coordinator.worker: true` to enable
a CPU Dask worker in that same scheduler Job, so preparation and output writing
run on CPU while inference runs on GPU. No `run(client, ...)` wrapper, inline
script dependencies, or custom Dockerfile is needed.

## Dependencies and bootstrap

Dependencies come from `pyproject.toml` and `uv.lock`. The base install includes
Dask, the HF client, PyYAML, Pydantic v2 validation, and Iroh encrypted transport. The `inference`
extra adds `dask[dataframe]` and vLLM (Linux x86_64 only). Do not pass `--extra inference`
to the local launcher: YAML's `environment.extras: [inference]` selects that
remote environment. Each Job runs `uv sync --locked --no-dev` with the selected
extras before starting; the submitting machine does not need PyTorch or vLLM.

The YAML uses the official `vllm/vllm-openai:v0.29.0` image, pinned to its Linux
amd64 digest, for CUDA tooling (including `nvcc`), Python, and uv. Despite the
image name, hfdask runs its Python bootstrap, not the OpenAI server. HF recommends
this runtime for [offline batch inference with uv](https://huggingface.co/docs/hub/jobs-popular-images#vllm).

`uv sync` still installs the project's locked dependencies into an isolated
virtual environment; it does not reuse the image's preinstalled Python packages.
The pinned image release matches vLLM 0.29.0 in `uv.lock`; keep these compatible
when upgrading. The GPU host supplies the NVIDIA driver.


External projects must declare `hfdask` in their dependencies (or a selected
project extra) and regenerate `uv.lock`. Their workload dependencies and any
YAML-selected extras must also be declared in that project.

## Cluster configuration

[`examples/inference.yaml`](examples/inference.yaml) defines the run:

| Field | Purpose |
|---|---|
| `namespace` | HF namespace that owns the Jobs |
| `coordinator.flavor` | Hardware for the scheduler and script |
| `coordinator.worker` | Enable a Dask worker in the scheduler Job (default `false`) |
| `workers.flavor`, `workers.count` | Remote worker hardware and number of machines |
| `environment.image`, `environment.extras` | Bootstrap image and locked project extras |
| `mounts` | Storage sources, mount targets, and read-only settings |
| `timeout` | Remote Job timeout |
| `network.public_relays` | Explicit opt-in to public discovery and relay fallback |

PyYAML safely loads the document; strict Pydantic schemas validate it before
source staging or Job submission. Unknown keys are rejected at every YAML level.
Counts must be integers (not booleans, quoted numbers, or floats); boolean fields
must be YAML booleans, not quoted strings. `workers.count` defaults to 1 and must
be 1–63; the optional coordinator worker is additional. `timeout` defaults to
`1h` and accepts positive integer durations in `s`, `m`, `h`, or `d` (for example,
`30m`). Public relay consent is always required, never inferred.

Python configuration APIs use the same strict validation conventions.
`JobSpec` and `WorkerGroup` remain frozen dataclasses with positional constructors
and `dataclasses.replace` support. Schema validation failures raise
`pydantic.ValidationError`, a `ValueError` subclass with structured field paths;
this replaces the previous `TypeError` for some invalid configuration types.
Validation does not coerce strings or booleans to worker counts, and error text
is not a stable API. Workload kwargs retain standard-library JSON serialization
semantics. Launch planning remains separate from configuration validation.

Validation is organized around locally readable contracts, not a catch-all validator:
`PositiveCount`, `Duration`, `RelativeScript`, `MountSource`, `MountTarget`,
`CustomTag`, and `WorkloadKwargs` describe individual inputs. Named
`AfterValidator` functions enforce concepts such as distinct identities and safe
mount paths. `LaunchPlan.require_identity_per_job` and
`PackagePlan.require_locked_runner_dependency` express relationships between inputs;
launch plans validate the Job limit before expanding topology. Persistent workloads
have an explicit empty-entrypoint/empty-kwargs contract. Runner mesh metadata is
checked before endpoint startup. Local Dask accepts memory limit `"0"` to disable
its limit; hardware-budgeted workers require a positive limit or `"auto"`.

Runtime evidence is deliberately not treated as static configuration. Named
boundaries still verify Git-selected files and source budgets, private bucket
visibility, archive checksums and safe members, GPU capacity, matching live workers,
peer identity, service ownership, and descriptor capacity. The bootstrap stays
standard-library-only because it runs before the locked environment is installed.
Protocol handshakes, connection admission, lifecycle deadlines, and cleanup checks
remain next to the operations whose state they protect.

Mount sources follow HF CLI conventions: `hf://models/namespace/repo`,
`hf://datasets/namespace/repo`, `hf://spaces/namespace/repo`, or
`hf://buckets/namespace/bucket`, optionally followed by a subfolder. Repository
mounts are read-only and accept a `revision` (branch, tag, or commit; defaults
to the Hub's `main`). Bucket mounts do not accept revisions; explicitly set
`read_only: false` for outputs.

This example pins the model and dataset commits in YAML, mounting them at
`/model` and `/dataset`. Files are fetched on demand; the workload only reads
filesystem paths. Use a fresh, seeded writable output prefix for every run and
verify ownership/privacy.

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
weights out of the source bundle; use mounts. Source limits are **8 MiB compressed,
8 MiB uncompressed, and 2,000 files**, including the lockfile.

Like `hf jobs uv run`, the launcher automatically creates or reuses the namespace's
`jobs-artifacts` bucket and stages files in a unique subfolder. It verifies that
the bucket is private before uploading your project archive, mounts that subfolder
read-only, and verifies the archive checksum before extraction. No source-bucket
configuration is required.
The CLI prints the artifact URI; source artifacts are retained after the run and
incur storage charges until removed, as with HF's artifact-staging pattern.

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

The [AG News smoke example](examples/README.md) uses standard `dask.delayed`
tasks for a CPU → GPU → CPU pipeline. CPU worker addresses are selected from
scheduler information where the `GPU` resource is absent or zero. Preparation,
concatenation, and saving are restricted to those addresses; inference tasks
request `resources={"GPU": 1}`. `summary.compute(optimize_graph=False)` preserves
the separately annotated stage boundaries.

On a CPU worker, `prepare` reads the mounted test Parquet, samples 128 rows
(32 per category, seed 23), and splits them into eight 16-row partitions.
`WorkerSetup` skips CPU workers and loads the Qwen3-0.6B engine from `/model`
once per GPU worker, including workers that join or restart later. The same
graph supports multiple GPU workers, with one worker per visible GPU.
Both Hub revisions are pinned in [`examples/inference.yaml`](examples/inference.yaml),
not in workload code.

CPU tasks concatenate predictions and write `/output/results.parquet`
(`id`, `text`, `label`, `prediction`) and `/output/summary.json` with row count,
invalid-prediction count, and accuracy. Only the summary is collected by the
script submitting the graph. The `inference` event topic includes `prepared`
and `results_saved` alongside model-loading and partition-completion events.
Metrics are descriptive, not pass/fail thresholds. The YAML and HF Job volume
configuration record the input repository revisions. This is not streaming
output or an atomic recovery protocol.
There are no per-shard checkpoints or resume/skip-existing semantics.

The [AG News card](https://huggingface.co/datasets/fancyzhx/ag_news) lists its
license as unknown and describes research purposes; public access is not a
blanket reuse license. [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) uses
Apache-2.0. Review terms before reusing or redistributing inputs or outputs.

Job-local disks and scheduler state are ephemeral; automatic whole-cluster
resume is not implemented. Long workloads should write independently
recoverable shards if they need durable progress.

## Security

Iroh connects Jobs over authenticated encrypted QUIC. Dask services bind to
loopback and nodes accept peers from their fixed identity roster. The CLI
requires explicit `network.public_relays: true`, enabling n0 discovery
and public relay fallback. Discovery and relay services can observe connection
metadata.

Run only trusted workload code and images: Dask tasks execute arbitrary Python
and can access mounted data. HF credentials stay in the submitting process;
each node receives its own Iroh key through Job secrets. Keep credentials out
of the source bundle and grant mounts only the access the workload needs.

## Development

```sh
uv sync --group dev
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
