# Offline vLLM: CPU coordinator + four H200 workers

The main example is [`job.py`](job.py), an ordinary Dask script, with cluster
configuration in [`inference.yaml`](inference.yaml). No custom Dockerfile, image
build, or inline dependency block is needed. One `cpu-basic` Job runs the
scheduler and script; four `h200` Jobs run
the Dask workers. **Five Jobs, four Dask workers, four independent GPUs.** The
CPU coordinator is not a Dask worker. This is data-parallel offline inference,
not a model spread across four GPUs and not a vLLM server.

## Configure and run

1. Authenticate locally with `hf auth login` and set `namespace` in the YAML.
   Submission reserves paid HF Jobs; verify hardware access and budget first.
2. Stage a complete, pinned model snapshot, including tokenizer, under the input
   bucket's `model/`, and Parquet files under `prompts/`. Each row needs string
   `id` and `prompt` fields; use globally unique IDs and raw completion prompts
   suitable for the model. No chat template is applied. Choose a model that fits
   one H200 and keep prompts plus the requested 128 output tokens within 4096
   tokens. Size Parquet partitions for sensible inference batches.
3. Set the read-only input source to `hf://buckets/your-namespace/input`, mounted
   at `/input`. Set a separate writable output source to a **fresh run prefix**,
   for example `hf://buckets/your-namespace/output/run-001`, mounted at `/output`.
   The checked-in YAML uses the output bucket root; change it before each run.
   Create the buckets and verify ownership/privacy before submission.
4. Review `coordinator.flavor: cpu-basic`, `workers.flavor: h200`,
   `workers.count: 4`, and `timeout: 2h`. The count is remote machines. The CLI
   currently requires explicit `network.public_relays: true`, permitting public
   n0 discovery/relay fallback and its connection-metadata exposure.
5. From the repository root:

   ```sh
   uv lock
   uv run --extra p2p hfdask run --cluster examples/inference.yaml examples/job.py
   ```

The CLI generates distinct node identities, ships the project, waits for the
workers, and executes the script on the CPU coordinator with the cluster's Dask
client made current. Native Dask operations such as `.compute()` use that client;
no `run(client, ...)` entrypoint or manual cluster connection is needed.

### Locked dependencies and image

`pyproject.toml` declares PyYAML as a base dependency. Its `inference` extra
contains `dask[dataframe]` and vLLM, with vLLM restricted to Linux x86_64.
`uv lock` resolves the project without installing the inference stack locally.
Use **`--extra p2p` locally**, not `--extra inference`, which can install the
large GPU stack on a Linux x86_64 submitter. YAML's `environment.extras:
[inference]` selects dependencies for remote Jobs; the CLI also adds `p2p` when
shipping this hfdask project. Every Job syncs the shipped lock with
`uv sync --locked --no-dev` and the selected extras.

`environment.image` explicitly names the stock Python + uv image
`ghcr.io/astral-sh/uv:python3.12-bookworm-slim`. There is no default vLLM image,
CUDA image build, or serving process. Installed wheels supply CUDA user-space
libraries; the GPU host must supply compatible NVIDIA drivers and `nvidia-smi`
for hardware discovery. All nodes use the same locked environment, including
the CPU coordinator. Pin a tested image digest for production. The CLI configures
non-daemon Dask workers and spawn multiprocessing for Dask and vLLM.

External projects must declare `hfdask[p2p]` in their project dependencies (or a
selected extra), declare their workload dependencies and YAML-selected extras,
and regenerate `uv.lock`.

### What is shipped

The script, `pyproject.toml`, and `uv.lock` must be present and not excluded.
Git selects tracked and eligible untracked files, but the archive uses their
**current working-tree contents**, not committed `HEAD`. Ignored files and
recognized secret paths (including `.env*`, private-key files, and `.hfdask`)
are excluded; symlinks are rejected. Review project contents and ignore rules:
secret-path filtering is not a general secret scanner.

Limits are **512 KiB compressed, 8 MiB uncompressed, and 2,000 files**, including
`uv.lock`. Keep models, datasets, generated outputs, caches, and credentials out
of the source bundle; use bucket mounts for data.

## Execution and recovery limits

`job.py` reads `/input/prompts/*.parquet` and applies
`dataset.map_partitions(infer, meta=...)`. Only GPU machines host Dask workers,
so graph tasks, including Parquet reads, run there without explicit H200 routing
or `GPU=1` task annotations. `infer` imports vLLM at module level but constructs
`LLM` lazily on the first nonempty partition, caching it on the worker. Each
replica uses `tensor_parallel_size=1`; models are not serialized into tasks.
Worker restart requires a model reload.

`predictions.compute()` gathers **all predictions into CPU coordinator RAM**.
Only after the entire computation succeeds does the coordinator write
`/output/results.parquet`. Ensure the full result fits coordinator memory; this
is not a streaming output pipeline. Always use a fresh output prefix.

This script has no per-shard durable writes, readback hashes, output manifests,
resume, or skip-existing behavior.
A failure before the final write leaves no completed prediction checkpoint; the
final write is not an atomic recovery protocol. Retrieve and validate the output
after success. Job-local disks and scheduler state are ephemeral; automatic
whole-cluster resume is not implemented.

### Job cleanup

The CLI prints and updates a **public recovery manifest** at
`.hfdask/run-*.json`; use `--manifest PATH` to choose an unused path. It contains
Job handles, not private node keys or prediction checkpoints. Successful waiting
releases and verifies the remaining Jobs; launch failures and interrupts attempt
to close known Jobs. If the process is lost or cleanup is reported unverified,
retain the manifest and inspect all recorded Jobs, including partial launches.
Use `Cluster.from_manifest(...).close()` with HF credentials as shown in
[Cleanup and recovery](../README.md#cleanup-and-recovery), and reconcile
ambiguous submissions by cluster labels before retrying. Do not delete the
manifest until cleanup is verified.

An actual four-H200 vLLM run with this CLI and stock image has **not been
GPU-validated**.

References: [vLLM offline batches](https://docs.vllm.ai/en/latest/getting_started/quickstart/#offline-batched-inference)
and [multiprocessing constraints](https://docs.vllm.ai/en/latest/usage/troubleshooting/#python-multiprocessing).
