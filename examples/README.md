# AG News demo: CPU → GPU → CPU

[`job.py`](job.py) is an ordinary `dask.delayed` script using
[`inference.yaml`](inference.yaml). By default, two Jobs run the pipeline:
one `cpu-basic` Job hosts the scheduler, script, and CPU worker; one `l4x1` Job
hosts one GPU worker. `coordinator.worker: true` enables the CPU worker inside
the scheduler Job without adding another Job; the setting defaults to `false`.
This is offline vLLM inference, not a serving process or a model spread across
GPUs. Multiple GPU workers are supported, with one worker per visible GPU.

## Configure and run

1. Authenticate locally with `hf auth login`; set `namespace` in the YAML.
   Submission reserves paid HF Jobs: verify hardware access and budget first.
2. Create an output bucket, verify ownership/privacy, and choose a **fresh prefix
   for every run**, such as `hf://buckets/your-namespace/output/ag-news-run-001`.
   Seed that prefix by uploading a small marker file before submission, then set
   it as the YAML's writable `/output` mount. The YAML also mounts the pinned
   model at `/model` and dataset at `/dataset`, both read-only.
   **No input bucket, model staging, or dataset staging is needed**.
3. Keep `coordinator.worker: true` for the CPU stages. Review
   `workers.flavor: l4x1`, `workers.count: 1`, and `timeout: 30m`.
   Required `network.public_relays: true` permits public n0 discovery and relay
   fallback, exposing connection metadata to those services.
4. From the repository root:

   ```sh
   uv sync --no-dev
   uv run --no-sync hfdask run --cluster examples/inference.yaml examples/job.py
   ```

Iroh encrypted transport is included in the base install.
Do not install `--extra inference` locally: YAML's
`environment.extras: [inference]` selects the remote dataframe/vLLM dependencies.
Each Job syncs the shipped lock with `uv sync --locked --no-dev` and those extras,
including the CPU coordinator. The launcher needs no local PyTorch or vLLM.

The image is official `vllm/vllm-openai:v0.29.0`, pinned to its Linux amd64 digest
in YAML: no custom Dockerfile or image build. It supplies CUDA tooling, including
`nvcc`, plus Python and uv, as recommended by the
[HF Jobs image guide](https://huggingface.co/docs/hub/jobs-popular-images#vllm).
hfdask runs its bootstrap instead of the serving entrypoint; inference is offline.
The GPU host must provide compatible NVIDIA drivers and `nvidia-smi`.

The isolated uv environment still installs the locked project dependencies,
not the image's preinstalled Python packages. The image release matches vLLM
0.29.0 in `uv.lock`; keep them compatible when upgrading. The CLI configures non-daemon workers and spawn
multiprocessing, ships current working-tree source, and makes the cluster's Dask
client current for the script. Following `hf jobs uv run`'s staging pattern,
it automatically uploads project source to a unique prefix in a private
`jobs-artifacts` bucket and mounts it read-only. No manual source upload is needed.
See [source shipping](../README.md#source-shipping) for exclusions, size limits,
and artifact retention; keep weights, data, and secrets out of the bundle.

## Inputs and execution

The script selects CPU worker addresses from `client.scheduler_info()` where
worker resources have no `GPU` entry or a value of zero. It requires both CPU-only
and GPU workers and builds three stages with standard `dask.delayed` tasks:

1. **CPU preparation:** `prepare` reads
   `/dataset/data/test-00000-of-00001.parquet` from the read-only `fancyzhx/ag_news`
   mount. It samples **128 rows: 32 each of World, Sports, Business, and Sci/Tech**,
   with seed **23**, preserving source row IDs, and splits them into
   **eight 16-row partitions**.
2. **GPU inference:** each delayed `infer` task is annotated with
   `resources={"GPU": 1}` and reuses the worker-local engine.
3. **CPU output:** delayed concatenation and `save_results` tasks combine
   predictions, write outputs, and calculate metrics.

Preparation and output tasks use
`dask.annotate(workers=cpu_workers, allow_other_workers=False)` to restrict them
to CPU workers. `summary.compute(optimize_graph=False)` preserves the separately
annotated CPU/GPU boundaries; only the summary returns to the submitting script
running on the coordinator.

Before submitting the graph, the script registers `WorkerSetup`, a Dask
`WorkerPlugin` that skips CPU workers and loads `Qwen/Qwen3-0.6B` from `/model`
once per GPU worker. Registration waits for setup on existing workers; Dask
also runs setup on workers that join or restart later.

Both revisions are pinned in `inference.yaml`; `job.py` contains only filesystem
paths and no Hub download logic. Mounts fetch files on demand, rather than
preloading both repositories on every node:

| Input | Revision |
|---|---|
| `fancyzhx/ag_news` | `eb185aade064a813bc0b7f42de02595523103ca4` |
| `Qwen/Qwen3-0.6B` | `c1899de289a04d12100db370d81485cdf75e47ca` |

The model uses one GPU (`tensor_parallel_size=1`), a non-thinking chat template,
temperature 0, and at most 16 output tokens. Workers emit structured Dask events
on the `inference` topic: `model_loading`, `model_loaded`, and
`partition_complete` track inference, including loading/inference elapsed time
and partition row counts. CPU tasks emit `prepared` and `results_saved` with row
counts. Dask adds worker attribution and timestamps.
Use `get_client().get_events("inference")` while the cluster is running to inspect
them; events are bounded scheduler history, not durable HF Job stdout logs.

The [AG News card](https://huggingface.co/datasets/fancyzhx/ag_news) lists the
license as **unknown** and describes use for **research purposes**. Public access
does not establish unrestricted reuse rights; review terms before reuse or
redistribution, including output article text.
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) is **Apache-2.0**.

## Outputs

CPU worker tasks concatenate the small result in memory and write:

- `/output/results.parquet`: `id`, `text`, `label`, `prediction`.
- `/output/summary.json`: `rows`, `invalid_predictions`, and `accuracy`, also
  collected by the submitting script and printed to coordinator stdout.

Predictions outside the four category strings count as invalid; all rows contribute
to accuracy. Metrics are descriptive, not pass/fail thresholds. Repository names
and pinned revisions are recorded in YAML and the HF Job's volume configuration.

There is no resume, skip-existing behavior, or per-partition durable checkpoint.
Final writes are not an atomic recovery protocol. Job-local caches and scheduler
state are ephemeral; retries need a fresh seeded output prefix and reload inputs.
This CPU → GPU → CPU example is intentionally small, not a streaming output pipeline.

## Cost and cleanup

Check HF pricing for `l4x1` and `cpu-basic` before submitting. The configured
**30-minute timeout is not an assured spending cap**. Storage charges, retries,
and unverified cleanup can add cost.

The CLI records public Job handles in `.hfdask/run-*.json` (or an unused
`--manifest PATH`), waits for completion, and verifies cleanup. Failures and
interrupts attempt to close known Jobs. If the process is lost or cleanup is
unverified, retain the manifest, inspect every recorded Job, and follow
[cleanup and recovery](../README.md#cleanup-and-recovery). Reconcile ambiguous
submissions by cluster labels before retrying. This recovery manifest is not a
prediction checkpoint; keep it until Job termination is verified.
