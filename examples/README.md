# Examples

## CPU DataFrame

[`cpu.py`](cpu.py) is the smallest cluster example. It uses an ordinary,
unannotated Dask DataFrame graph and [`cpu.yaml`](cpu.yaml) launches two
`cpu-basic` Jobs:

- the coordinator runs the script and scheduler, reserves one core for the
  scheduler, and uses its remaining cores for workers;
- the worker-only Job starts one single-threaded Dask worker per complete core.

The example creates four partitions per live worker and computes a simple column
transformation and reduction. No task has Hugging Face or hardware annotations, so
the Dask scheduler is free to use every worker. Follow the root
[CPU-only quickstart](../README.md#cpu-only-quickstart) to run it.

## GPU inference

[`job.py`](job.py) and [`inference.yaml`](inference.yaml) demonstrate a
heterogeneous CPU → GPU → CPU graph. By default:

- one `cpu-basic` coordinator hosts the script, scheduler, and per-core CPU workers;
- one `l4x1` Job hosts one GPU-assigned worker and CPU-only workers on its remaining
  cores.

This is offline vLLM inference, not a serving process or a model spread across
GPUs. Submission reserves paid HF Jobs.

### Configure and run

1. Create a Git-backed uv project. Add the workload libraries as normal project
   dependencies, pinning vLLM to the version supplied by the example image. Add
   hfdask to the `inference` dependency group selected by the cluster YAML:

   ```sh
   mkdir hfdask-inference
   cd hfdask-inference
   git init
   uv init --bare --python 3.12
   uv python pin 3.12
   uv add "dask[dataframe,distributed]>=2025.1,<2027" pandas \
     "vllm==0.29.0; sys_platform == 'linux' and platform_machine == 'x86_64'"
   uv add --group inference hfdask
   curl -fL https://raw.githubusercontent.com/Hanno-Labs/hfdask/v0.1.1/examples/job.py -o job.py
   curl -fL https://raw.githubusercontent.com/Hanno-Labs/hfdask/v0.1.1/examples/inference.yaml -o cluster.yaml
   ```

2. Authenticate with `uv run --group inference hf auth login` and set `namespace`
   in `cluster.yaml`.
3. Create a private output bucket and seed a fresh prefix such as
   `hf://buckets/your-namespace/output/ag-news-run-001`; set it as `/output`.
4. Review the `cpu-basic` and `l4x1` flavors and 30-minute timeout, then run:

   ```sh
   uv run --group inference hfdask run --cluster cluster.yaml job.py
   ```

The model and dataset are mounted directly from the Hub. No input bucket, manual
model staging, or custom Dockerfile is required.

The YAML uses `vllm/vllm-openai:v0.29.0`, pinned to its Linux amd64 digest. The
image supplies CUDA tooling, Python, and uv; hfdask runs its own bootstrap instead
of the serving entrypoint. The locked project environment installs vLLM 0.29.0
separately, so keep the image and lockfile versions compatible.

### Graph and placement

The graph has three stages:

1. `prepare` runs on a CPU worker, reads the mounted AG News test Parquet, samples
   128 deterministic rows, and creates eight 16-row partitions.
2. `infer` tasks request `resources={"GPU": 1}` and reuse a Qwen3-0.6B engine loaded
   once by the GPU worker's `WorkerSetup` plugin.
3. Concatenation and `save_results` run on CPU workers and write durable output.

CPU addresses are selected from scheduler metadata where the `GPU` resource is
absent or zero. CPU stages use hard worker affinity; inference uses Dask's numeric
GPU reservation. `summary.compute(optimize_graph=False)` prevents optimization from
fusing those placement boundaries.

Both Hub repositories are pinned in YAML and accessed only through filesystem paths:

| Input | Mount | Revision |
|---|---|---|
| `fancyzhx/ag_news` | `/dataset` | `eb185aade064a813bc0b7f42de02595523103ca4` |
| `Qwen/Qwen3-0.6B` | `/model` | `c1899de289a04d12100db370d81485cdf75e47ca` |

The [AG News card](https://huggingface.co/datasets/fancyzhx/ag_news) lists its
license as unknown and describes research use. Public access does not establish
unrestricted reuse rights. [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
is Apache-2.0.

### Observability and output

Workers emit structured Dask events on the `inference` topic for preparation,
model loading, partition completion, and result saving. While the cluster is live,
inspect the scheduler's bounded event history with
`get_client().get_events("inference")`.

The CPU output stage writes:

- `/output/results.parquet`: `id`, `text`, `label`, and `prediction`;
- `/output/summary.json`: row count, invalid-prediction count, and accuracy.

Metrics are descriptive, not pass/fail thresholds. There is no per-partition
checkpoint, skip-existing behavior, or automatic resume. Use a fresh output prefix
when retrying. See the root [cleanup and recovery](../README.md#cleanup-and-recovery)
section if the submitting process or cleanup fails.
