# Offline vLLM: CPU coordinator + four H200 workers

The main example is `offline.py`: one `cpu-basic` Job hosts the scheduler and
a CPU coordinator worker; four separate `h200` Jobs each run one persistent
vLLM replica. Five Jobs, five Dask workers, four independent GPUs. This is
data-parallel batch inference, not a model spread across four GPUs.

1. Build `examples/Dockerfile` from the repository root with
   `--build-arg VLLM_IMAGE=<tested-vllm-image@sha256:...>`. Supply an immutable
   CUDA/vLLM image compatible with H200 and a model that fits one GPU. The image
   must be pullable by HF Jobs. Building/publishing it is an explicit operator
   step; this example does not publish anything or create credentials.
2. Stage a complete, pinned model snapshot (including tokenizer) under a private
   bucket's `model/`, and JSONL shards under `input/`. Each row has string `id`
   and `prompt` fields. Use globally unique IDs. Start with 32 prompts per shard
   and raw completion prompts suitable for your model; no chat template is
   silently applied. Prompts plus requested output must fit 4096 tokens.
3. Mount that immutable bucket read-only at `/data`, and a separate existing
   private output bucket at `/output`. Verify ownership/privacy first. Outputs
   must use a fresh prefix for each run; this example does not skip old shards.
4. Provide five distinct existing Iroh identities and explicitly choose the
   relay policy. The following is the full submission shape:

```python
import json
import os
from pathlib import Path

from huggingface_hub import Volume
from hfdask import JobSpec, WorkerGroup
from hfdask.cluster import Identity, submit_cluster

identities = [Identity(bytes.fromhex(os.environ[f"NODE_KEY_{i}"])) for i in range(5)]
spec = JobSpec(
    namespace="your-namespace",
    image="your-pullable-image@sha256:...",
    entrypoint="offline:run",
    flavor="cpu-basic",
    workers=5,
    threads_per_worker=1,
    timeout="2h",
    kwargs={"input": "/data/input", "model": "/data/model",
            "output": "/output/run-001", "max_tokens": 128},
    volumes=[
        Volume(type="bucket", source="your-namespace/input-bucket",
               mount_path="/data", read_only=True),
        Volume(type="bucket", source="your-namespace/output-bucket",
               mount_path="/output"),
    ],
)
cluster = submit_cluster(
    spec, identities, public_relays=True,
    scheduler_worker=True,
    worker_groups=[WorkerGroup("h200", count=4, tags=("inference",))],
    on_submitted=lambda c: Path("cluster.json").write_text(json.dumps(c.manifest())),
)
try:
    cluster.wait(timeout=7200)
finally:
    cluster.close()
```

Group counts describe remote machines; the colocated CPU worker is additional.
Every GPU task reserves `GPU=1` and is pinned to an H200. The driver maintains
at most one in-flight inference batch per GPU, dynamically replenishing the
worker that finishes. CPU tasks read shards and save output; the driver runs
beside them on the scheduler Job. Models are loaded lazily once per worker and
reused, not sent through Dask. Worker restart requires model reload.

Each completed shard is written to the durable output mount, read back, hashed,
then given a manifest. Earlier completed shards survive later failures; automatic
resume/skip and global output consolidation are not implemented in this example.
After completion, download outputs to a fresh directory and verify manifest
hashes and row counts. Preserve `cluster.json` on launch errors and inspect/cancel
every recorded Job (including partially submitted clusters).

The Dockerfile disables daemon workers so vLLM can spawn subprocesses and sets
the spawn method before CUDA imports. vLLM is imported only inside GPU tasks,
so the CPU coordinator never loads the model. All nodes currently use the same
image. Hardware discovery requires `nvidia-smi` in the GPU runtime.

References: [vLLM offline batches](https://docs.vllm.ai/en/latest/getting_started/quickstart/#offline-batched-inference)
and [multiprocessing constraints](https://docs.vllm.ai/en/latest/usage/troubleshooting/#python-multiprocessing).
The example has CPU-side tests with a fake inference engine; an actual four-H200
vLLM execution has **not** been validated. The transport proof below is separate.

## Three-job encrypted Dask proof

This example uses synthetic bytes only. It starts a scheduler and two workers in
separate HF Jobs, with no public Dask ports and no VPN account. Public n0
discovery/relays are explicitly enabled; Iroh authenticates peers and encrypts
traffic end to end. Relay availability is not a production SLA.

From the repository root, install dependencies and generate disposable keys:

```sh
uv sync --extra p2p --group dev
uv run python examples/prepare.py /tmp/hfdask-example-keys --allow-public-relays
```

Stage this repository (excluding `.git`, `.venv`, and all secrets) plus the
generated **public** `mesh.json` in a private HF bucket prefix. Set `SOURCE_URI`
to that prefix and `OUTPUT_URI` to a separate writable prefix. Seed the output
prefix with a small file before mounting. Set `NAMESPACE` to your HF namespace.
Verify bucket ownership and privacy before uploading. Never upload `node-*.env`.

Run this once for each `NODE` value 0, 1, and 2:

```sh
hf jobs run --namespace "$NAMESPACE" --flavor cpu-basic --timeout 15m --detach \
  --secrets-file "/tmp/hfdask-example-keys/node-$NODE.env" \
  --volume "$SOURCE_URI:/source:ro" --volume "$OUTPUT_URI:/output" \
  python:3.11-slim sh /source/examples/job.sh "$NODE"
```

Record each returned job ID immediately. Node 0 drives the workload and closes
the workers. Inspect **all three** terminal states, and cancel any survivors.
Retrieve `/output/result.json` from the bucket in a fresh directory.

To colocate one of the two workers with the scheduler, pass `--scheduler-worker`
to `prepare.py`, launch only nodes 0 and 1, and append `--scheduler-worker` after
`"$NODE"` in both Job commands. The same proof then transfers from the remote
worker (port 21001) to the scheduler's worker (port 21002). The receiver logs
`mesh link authenticated node=0 target=1`.

The proof pins a 4 MiB payload producer to worker 1 and its dependent consumer to
worker 2. It checks the complete SHA-256, byte count, worker addresses, and
distinct hostnames. The client never gathers the payload. Worker 2 must also log
`mesh link authenticated node=2 target=1`, proving its data fetch traversed the
authenticated mesh rather than a direct TCP shortcut.

For a smaller real transport/security test, without public discovery or relays:

```sh
HFDASK_TEST_KEYS=1 uv run pytest -q tests/test_iroh_integration.py
```

That test checks a 1 MiB transfer, TCP half-close, and rejection of an unlisted
peer. It does **not** by itself prove cross-job discovery or relay connectivity.

## Recorded verification (2026-09-09)

### Heterogeneous CPU/GPU proof

Generate the mesh with:

```sh
uv run python examples/prepare.py /tmp/hfdask-mixed-keys --allow-public-relays \
  --flavors cpu-basic cpu-performance t4-small
```

Use the same staging and launch procedure above, but launch node 0 with
`--flavor cpu-basic`, node 1 with `--flavor cpu-performance`, and node 2 with
`--flavor t4-small`. Use the matching private key directory and append
`--kwargs '{"hardware":true}'` to each job command. Flavor labels in mesh.json
must match the actual requested Jobs; hardware capacities are detected at runtime.

This configuration passed on three real HF Jobs, all `COMPLETED`. The proof
routed the producer by CPU flavor and consumer by `HAS_GPU`, reserving `GPU=1`.
The consumer detected a Tesla T4 with 15 GiB VRAM, advertised a conservative
13.5 GiB VRAM budget, and its CUDA visibility matched the detected GPU UUID.
The CPU worker advertised no GPU. The same 4 MiB digest below was verified after
fresh durable readback, with an authenticated node-2-to-node-1 link. This verifies
placement and isolation, not CUDA kernel performance. Multi-GPU partitioning is
unit-tested but has not yet been exercised on a physical multi-GPU Job.

The proof intentionally caps each worker at one thread and 1 GiB RAM; inventory
metadata separately records full detected node capacity. Post-completion cleanup
still emits stream/worker shutdown warnings.

The colocated option also passed: two HF Jobs both completed, with the consumer
on the scheduler Job and the producer on the remote Job. The same 4 MiB digest
was verified from durable output, with `node=0 target=1` authenticated-link
evidence. The updated test suite passes 32 tests.

- Three separate HF `cpu-basic` Jobs all reached `COMPLETED`.
- Producer and consumer ran on distinct Job hostnames at worker ports 21001/21002.
- 4,194,304 bytes transferred; SHA-256:
  `15e4653eb3c495583132a2385bbd8572eeeed7e098eaa394bc0d584601f3ebd9`.
- Receiver logged `mesh link authenticated node=2 target=1`.
- The result was retrieved from durable storage and independently verified.
- All 26 tests passed, including the real QUIC/unauthorized-peer test; Ruff and
  strict mypy checks of the library passed.

This run used public discovery/relay configuration, but did not force a relay-only
route or measure which physical route Iroh selected. It proves cross-job Dask
execution through authenticated Iroh, not every NAT/firewall scenario. A stream
close warning occurred during shutdown after successful computation; all Jobs
nevertheless exited successfully. This is a smoke test, not a resilience or load
benchmark. No account-specific identifiers or test secrets are included here.
