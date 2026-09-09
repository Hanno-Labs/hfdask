"""AG News GPU smoke: ordinary Dask partitions, worker-local vLLM replicas."""

import json
import time
from pathlib import Path

import dask
import pandas as pd
from distributed import Worker, WorkerPlugin, get_client, get_worker
from vllm import LLM, SamplingParams

DATASET = Path("/dataset/data/test-00000-of-00001.parquet")
MODEL = "/model"
CATEGORIES = ("World", "Sports", "Business", "Sci/Tech")
OUTPUT = Path("/output")


def select_sample(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows[["text", "label"]].copy()
    rows.insert(0, "id", rows.index)
    sample = rows.groupby("label", sort=True).sample(n=32, random_state=23)
    sample["label"] = sample["label"].map(dict(enumerate(CATEGORIES))).astype("str")
    return sample.sort_values("id").reset_index(drop=True)


def prepare(path: Path) -> list[pd.DataFrame]:
    selected = select_sample(pd.read_parquet(path))
    get_worker().log_event("inference", {"phase": "prepared", "rows": len(selected)})
    return [selected.iloc[start : start + 16] for start in range(0, len(selected), 16)]


class WorkerSetup(WorkerPlugin):
    def setup(self, worker: Worker) -> None:
        if not worker.state.total_resources.get("GPU", 0):
            return
        started = time.monotonic()
        worker.log_event("inference", {"phase": "model_loading"})
        worker.inference_model = LLM(
            model=MODEL,
            tensor_parallel_size=1,
            max_model_len=2048,
            max_num_seqs=16,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
            trust_remote_code=False,
            seed=23,
        )
        worker.log_event(
            "inference",
            {
                "phase": "model_loaded",
                "elapsed": time.monotonic() - started,
            },
        )


def infer(partition: pd.DataFrame) -> pd.DataFrame:
    result = partition[["id", "text", "label"]].copy()
    result["prediction"] = pd.Series("", index=partition.index, dtype="str")
    if partition.empty:
        return result
    started = time.monotonic()
    worker = get_worker()

    engine = worker.inference_model
    tokenizer = engine.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "Classify this news article. Reply with exactly one "
                    "category: World, Sports, Business, Sci/Tech. No explanation.\n\n" + text,
                }
            ],
            enable_thinking=False,
            add_generation_prompt=True,
            tokenize=False,
        )
        for text in partition["text"]
    ]
    outputs = engine.generate(
        prompts,
        SamplingParams(temperature=0, max_tokens=16, seed=23),
        use_tqdm=False,
    )
    result["prediction"] = [
        output.outputs[0].text.strip() if output.outputs else "" for output in outputs
    ]
    worker.log_event(
        "inference",
        {
            "phase": "partition_complete",
            "rows": len(partition),
            "elapsed": time.monotonic() - started,
        },
    )
    return result


def save_results(results: pd.DataFrame, output: Path) -> dict[str, int | float | None]:
    output.mkdir(parents=True, exist_ok=True)
    results.to_parquet(output / "results.parquet", index=False)
    summary = {
        "rows": len(results),
        "invalid_predictions": int((~results["prediction"].isin(CATEGORIES)).sum()),
        "accuracy": float(results["prediction"].eq(results["label"]).fillna(False).mean())
        if len(results)
        else None,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    get_worker().log_event("inference", {"phase": "results_saved", "rows": len(results)})
    return summary


def main() -> None:
    client = get_client()
    workers = client.scheduler_info()["workers"]
    cpu_workers = [
        address for address, info in workers.items() if not info["resources"].get("GPU", 0)
    ]
    if not cpu_workers or len(cpu_workers) == len(workers):
        raise ValueError("This example requires both CPU-only and GPU workers")
    client.register_plugin(WorkerSetup(), name="job-setup")

    with dask.annotate(workers=cpu_workers, allow_other_workers=False):
        partitions = list(dask.delayed(prepare, nout=8)(DATASET))
    with dask.annotate(resources={"GPU": 1}):
        predictions = [dask.delayed(infer)(partition) for partition in partitions]
    with dask.annotate(workers=cpu_workers, allow_other_workers=False):
        results = dask.delayed(pd.concat)(predictions)
        summary = dask.delayed(save_results)(results, OUTPUT)

    # Keep the separately annotated CPU/GPU stages from being fused together.
    print(json.dumps(summary.compute(optimize_graph=False)), flush=True)


if __name__ == "__main__":
    main()
