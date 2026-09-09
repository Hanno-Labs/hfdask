"""Offline inference: Dask distributes partitions, vLLM batches each partition."""

import dask.dataframe as dd
import pandas as pd
from distributed import get_worker
from vllm import LLM, SamplingParams

MODEL = "/input/model"


def infer(partition: pd.DataFrame) -> pd.DataFrame:
    if partition.empty:
        return pd.DataFrame({"id": partition["id"], "text": pd.Series(dtype="str")})
    worker = get_worker()
    # Keep the replica on its GPU worker; never serialize a loaded model into a task.
    if not hasattr(worker, "inference_model"):
        worker.inference_model = LLM(
            model=MODEL, tensor_parallel_size=1, max_model_len=4096,
            gpu_memory_utilization=0.85, trust_remote_code=False,
        )
    outputs = worker.inference_model.generate(
        partition["prompt"].tolist(), SamplingParams(temperature=0, max_tokens=128),
        use_tqdm=False,
    )
    return pd.DataFrame(
        {"id": partition["id"], "text": [output.outputs[0].text for output in outputs]},
        index=partition.index,
    )


def main() -> None:
    dataset = dd.read_parquet("/input/prompts/*.parquet")
    predictions = dataset.map_partitions(infer, meta={"id": "str", "text": "str"})
    results = predictions.compute()
    results.to_parquet("/output/results.parquet", index=False)


if __name__ == "__main__":
    main()
