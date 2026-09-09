import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


@pytest.fixture
def example(monkeypatch):
    loads = []

    class Engine:
        def __init__(self, **kwargs):
            loads.append(kwargs)

        def generate(self, prompts, params, use_tqdm):
            return [SimpleNamespace(outputs=[SimpleNamespace(text="answer")]) for _ in prompts]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kwargs: kwargs))
    spec = importlib.util.spec_from_file_location(
        "example_job", Path(__file__).parents[1] / "examples" / "job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = SimpleNamespace()
    monkeypatch.setattr(module, "get_worker", lambda: worker)
    return module, loads


def test_engine_reused_and_index_preserved(example):
    job, loads = example
    rows = pd.DataFrame({"id": ["a", "b"], "prompt": ["hello", "world"]}, index=[5, 9])
    result = job.infer(rows)
    assert result.to_dict("list") == {"id": ["a", "b"], "text": ["answer", "answer"]}
    assert result.index.tolist() == [5, 9]
    job.infer(rows)
    assert len(loads) == 1
    assert loads[0]["tensor_parallel_size"] == 1


def test_empty_partition_does_not_load_model(example):
    job, loads = example
    rows = pd.DataFrame({"id": pd.Series(dtype="str"), "prompt": pd.Series(dtype="str")})
    assert job.infer(rows).empty
    assert loads == []


def test_dataframe_partition_workflow(example):
    import dask.dataframe as dd

    job, _loads = example
    rows = pd.DataFrame({"id": ["a", "b"], "prompt": ["hello", "world"]})
    result = dd.from_pandas(rows, npartitions=2).map_partitions(
        job.infer, meta={"id": "str", "text": "str"}).compute(scheduler="synchronous")
    assert result["text"].tolist() == ["answer", "answer"]
