import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from distributed import (
    Client,
    LocalCluster,
    Scheduler,
    SpecCluster,
    Worker,
    get_task_stream,
    get_worker,
)


@pytest.fixture
def example(monkeypatch):
    state = SimpleNamespace(loads=[], chats=[], batches=[], events=[], reply=" World \n")

    class Engine:
        def __init__(self, **kwargs):
            state.loads.append(kwargs)

        def get_tokenizer(self):
            return self

        def apply_chat_template(self, messages, **kwargs):
            state.chats.append((messages, kwargs))
            return "chat:" + messages[0]["content"]

        def generate(self, prompts, params, use_tqdm):
            state.batches.append((prompts, params, use_tqdm))
            return [SimpleNamespace(outputs=[] if state.reply is None else
                    [SimpleNamespace(text=state.reply)]) for _ in prompts]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kwargs: kwargs))
    spec = importlib.util.spec_from_file_location(
        "example_job", Path(__file__).parents[1] / "examples" / "job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state.worker = SimpleNamespace(
        address="tcp://fake:1234", state=SimpleNamespace(total_resources={"GPU": 1}),
        log_event=lambda topic, event: state.events.append((topic, event.copy())))
    monkeypatch.setattr(module, "get_worker", lambda: state.worker)
    return module, state


def inference_events(client, count):
    deadline = time.monotonic() + 10
    while True:
        events = [event for _, event in client.get_events("inference")]
        if len(events) >= count:
            return events
        if time.monotonic() >= deadline:
            pytest.fail(f"Expected {count} inference events, received {events!r}")
        time.sleep(0.05)


@pytest.fixture
def source():
    return pd.DataFrame({"text": [f"article {i}" for i in range(200)],
                         "label": [i // 50 for i in range(200)]}, index=range(1000, 1200))


def test_sample_determinism_balance_and_ids(example, source):
    job, _ = example
    sample = job.select_sample(source)
    pd.testing.assert_frame_equal(sample, job.select_sample(source))
    assert sample.groupby("label").size().to_dict() == dict.fromkeys(job.CATEGORIES, 32)
    expected = source.groupby("label", sort=True).sample(n=32, random_state=23)
    assert sample.id.tolist() == sorted(expected.index)
    assert sample.id.is_unique
    assert sample.text.tolist() == source.loc[sample.id, "text"].tolist()


def test_worker_setup_engine_reuse_template_parameters_and_logging(example, source, capsys):
    job, state = example
    rows = job.select_sample(source).iloc[5:21]
    job.WorkerSetup().setup(state.worker)
    assert len(state.loads) == 1
    assert not state.batches
    result = job.infer(rows)
    job.infer(rows)
    assert result.index.equals(rows.index)
    pd.testing.assert_frame_equal(result.drop(columns="prediction"), rows)
    assert result.prediction.tolist() == ["World"] * 16
    assert state.loads == [{"model": "/model",
        "tensor_parallel_size": 1, "max_model_len": 2048, "max_num_seqs": 16,
        "gpu_memory_utilization": 0.5, "enforce_eager": True, "trust_remote_code": False, "seed": 23}]
    assert state.chats[0][1] == {"enable_thinking": False,
                               "add_generation_prompt": True, "tokenize": False}
    assert all(category in state.chats[0][0][0]["content"] for category in job.CATEGORIES)
    assert rows.iloc[0].text in state.chats[0][0][0]["content"]
    assert state.batches[0][1:] == ({"temperature": 0, "max_tokens": 16, "seed": 23}, False)
    assert capsys.readouterr().out == ""
    assert [topic for topic, _ in state.events] == ["inference"] * 4
    events = [event for _, event in state.events]
    assert events[0] == {"phase": "model_loading"}
    assert set(events[1]) == {"phase", "elapsed"}
    assert events[1]["phase"] == "model_loaded"
    assert events[1]["elapsed"] >= 0
    for event in events[2:]:
        assert set(event) == {"phase", "rows", "elapsed"}
        assert event["phase"] == "partition_complete"
        assert event["rows"] == 16
        assert event["elapsed"] >= 0
    state.worker = SimpleNamespace(
        address="tcp://another:1234", state=SimpleNamespace(total_resources={"GPU": 1}),
        log_event=state.worker.log_event)
    job.WorkerSetup().setup(state.worker)
    job.infer(rows)
    assert len(state.loads) == 2


@pytest.mark.parametrize("resources", [{}, {"GPU": 0}])
def test_worker_setup_skips_cpu_workers(example, resources):
    job, state = example
    state.worker.state.total_resources = resources
    job.WorkerSetup().setup(state.worker)
    assert not hasattr(state.worker, "inference_model")
    assert not state.loads
    assert not state.events


def test_prepare_partitioning_and_logging(example, source, tmp_path):
    job, state = example
    path = tmp_path / "source.parquet"
    source.to_parquet(path)
    partitions = job.prepare(path)
    assert isinstance(partitions, list)
    assert len(partitions) == 8
    assert all(len(partition) == 16 for partition in partitions)
    pd.testing.assert_frame_equal(pd.concat(partitions), job.select_sample(source))
    assert state.events == [("inference", {"phase": "prepared", "rows": 128})]
    assert not state.loads


def test_empty_partition_and_completion(example, source, monkeypatch):
    job, state = example
    rows = job.select_sample(source)
    with monkeypatch.context() as patch:
        patch.setattr(job, "get_worker", lambda: pytest.fail("empty partition used worker"))
        result = job.infer(rows.iloc[:0])
    assert result.empty and list(result.columns) == ["id", "text", "label", "prediction"]
    assert not state.loads
    assert not state.events
    state.reply = None
    job.WorkerSetup().setup(state.worker)
    assert job.infer(rows.iloc[:1]).prediction.tolist() == [""]


@pytest.mark.parametrize("gpu_workers", [1, 2])
def test_main_mounted_dataset_partitioning_and_summary(
        example, source, tmp_path, monkeypatch, capsys, gpu_workers):
    job, _ = example
    source_path = tmp_path / "source.parquet"
    source.to_parquet(source_path)
    monkeypatch.setattr(job, "DATASET", source_path)
    monkeypatch.setattr(job, "OUTPUT", tmp_path / "output")
    monkeypatch.setattr(job, "get_worker", get_worker)
    worker_specs = {
        "cpu": {"cls": Worker, "options": {"nthreads": 1, "resources": {}}},
        **{f"gpu-{i}": {"cls": Worker, "options": {
            "nthreads": 1, "resources": {"GPU": 1}}} for i in range(gpu_workers)},
    }
    with SpecCluster(workers=worker_specs, scheduler={
            "cls": Scheduler, "options": {"dashboard_address": None}}) as cluster, \
            Client(cluster) as client:
        client.wait_for_workers(1 + gpu_workers, timeout=10)
        workers = client.scheduler_info()["workers"]
        cpus = {address for address, info in workers.items()
                if not info["resources"].get("GPU", 0)}
        gpus = set(workers) - cpus
        assert len(cpus) == 1
        assert len(gpus) == gpu_workers
        with get_task_stream(client) as tasks:
            job.main()
        events = inference_events(client, 10 + 2 * gpu_workers)
        models = client.run(lambda dask_worker: hasattr(dask_worker, "inference_model"))
        assert models == {address: address in gpus for address in workers}

    assert len(events) == 10 + 2 * gpu_workers
    assert all(event["worker"] in workers for event in events)
    for address in gpus:
        model_events = [event for event in events if event["worker"] == address
                        and event["phase"].startswith("model_")]
        assert [event["phase"] for event in model_events] == ["model_loading", "model_loaded"]
    for phase in ("prepared", "results_saved"):
        phase_events = [event for event in events if event["phase"] == phase]
        assert len(phase_events) == 1
        assert phase_events[0]["worker"] in cpus
        assert phase_events[0]["rows"] == 128
    completed = [event for event in events if event["phase"] == "partition_complete"]
    assert len(completed) == 8
    assert all(event["worker"] in gpus for event in completed)
    assert all(event["rows"] == 16 and event["elapsed"] >= 0 for event in completed)
    for name, allowed, count in (("prepare", cpus, 1), ("infer", gpus, 8),
                                 ("concat", cpus, 1), ("save_results", cpus, 1)):
        executed = [task for task in tasks.data if str(task["key"]).startswith(name + "-")]
        assert len(executed) == count
        assert all(task["worker"] in allowed for task in executed)
    summary = json.loads((job.OUTPUT / "summary.json").read_text())
    assert summary == {"rows": 128, "invalid_predictions": 0, "accuracy": 0.25}
    assert json.loads(capsys.readouterr().out) == summary
    pd.testing.assert_frame_equal(
        pd.read_parquet(job.OUTPUT / "results.parquet"),
        job.select_sample(source).assign(prediction="World"), check_dtype=False)


@pytest.mark.parametrize("resources,missing", [({}, "GPU"), ({"GPU": 1}, "CPU")])
def test_main_requires_cpu_and_gpu_workers(example, monkeypatch, resources, missing):
    job, state = example
    client = SimpleNamespace(
        scheduler_info=lambda: {"workers": {"worker": {"resources": resources}}},
        register_plugin=lambda *args, **kwargs: pytest.fail("registered before validation"))
    monkeypatch.setattr(job, "get_client", lambda: client)
    with pytest.raises(ValueError, match=missing):
        job.main()
    assert not state.loads
    assert not state.events


def test_plugin_initializes_later_workers(example, source, monkeypatch):
    job, _ = example
    monkeypatch.setattr(job, "get_worker", get_worker)
    rows = job.select_sample(source).iloc[:16]
    with LocalCluster(n_workers=1, threads_per_worker=1, processes=False,
                      resources={"GPU": 1}, dashboard_address=None) as cluster, Client(cluster) as client:
        client.register_plugin(job.WorkerSetup(), name="job-setup")
        original = set(client.scheduler_info()["workers"])
        cluster.scale(2)
        client.wait_for_workers(2, timeout=10)
        workers = set(client.scheduler_info()["workers"])
        assert len(workers - original) == 1
        for address in workers:
            for _ in range(2):
                result = client.submit(job.infer, rows, workers=[address], pure=False).result()
                assert result.prediction.tolist() == ["World"] * 16
        events = inference_events(client, 8)
    assert len(events) == 8
    assert {event["worker"] for event in events} == workers
    for address in workers:
        phases = [event["phase"] for event in events if event["worker"] == address]
        assert phases == ["model_loading", "model_loaded",
                          "partition_complete", "partition_complete"]


def test_infer_does_not_initialize_model(example, source):
    job, state = example
    with pytest.raises(AttributeError, match="inference_model"):
        job.infer(job.select_sample(source).iloc[:1])
    assert not state.loads


@pytest.mark.parametrize("predictions,labels,invalid,accuracy", [
    pytest.param(["World", "Sports", "Business", "Sci/Tech"],
                 ["World", "Sports", "Business", "Sci/Tech"], 0, 1.0, id="all-correct"),
    pytest.param(["Sports", "World"], ["World", "Sports"], 0, 0.0, id="valid-but-wrong"),
    pytest.param(["World", "Sports", "world", "", None, pd.NA],
                 ["World", "World", "World", "World", "World", "World"],
                 4, 1 / 6, id="mixed-invalid-and-missing"),
    pytest.param(["World", "Sports", pd.NA], ["World", pd.NA, pd.NA],
                 1, 1 / 3, id="missing-labels-count-as-incorrect"),
    pytest.param(["world", "", None], ["World", "World", "World"],
                 3, 0.0, id="all-invalid"),
    pytest.param([], [], 0, None, id="empty"),
])
def test_save_results_metrics_and_parquet(example, tmp_path, capsys,
                                         predictions, labels, invalid, accuracy):
    job, state = example
    results = pd.DataFrame({
        "id": range(len(predictions)),
        "text": pd.Series([f"article {i}" for i in range(len(predictions))], dtype="string"),
        "label": pd.Series(labels, dtype="string"),
        "prediction": pd.Series(predictions, dtype="string"),
    })
    results.index = range(100, 100 + len(results))
    output = tmp_path / "nested" / "output"
    summary = job.save_results(results, output)
    expected = {"rows": len(results), "invalid_predictions": invalid,
                "accuracy": pytest.approx(accuracy) if accuracy is not None else None}
    assert summary == expected
    assert state.events == [("inference", {"phase": "results_saved", "rows": len(results)})]
    assert json.loads((output / "summary.json").read_text()) == expected
    assert capsys.readouterr().out == ""
    pd.testing.assert_frame_equal(pd.read_parquet(output / "results.parquet"),
                                  results.reset_index(drop=True))
