import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "offline", Path(__file__).parents[1] / "examples" / "offline.py")
offline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(offline)


def test_rows_and_durable_output(tmp_path):
    path = tmp_path / "batch.jsonl"
    path.write_text('{"id":"1","prompt":"Hello"}\n')
    rows = offline.read_batch(str(path))
    manifest = offline.save_batch(rows, str(tmp_path / "out"), path.name)
    assert manifest["rows"] == 1
    assert json.loads((tmp_path / "out/batch.manifest.json").read_text()) == manifest
    assert offline.read_batch(str(tmp_path / "out/batch.jsonl")) == rows
    path.write_text('{"id":"1","prompt":"Hello"}\n' * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        offline.read_batch(str(path))


def test_engine_reused(monkeypatch):
    loads = []
    worker = SimpleNamespace(address="gpu")

    class Engine:
        def __init__(self, **kwargs):
            loads.append(kwargs)

        def generate(self, prompts, params, use_tqdm):
            return [SimpleNamespace(outputs=[SimpleNamespace(text="answer")]) for _ in prompts]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kwargs: kwargs))
    monkeypatch.setattr(offline, "get_worker", lambda: worker)
    rows = [{"id": "a", "prompt": "hello"}]
    assert offline.generate(rows, "/model", 10)[0]["text"] == "answer"
    offline.generate(rows, "/model", 10)
    assert len(loads) == 1
    assert loads[0]["tensor_parallel_size"] == 1
    with pytest.raises(ValueError, match="switch"):
        offline.generate(rows, "/other", 10)
