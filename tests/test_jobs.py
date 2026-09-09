import json
from enum import Enum
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from huggingface_hub import HfApi, Volume

from hfdask import Job, JobFailed, JobSpec, submit


def spec(**kwargs):
    return JobSpec(namespace="example", image="registry/image:tag",
                   entrypoint="workload:run", **kwargs)


@pytest.mark.parametrize("kwargs", [{"workers": 0}, {"threads_per_worker": 0},
                                  {"kwargs": {"bad": float("nan")}}])
def test_invalid_spec(kwargs):
    with pytest.raises(ValueError):
        spec(**kwargs)


def test_volume_validation():
    with pytest.raises(ValueError):
        spec(volumes=[Volume(type="bucket", source="example/data", mount_path="relative")])


def test_submit_has_no_shell_or_forwarded_secrets():
    api = MagicMock(spec=HfApi)
    api.run_job.return_value.id = "job1"
    config = spec(kwargs={"text": "$(do-not-execute); hello"})
    job = submit(config, api=api)
    assert job.id == "job1"
    args = api.run_job.call_args.kwargs
    assert args["namespace"] == "example"
    assert json.loads(args["command"][-1]) == config.kwargs
    assert not {"env", "secrets", "expose", "ssh"}.intersection(args)


@pytest.mark.parametrize("stage", ["ERROR", "CANCELED", "DELETED"])
def test_failed(stage):
    api = MagicMock(spec=HfApi)
    api.inspect_job.return_value.status.stage = stage
    with pytest.raises(JobFailed, match=stage):
        Job("job1", "example", api).wait()


def test_success_transition(monkeypatch):
    api = MagicMock(spec=HfApi)
    api.inspect_job.side_effect = [SimpleNamespace(status=SimpleNamespace(stage=s))
                                   for s in ["RUNNING", "COMPLETED"]]
    monkeypatch.setattr("hfdask.jobs.time.sleep", lambda _: None)
    assert Job("job1", "example", api).wait() == "COMPLETED"


def test_timeout_does_not_cancel():
    api = MagicMock(spec=HfApi)
    api.inspect_job.return_value.status.stage = "RUNNING"
    with pytest.raises(TimeoutError, match="not canceled"):
        Job("job1", "example", api).wait(timeout=0)
    api.cancel_job.assert_not_called()


def test_cancel_scoped():
    api = MagicMock(spec=HfApi)
    Job("job1", "example", api).cancel()
    api.cancel_job.assert_called_once_with(job_id="job1", namespace="example")


def test_enum_status():
    class Stage(str, Enum):
        COMPLETED = "COMPLETED"

    api = MagicMock(spec=HfApi)
    api.inspect_job.return_value.status.stage = Stage.COMPLETED
    assert Job("job1", "example", api).wait(timeout=0) == "COMPLETED"
