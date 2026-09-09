import base64
import io
import json
import random
import subprocess
import tarfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hfdask import cli
from hfdask.cluster import Cluster, Identity, LaunchError, submit_cluster
from hfdask.jobs import JobSpec, submit


@pytest.fixture
def project(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n'
        'dependencies = ["hfdask[p2p]==0.1.0"]\n'
        '[project.optional-dependencies]\ninference = []\n'
    )
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "job.py").write_text("from distributed import Client\nclient = Client()\n")
    (tmp_path / "inference.yaml").write_text(
        "namespace: example\ncoordinator:\n  flavor: cpu-basic\n"
        "workers:\n  flavor: h200\n  count: 4\nenvironment:\n"
        "  image: ghcr.io/astral-sh/uv:python3.12-bookworm-slim\n"
        "  extras: [inference]\ntimeout: 2h\nnetwork:\n  public_relays: true\n"
        "mounts:\n  - source: hf://buckets/example/data\n"
        "    target: /data\n    read_only: true\n"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_yaml_contract(project):
    spec, options, timeout = cli.load_cluster(project / "inference.yaml", project, "job.py")
    assert spec.workers == 4
    assert spec.flavor == "h200"
    assert spec.entrypoint == "hfdask.runner:run_script"
    assert spec.kwargs == {"script": "job.py"}
    assert spec.volumes[0].source == "example/data"
    assert spec.volumes[0].read_only is True
    assert options == {"public_relays": True, "scheduler_worker": False,
                       "scheduler_flavor": "cpu-basic"}
    assert timeout == 7200
    assert spec.env["DASK_DISTRIBUTED__WORKER__DAEMON"] == "False"
    assert spec.bootstrap[:2] == ("python", "-c")
    assert spec.env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert json.loads(spec.bootstrap[3]) == ["inference"]
    assert int(spec.bootstrap[4]) == len(spec.bootstrap) - 5
    assert spec.command()[len(spec.bootstrap):][:4] == [
        "python", "-m", "hfdask.runner", "hfdask.runner:run_script"]


def test_archive_selection(project):
    (project / ".gitignore").write_text("ignored.txt\ntracked-ignored.txt\n")
    for name in ("ignored.txt", "tracked-ignored.txt", ".env", "private.pem", "mise.local.toml"):
        (project / name).write_text("secret")
    subprocess.run(["git", "add", "-f", "tracked-ignored.txt", ".env"], check=True)
    bootstrap = cli.package_project(project, "job.py", ["inference"])
    payload = base64.b64decode("".join(bootstrap[5:]))
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        names = archive.getnames()
        assert {"job.py", "uv.lock", "pyproject.toml"} <= set(names)
        assert not set(names) & {"ignored.txt", "tracked-ignored.txt", ".env", "private.pem",
                                 "mise.local.toml"}


@pytest.mark.parametrize("name", [
    "node-0.env", "config/node-worker.env", "secrets/node-key",
    ".secrets/node-key", "config/secrets/node-key", "config/.secrets/node-key",
])
@pytest.mark.parametrize("tracked", [False, True])
def test_archive_excludes_node_keys_and_secret_directories(project, name, tracked):
    path = project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("HFDASK_NODE_KEY=private-key-material\n")
    if tracked:
        subprocess.run(["git", "add", "-f", "--", name], check=True)
    bootstrap = cli.package_project(project, "job.py", [])
    payload = base64.b64decode("".join(bootstrap[5:]))
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        assert name not in archive.getnames()
        for member in archive.getmembers():
            with archive.extractfile(member) as source:
                assert b"private-key-material" not in source.read()


@pytest.mark.parametrize("script", ["../job.py", "/job.py", "scripts/../job.py", "job.txt"])
def test_script_must_be_project_relative(project, script):
    with pytest.raises(ValueError, match="project-relative"):
        cli.package_project(project, script, [])


@pytest.mark.parametrize("script", ["examples/job.py", "./examples/job.py"])
def test_nested_script(project, script):
    (project / "examples").mkdir()
    (project / "examples/job.py").write_text("print('nested')\n")
    spec, _, _ = cli.load_cluster(project / "inference.yaml", project, script)
    assert spec.kwargs == {"script": script}
    assert spec.env["PYTHONPATH"] == "/tmp/hfdask-project:/tmp/hfdask-project/examples"
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode("".join(spec.bootstrap[5:]))),
                      mode="r:gz") as archive:
        assert "examples/job.py" in archive.getnames()


def test_reject_symlinked_script_directory(project):
    (project / "examples").symlink_to(project, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        cli.package_project(project, "examples/job.py", [])


def test_reject_symlink(project):
    (project / "link").symlink_to(project / "job.py")
    with pytest.raises(ValueError, match="symlinks"):
        cli.package_project(project, "job.py", [])


def test_missing_lock(project):
    (project / "uv.lock").unlink()
    with pytest.raises(ValueError, match="uv lock"):
        cli.package_project(project, "job.py", [])


def test_require_locked_runner_dependency(project):
    (project / "pyproject.toml").write_text('[project]\nname="demo"\ndependencies=[]\n')
    with pytest.raises(ValueError, match=r"hfdask\[p2p\]"):
        cli.package_project(project, "job.py", [])


def test_archive_limit(project, monkeypatch):
    monkeypatch.setattr(cli, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="512 KiB"):
        cli.package_project(project, "job.py", [])


@pytest.mark.parametrize("replacement", ["count: true", "count: 64", "count: 0"])
def test_invalid_worker_count(project, replacement):
    path = project / "inference.yaml"
    path.write_text(path.read_text().replace("count: 4", replacement))
    with pytest.raises(ValueError, match="workers.count"):
        cli.load_cluster(path, project, "job.py")


def test_mount_read_only_requires_boolean(project):
    path = project / "inference.yaml"
    path.write_text(path.read_text().replace("read_only: true", 'read_only: "true"'))
    with pytest.raises(TypeError, match="mount.read_only must be a boolean"):
        cli.load_cluster(path, project, "job.py")


def test_explicit_relay_consent(project):
    path = project / "inference.yaml"
    path.write_text(path.read_text().replace("public_relays: true", "public_relays: false"))
    with pytest.raises(ValueError, match="Explicitly"):
        cli.load_cluster(path, project, "job.py")


def test_bootstrap_forwards_appended_mesh_arguments(project, monkeypatch, capsys):
    # Execute only the generated bootstrap with all subprocess/runtime operations mocked.
    (project / "large.lock").write_bytes(random.Random(0).randbytes(100_000))
    bootstrap = cli.package_project(project, "job.py", ["inference"])
    assert int(bootstrap[4]) > 1
    assert all(len(chunk) <= 32 * 1024 for chunk in bootstrap[5:])
    assert sum(len(arg) + 1 for arg in bootstrap) < 2 * 1024 * 1024
    monkeypatch.setattr("sys.argv", ["-c", *bootstrap[3:], "python", "-m", "hfdask.runner",
                                   "hfdask.runner:run_script", "--mesh", "{}", "--node", "2"])
    import os
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/uv")
    sync = MagicMock()
    execute = MagicMock()
    monkeypatch.setattr(subprocess, "run", sync)
    monkeypatch.setattr(os, "execv", execute)
    destination = project / "extracted"
    code = bootstrap[2].replace('/tmp/hfdask-project', str(destination))
    # Execute our generated bootstrap to verify its mocked runtime contract.
    exec(compile(code, "<bootstrap>", "exec"), {})  # noqa: S102
    sync.assert_called_once_with(
        ["/usr/bin/uv", "sync", "--locked", "--no-dev", "--extra", "inference"], check=True)
    assert execute.call_args.args[1] == [
        "/usr/bin/uv", "run", "--no-sync", "python", "-m", "hfdask.runner",
        "hfdask.runner:run_script", "--mesh", "{}", "--node", "2"]
    assert (destination / "job.py").read_text() == (project / "job.py").read_text()
    assert (destination / "large.lock").read_bytes() == (project / "large.lock").read_bytes()
    assert capsys.readouterr().out.splitlines() == [
        "hfdask: staging project source", "hfdask: syncing locked environment",
        "hfdask: starting runner"]


@pytest.mark.parametrize("failure", [None, "wait", "launch", "interrupt"])
def test_lifecycle(project, monkeypatch, failure):
    cluster = MagicMock()
    cluster.manifest.return_value = {"cluster_id": "public-id", "jobs": [{"id": "job1"}]}
    monkeypatch.setattr(Identity, "public_id", lambda self: "public")
    captured = {}

    def launch(spec, identities, **options):
        captured.update(spec=spec, identities=identities, options=options)
        options["on_submitted"](cluster)
        assert json.loads(Path("manifest.json").read_text()) == cluster.manifest()
        if failure == "launch":
            raise LaunchError(cluster)
        return cluster

    monkeypatch.setattr(cli, "submit_cluster", launch)
    if failure == "wait":
        cluster.wait.side_effect = TimeoutError("still running")
    elif failure == "interrupt":
        cluster.wait.side_effect = KeyboardInterrupt()
    result = cli.main(["run", "--cluster", "inference.yaml",
                       "--manifest", "manifest.json", "job.py"])
    assert result == (130 if failure == "interrupt" else 1 if failure else 0)
    assert len(captured["identities"]) == 5
    assert len({identity.secret for identity in captured["identities"]}) == 5
    if failure:
        cluster.close.assert_called_once()
    else:
        cluster.wait.assert_called_once_with(timeout=7200)
    assert "secret" not in Path("manifest.json").read_text()


def test_submission_propagates_environment_and_wrapper(monkeypatch):
    api = MagicMock()
    api.run_job.return_value.id = "job1"
    spec = JobSpec(namespace="example", image="stock", entrypoint="hfdask.runner:run_script",
                   workers=1, bootstrap=("wrapper",), env={"SETTING": "value"})
    submit(spec, api=api)
    assert api.run_job.call_args.kwargs["env"] == spec.env
    assert api.run_job.call_args.kwargs["command"][0] == "wrapper"
    monkeypatch.setattr(Identity, "public_id", lambda self: self.secret.hex())
    submit_cluster(spec, [Identity(b"a" * 32), Identity(b"b" * 32)], api=api,
                   public_relays=True)
    args = api.run_job.call_args.kwargs
    assert args["env"] == spec.env
    assert args["command"][0] == "wrapper"
    assert args["command"][-2:] == ["--node", "1"]


def test_interrupt_submission_retains_known_jobs(monkeypatch):
    api = MagicMock()
    api.run_job.return_value.id = "known"
    monkeypatch.setattr(Identity, "public_id", lambda self: self.secret.hex())

    def interrupt(cluster):
        raise KeyboardInterrupt()

    with pytest.raises(LaunchError) as raised:
        submit_cluster(JobSpec(namespace="example", image="stock",
                               entrypoint="module:run", workers=1),
                       [Identity(b"a" * 32), Identity(b"b" * 32)], api=api,
                       public_relays=True, on_submitted=interrupt)
    assert isinstance(raised.value.__cause__, KeyboardInterrupt)
    assert raised.value.cluster.jobs[0].id == "known"


def test_own_project_adds_p2p_extra(project):
    path = project / "pyproject.toml"
    path.write_text(path.read_text().replace('name = "demo"', 'name = "hfdask"')
                    + 'p2p = ["iroh==1.1.0"]\n')
    bootstrap = cli.package_project(project, "job.py", ["inference"])
    assert json.loads(bootstrap[3]) == ["inference", "p2p"]


def test_invalid_config_never_submits(project, monkeypatch):
    launch = MagicMock()
    monkeypatch.setattr(cli, "submit_cluster", launch)
    (project / "uv.lock").unlink()
    assert cli.main(["run", "--cluster", "inference.yaml", "job.py"]) == 1
    launch.assert_not_called()


def test_bootstrap_rejects_traversal(project, monkeypatch):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        entry = tarfile.TarInfo("../escaped")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    monkeypatch.setattr("sys.argv", ["-c", "[]", "1",
                                   base64.b64encode(payload.getvalue()).decode()])
    code = cli._BOOTSTRAP.replace("/tmp/hfdask-project", str(project / "extracted"))
    with pytest.raises(SystemExit, match="Unsafe"):
        # Execute our bootstrap to exercise archive traversal rejection.
        exec(compile(code, "<bootstrap>", "exec"), {})  # noqa: S102
    assert not (project / "escaped").exists()


def test_archive_exceeds_512_kib(project):
    (project / "large.lock").write_bytes(random.Random(0).randbytes(513 * 1024))
    with pytest.raises(ValueError, match="512 KiB"):
        cli.package_project(project, "job.py", [])


@pytest.mark.parametrize("arguments, message", [
    (["[]", "0"], "chunk count"),
    (["[]", "23"], "chunk count"),
    (["[]", "2", "AAAA"], "chunk count"),
    (["[]", "1", "A" * (32 * 1024 + 1)], "chunk exceeds"),
])
def test_bootstrap_rejects_invalid_chunks(monkeypatch, arguments, message):
    monkeypatch.setattr("sys.argv", ["-c", *arguments])
    with pytest.raises(SystemExit, match=message):
        # Execute our bootstrap to exercise malformed argv rejection.
        exec(compile(cli._BOOTSTRAP, "<bootstrap>", "exec"), {})  # noqa: S102


def test_manifest_is_public_and_atomic(tmp_path):
    cluster = Cluster("public", [])
    path = tmp_path / "manifest.json"
    cli.save_manifest(path, cluster)
    assert json.loads(path.read_text()) == {"cluster_id": "public", "jobs": []}
    assert list(tmp_path.iterdir()) == [path]
