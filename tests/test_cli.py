import hashlib
import io
import json
import random
import subprocess
import tarfile
from importlib.resources import files
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from hfdask import bootstrap, cli
from hfdask.cluster import Cluster, LaunchError, submit_cluster
from hfdask.jobs import JobSpec, submit


@pytest.fixture
def project(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "0.1.0"\n'
        'dependencies = ["dask[distributed]"]\n'
        '[dependency-groups]\nrunner = ["hfdask==0.1.0"]\n'
        'inference = [{ include-group = "runner" }]\n'
    )
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "job.py").write_text("from distributed import Client\nclient = Client()\n")
    (tmp_path / "inference.yaml").write_text(
        "namespace: example\ncoordinator:\n  flavor: cpu-basic\n"
        "workers:\n  flavor: h200\n  count: 4\nenvironment:\n"
        "  image: ghcr.io/astral-sh/uv:python3.12-bookworm-slim\n"
        "  groups: [inference]\ntimeout: 2h\n"
        "mounts:\n  - source: hf://buckets/example/data\n"
        "    target: /data\n    read_only: true\n"
    )
    monkeypatch.chdir(tmp_path)
    api = MagicMock()
    api.bucket_info.return_value.private = True
    monkeypatch.setattr(cli, "HfApi", lambda: api)
    return tmp_path


def test_yaml_contract(project):
    spec, options, timeout, groups = cli.load_cluster(project / "inference.yaml", project, "job.py")
    assert spec.bootstrap == ()
    spec = cli.prepare_spec(spec, project, "job.py", groups, cli.HfApi())
    assert spec.workers == 4
    assert spec.flavor == "h200"
    assert spec.entrypoint == "hfdask.runner:run_script"
    assert spec.kwargs == {"script": "job.py"}
    assert spec.volumes[0].source == "example/data"
    assert spec.volumes[0].read_only is True
    assert options == {
        "scheduler_worker": False,
        "scheduler_flavor": "cpu-basic",
    }
    assert timeout == 7200
    assert spec.env["DASK_DISTRIBUTED__WORKER__DAEMON"] == "False"
    assert spec.bootstrap[:2] == ("python3", "/tmp/hfdask-source/bootstrap.py")
    assert spec.env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert json.loads(spec.bootstrap[2]) == ["inference"]
    assert len(spec.bootstrap) == 4
    assert len(spec.bootstrap[3]) == 64
    assert spec.command()[len(spec.bootstrap) :][:4] == [
        "python",
        "-m",
        "hfdask.runner",
        "hfdask.runner:run_script",
    ]


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("count", [1, 63])
def test_yaml_coordinator_worker(project, worker, count):
    path = project / "inference.yaml"
    path.write_text(
        path.read_text()
        .replace("  flavor: cpu-basic", f"  flavor: cpu-basic\n  worker: {str(worker).lower()}")
        .replace("  count: 4", f"  count: {count}")
    )
    spec, options, _, _ = cli.load_cluster(path, project, "job.py")
    assert options["scheduler_worker"] is worker
    assert spec.workers == count + int(worker)
    assert spec.flavor == "h200"
    assert options["scheduler_flavor"] == "cpu-basic"


@pytest.mark.parametrize("worker", ['"true"', '"false"', "1", "0", "null", "[]", "{}"])
def test_coordinator_worker_requires_boolean(project, worker):
    path = project / "inference.yaml"
    path.write_text(
        path.read_text().replace("  flavor: cpu-basic", f"  flavor: cpu-basic\n  worker: {worker}")
    )
    with pytest.raises(ValidationError, match=r"coordinator\.worker"):
        cli.load_cluster(path, project, "job.py")


def test_archive_selection(project):
    (project / ".gitignore").write_text("ignored.txt\ntracked-ignored.txt\n")
    for name in ("ignored.txt", "tracked-ignored.txt", ".env", "private.pem", "mise.local.toml"):
        (project / name).write_text("secret")
    subprocess.run(["git", "add", "-f", "tracked-ignored.txt", ".env"], check=True)
    payload = cli.package_project(project, "job.py", ["inference"])
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        names = archive.getnames()
        assert {"job.py", "uv.lock", "pyproject.toml"} <= set(names)
        assert not set(names) & {
            "ignored.txt",
            "tracked-ignored.txt",
            ".env",
            "private.pem",
            "mise.local.toml",
        }


@pytest.mark.parametrize(
    "name",
    [
        "node-0.env",
        "config/node-worker.env",
        "secrets/node-key",
        ".secrets/node-key",
        "config/secrets/node-key",
        "config/.secrets/node-key",
    ],
)
@pytest.mark.parametrize("tracked", [False, True])
def test_archive_excludes_node_keys_and_secret_directories(project, name, tracked):
    path = project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("HFDASK_NODE_KEY=private-key-material\n")
    if tracked:
        subprocess.run(["git", "add", "-f", "--", name], check=True)
    payload = cli.package_project(project, "job.py", ["inference"])
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
    spec, _, _, groups = cli.load_cluster(project / "inference.yaml", project, script)
    assert spec.kwargs == {"script": script}
    assert spec.env["PYTHONPATH"] == "/tmp/hfdask-project:/tmp/hfdask-project/examples"
    with tarfile.open(
        fileobj=io.BytesIO(cli.package_project(project, script, groups)), mode="r:gz"
    ) as archive:
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
    (project / "pyproject.toml").write_text(
        '[project]\nname="demo"\ndependencies=[]\n[dependency-groups]\ndeploy=["hfdask"]\n'
    )
    with pytest.raises(ValueError, match="Add hfdask"):
        cli.package_project(project, "job.py", [])


def test_require_declared_dependency_group(project):
    with pytest.raises(ValueError, match="undeclared dependency group"):
        cli.package_project(project, "job.py", ["missing"])


def test_archive_limit(project, monkeypatch):
    monkeypatch.setattr(cli, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(ValueError, match="8 MiB"):
        cli.package_project(project, "job.py", ["inference"])


@pytest.mark.parametrize("replacement", ["count: true", "count: 64", "count: 0"])
def test_invalid_worker_count(project, replacement):
    path = project / "inference.yaml"
    path.write_text(path.read_text().replace("count: 4", replacement))
    with pytest.raises(ValueError, match="workers.count"):
        cli.load_cluster(path, project, "job.py")


def test_mount_read_only_requires_boolean(project):
    path = project / "inference.yaml"
    path.write_text(path.read_text().replace("read_only: true", 'read_only: "true"'))
    with pytest.raises(ValidationError, match=r"mounts\.0\.read_only"):
        cli.load_cluster(path, project, "job.py")


@pytest.mark.parametrize("kind", ["models", "datasets", "spaces"])
@pytest.mark.parametrize("revision", [None, "main", "release/v1", "a" * 40])
def test_repository_mounts(project, kind, revision):
    path = project / "inference.yaml"
    config = path.read_text().replace(
        "hf://buckets/example/data", f"hf://{kind}/example/repo/input/nested"
    )
    config = config.replace("    read_only: true\n", "")
    if revision is not None:
        config += f"    revision: {revision}\n"
    path.write_text(config)
    spec, _, _, _ = cli.load_cluster(path, project, "job.py")
    volume = spec.volumes[0]
    assert volume.type == kind[:-1]
    assert volume.source == "example/repo"
    assert volume.path == "input/nested"
    assert volume.revision == revision
    assert volume.read_only is True
    assert volume.mount_path == "/data"


@pytest.mark.parametrize("kind", ["models", "datasets", "spaces"])
def test_repositories_reject_writable_mounts(project, kind):
    path = project / "inference.yaml"
    path.write_text(
        path.read_text()
        .replace("hf://buckets/", f"hf://{kind}/")
        .replace("read_only: true", "read_only: false")
    )
    with pytest.raises(ValueError, match="must be read-only"):
        cli.load_cluster(path, project, "job.py")


@pytest.mark.parametrize("revision", ['""', "null", "123", "true"])
def test_repository_revision_must_be_nonempty_text(project, revision):
    path = project / "inference.yaml"
    path.write_text(
        path.read_text().replace("hf://buckets/", "hf://models/") + f"    revision: {revision}\n"
    )
    with pytest.raises(ValidationError, match=r"mounts\.0(?:\.revision)?"):
        cli.load_cluster(path, project, "job.py")


def test_bucket_revision_is_rejected(project):
    path = project / "inference.yaml"
    path.write_text(path.read_text() + "    revision: main\n")
    with pytest.raises(ValueError, match="not buckets"):
        cli.load_cluster(path, project, "job.py")


@pytest.mark.parametrize(
    "source", ["hf://unknown/org/repo", "hf://models/org", "hf://datasets/org/repo/../data"]
)
def test_invalid_mount_source(project, source):
    path = project / "inference.yaml"
    path.write_text(path.read_text().replace("hf://buckets/example/data", source))
    with pytest.raises(ValueError, match="mount.source"):
        cli.load_cluster(path, project, "job.py")


def test_example_pins_input_mounts():
    root = Path(__file__).parents[1]
    spec, _, _, _ = cli.load_cluster(root / "examples/inference.yaml", root, "examples/job.py")
    model, dataset, output = spec.volumes
    assert (model.type, model.source, model.mount_path, model.revision) == (
        "model",
        "Qwen/Qwen3-0.6B",
        "/model",
        "c1899de289a04d12100db370d81485cdf75e47ca",
    )
    assert (dataset.type, dataset.source, dataset.mount_path, dataset.revision) == (
        "dataset",
        "fancyzhx/ag_news",
        "/dataset",
        "eb185aade064a813bc0b7f42de02595523103ca4",
    )
    assert model.read_only and dataset.read_only
    assert output.type == "bucket" and output.read_only is False


def test_cpu_example_is_unannotated_dataframe_cluster():
    root = Path(__file__).parents[1]
    spec, options, _, groups = cli.load_cluster(root / "examples/cpu.yaml", root, "examples/cpu.py")
    assert spec.flavor == "cpu-basic"
    assert spec.workers == 2
    assert not spec.volumes
    assert options["scheduler_worker"] is True
    assert groups == ["deploy"]
    source = (root / "examples/cpu.py").read_text()
    assert "dask.dataframe" in source
    assert "dask.annotate" not in source


def test_removed_network_configuration_is_rejected(project):
    path = project / "inference.yaml"
    path.write_text(path.read_text() + "network:\n  public_relays: true\n")
    with pytest.raises(ValueError, match="network"):
        cli.load_cluster(path, project, "job.py")


def test_bootstrap_forwards_appended_cluster_arguments(project, monkeypatch, capsys):

    (project / "large.lock").write_bytes(random.Random(0).randbytes(100_000))
    spec, _, _, groups = cli.load_cluster(project / "inference.yaml", project, "job.py")
    api = cli.HfApi()
    command = cli.prepare_spec(spec, project, "job.py", groups, api).bootstrap
    payload, _ = api.batch_bucket_files.call_args.kwargs["add"][0]
    mounted = project / "project.tar.gz"
    mounted.write_bytes(payload)
    assert len(payload) > 32 * 1024
    assert sum(len(arg) + 1 for arg in command) < 8192
    monkeypatch.setattr(
        "sys.argv",
        [
            *command[1:],
            "python",
            "-m",
            "hfdask.runner",
            "hfdask.runner:run_script",
            "--cluster",
            "{}",
            "--node",
            "2",
        ],
    )
    import os
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/uv")
    sync = MagicMock()
    execute = MagicMock()
    monkeypatch.setattr(subprocess, "run", sync)
    monkeypatch.setattr(os, "execv", execute)
    destination = project / "extracted"
    monkeypatch.setattr(bootstrap, "SOURCE", mounted)
    monkeypatch.setattr(bootstrap, "ROOT", destination)
    bootstrap.main()
    sync.assert_called_once_with(
        ["/usr/bin/uv", "sync", "--locked", "--no-dev", "--group", "inference"], check=True
    )
    assert execute.call_args.args[1] == [
        "/usr/bin/uv",
        "run",
        "--no-sync",
        "python",
        "-m",
        "hfdask.runner",
        "hfdask.runner:run_script",
        "--cluster",
        "{}",
        "--node",
        "2",
    ]
    assert (destination / "job.py").read_text() == (project / "job.py").read_text()
    assert (destination / "large.lock").read_bytes() == (project / "large.lock").read_bytes()
    assert capsys.readouterr().out.splitlines() == [
        "hfdask: staging project source",
        "hfdask: syncing locked environment",
        "hfdask: starting runner",
    ]


@pytest.mark.parametrize("failure", [None, "wait", "launch", "interrupt"])
def test_lifecycle(project, monkeypatch, failure):
    cluster = MagicMock()
    cluster.manifest.return_value = {"cluster_id": "public-id", "jobs": [{"id": "job1"}]}
    captured = {}

    def launch(spec, **options):
        captured.update(spec=spec, options=options)
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
    result = cli.main(
        ["run", "--cluster", "inference.yaml", "--manifest", "manifest.json", "job.py"]
    )
    assert result == (130 if failure == "interrupt" else 1 if failure else 0)
    assert captured["options"]["api"] is cli.HfApi()
    assert "groups" not in captured["options"]
    cli.HfApi().batch_bucket_files.assert_called_once()
    if failure:
        cluster.close.assert_called_once()
    else:
        cluster.wait.assert_called_once_with(timeout=7200)
    assert "secret" not in Path("manifest.json").read_text()


@pytest.mark.parametrize("worker", [None, False, True])
def test_cli_coordinator_worker_submission(project, monkeypatch, worker):
    path = project / "inference.yaml"
    text = path.read_text().replace("  count: 4", "  count: 1")
    if worker is not None:
        text = text.replace(
            "  flavor: cpu-basic", f"  flavor: cpu-basic\n  worker: {str(worker).lower()}"
        )
    path.write_text(text)
    monkeypatch.setattr(Cluster, "wait", MagicMock())
    api = cli.HfApi()
    api.run_job.return_value.id = "job1"

    assert (
        cli.main(["run", "--cluster", "inference.yaml", "--manifest", "manifest.json", "job.py"])
        == 0
    )
    assert api.run_job.call_count == 2
    calls = [call.kwargs for call in api.run_job.call_args_list]
    assert [call["flavor"] for call in calls] == ["cpu-basic", "h200"]
    for node, call in enumerate(calls):
        command = call["command"]
        assert ("--scheduler-worker" in command) is (worker is True)
        assert command[command.index("--node") + 1] == str(node)
        assert command[command.index("--workers") + 1] == str(1 + int(worker is True))
        config = json.loads(command[command.index("--cluster") + 1])
        assert config["job_nodes"] == 2
        assert config["node_flavors"] == ["cpu-basic", "h200"]
        assert config["schema"] == 2


def test_submission_propagates_environment_and_wrapper(monkeypatch):
    api = MagicMock()
    api.run_job.return_value.id = "job1"
    spec = JobSpec(
        namespace="example",
        image="stock",
        entrypoint="hfdask.runner:run_script",
        workers=1,
        bootstrap=("wrapper",),
        env={"SETTING": "value"},
    )
    submit(spec, api=api)
    assert api.run_job.call_args.kwargs["env"] == spec.env
    assert api.run_job.call_args.kwargs["command"][0] == "wrapper"
    submit_cluster(spec, api=api)
    args = api.run_job.call_args.kwargs
    assert args["env"] == spec.env
    assert args["command"][0] == "wrapper"
    assert args["command"][-2:] == ["--node", "1"]


def test_interrupt_submission_retains_known_jobs(monkeypatch):
    api = MagicMock()
    api.run_job.return_value.id = "known"

    def interrupt(cluster):
        raise KeyboardInterrupt()

    with pytest.raises(LaunchError) as raised:
        submit_cluster(
            JobSpec(namespace="example", image="stock", entrypoint="module:run", workers=1),
            api=api,
            on_submitted=interrupt,
        )
    assert isinstance(raised.value.__cause__, KeyboardInterrupt)
    assert raised.value.cluster.jobs[0].id == "known"


def test_own_project_needs_no_runner_dependency(project):
    path = project / "pyproject.toml"
    path.write_text(path.read_text().replace('name = "demo"', 'name = "hfdask"'))
    assert isinstance(cli.package_project(project, "job.py", []), bytes)


@pytest.mark.parametrize("dependency", ["hfdask", "hfdask>=0.1", "hfdask[inference]==0.1.0"])
def test_base_runner_dependency(project, dependency):
    path = project / "pyproject.toml"
    path.write_text(f'[project]\nname="demo"\ndependencies=["{dependency}"]\n')
    assert isinstance(cli.package_project(project, "job.py", []), bytes)


def test_invalid_config_never_submits(project, monkeypatch):
    launch = MagicMock()
    monkeypatch.setattr(cli, "submit_cluster", launch)
    (project / "uv.lock").unlink()
    assert cli.main(["run", "--cluster", "inference.yaml", "job.py"]) == 1
    launch.assert_not_called()
    cli.HfApi().create_bucket.assert_not_called()
    cli.HfApi().batch_bucket_files.assert_not_called()


def test_bootstrap_rejects_traversal(project, monkeypatch):
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:gz") as archive:
        entry = tarfile.TarInfo("../escaped")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    mounted = project / "project.tar.gz"
    mounted.write_bytes(payload.getvalue())
    monkeypatch.setattr(
        "sys.argv", ["bootstrap.py", "[]", hashlib.sha256(payload.getvalue()).hexdigest()]
    )
    monkeypatch.setattr(bootstrap, "SOURCE", mounted)
    monkeypatch.setattr(bootstrap, "ROOT", project / "extracted")
    with pytest.raises(SystemExit, match="Unsafe"):
        bootstrap.main()
    assert not (project / "escaped").exists()


def test_archive_above_old_inline_limit(project):
    (project / "large.lock").write_bytes(random.Random(0).randbytes(513 * 1024))
    assert len(cli.package_project(project, "job.py", ["inference"])) > 512 * 1024


@pytest.mark.parametrize("limit", ["MAX_SOURCE_BYTES", "MAX_FILES"])
def test_source_limits(project, monkeypatch, limit):
    monkeypatch.setattr(cli, limit, 1)
    with pytest.raises(ValueError, match="8 MiB/2000 files"):
        cli.package_project(project, "job.py", [])


@pytest.mark.parametrize("kind", ["checksum", "invalid", "compressed", "files", "size", "symlink"])
def test_bootstrap_rejects_bad_source(project, monkeypatch, kind):
    payload = b"not an archive"
    if kind in {"files", "size", "symlink"}:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            for index in range(2001 if kind == "files" else 1):
                member = tarfile.TarInfo(str(index))
                if kind == "size":
                    member.size = 8388609
                    archive.addfile(member, io.BytesIO(b"x" * member.size))
                else:
                    if kind == "symlink":
                        member.type = tarfile.SYMTYPE
                        member.linkname = "/etc/passwd"
                    archive.addfile(member)
        payload = buffer.getvalue()
    elif kind == "compressed":
        payload = b"x" * 8388609
    mounted = project / "project.tar.gz"
    mounted.write_bytes(payload)
    checksum = "0" * 64 if kind == "checksum" else hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr("sys.argv", ["bootstrap.py", "[]", checksum])
    destination = project / "extracted"
    monkeypatch.setattr(bootstrap, "SOURCE", mounted)
    monkeypatch.setattr(bootstrap, "ROOT", destination)
    expected = tarfile.ReadError if kind == "invalid" else SystemExit
    with pytest.raises(expected):
        bootstrap.main()
    assert not destination.exists() or not list(destination.iterdir())


def test_staging_uses_private_bucket_multipart_and_unique_prefix(project, capsys):
    spec, _, _, groups = cli.load_cluster(project / "inference.yaml", project, "job.py")
    api = cli.HfApi()
    staged = cli.prepare_spec(spec, project, "job.py", groups, api)
    api.create_bucket.assert_called_once_with("example/jobs-artifacts", private=True, exist_ok=True)
    api.bucket_info.assert_called_once_with("example/jobs-artifacts")
    assert [call[0] for call in api.method_calls] == [
        "create_bucket",
        "bucket_info",
        "batch_bucket_files",
    ]
    (payload, remote_path), (bootstrap_bytes, bootstrap_path) = (
        api.batch_bucket_files.call_args.kwargs["add"]
    )
    assert api.batch_bucket_files.call_args.args == ("example/jobs-artifacts",)
    assert isinstance(payload, bytes)
    volume = staged.volumes[-1]
    assert volume.source == "example/jobs-artifacts"
    assert remote_path == f"{volume.path}/project.tar.gz"
    assert bootstrap_path == f"{volume.path}/bootstrap.py"
    assert bootstrap_bytes == files("hfdask").joinpath("bootstrap.py").read_bytes()
    assert volume.path.startswith("hfdask-source/")
    assert volume.mount_path == "/tmp/hfdask-source"
    assert volume.read_only is True
    assert staged.bootstrap == (
        "python3",
        "/tmp/hfdask-source/bootstrap.py",
        json.dumps(groups),
        hashlib.sha256(payload).hexdigest(),
    )
    assert staged.env == spec.env
    assert spec.bootstrap == ()
    again = cli.prepare_spec(spec, project, "job.py", groups, api)
    assert again.volumes[-1].path != volume.path
    log = capsys.readouterr().err
    assert f"hf://buckets/example/jobs-artifacts/{remote_path}" in log
    assert "storage charges" in log


@pytest.mark.parametrize("privacy", [False, None, "true", 1])
def test_staging_fails_closed_for_unverified_privacy(project, monkeypatch, privacy):
    api = cli.HfApi()
    api.bucket_info.return_value.private = privacy
    launch = MagicMock()
    monkeypatch.setattr(cli, "submit_cluster", launch)
    assert cli.main(["run", "--cluster", "inference.yaml", "job.py"]) == 1
    api.batch_bucket_files.assert_not_called()
    launch.assert_not_called()


@pytest.mark.parametrize("operation", ["create_bucket", "bucket_info", "batch_bucket_files"])
def test_staging_api_failure_never_submits(project, monkeypatch, operation):
    api = cli.HfApi()
    getattr(api, operation).side_effect = RuntimeError("staging failed")
    launch = MagicMock()
    monkeypatch.setattr(cli, "submit_cluster", launch)
    assert cli.main(["run", "--cluster", "inference.yaml", "job.py"]) == 1
    launch.assert_not_called()
    if operation != "batch_bucket_files":
        api.batch_bucket_files.assert_not_called()


def test_mount_prefix_is_separate_from_bucket(project):
    path = project / "inference.yaml"
    path.write_text(path.read_text().replace("example/data", "example/data/input/nested"))
    spec, _, _, _ = cli.load_cluster(path, project, "job.py")
    assert spec.volumes[0].source == "example/data"
    assert spec.volumes[0].path == "input/nested"


@pytest.mark.parametrize(
    "target", ["/tmp", "/tmp/hfdask-source", "/tmp/hfdask-source/nested", "/tmp/hfdask-project"]
)
def test_mount_cannot_overlap_bootstrap_paths(project, target):
    path = project / "inference.yaml"
    path.write_text(path.read_text().replace("target: /data", f"target: {target}"))
    with pytest.raises(ValueError, match="must not overlap"):
        cli.load_cluster(path, project, "job.py")


def test_submission_cause_logging_is_preserved(project, monkeypatch, capsys):
    cluster = MagicMock()
    cluster.manifest.return_value = {"cluster_id": "public", "jobs": []}

    def launch(*args, **kwargs):
        raise LaunchError(cluster) from RuntimeError("HF413 body too large")

    monkeypatch.setattr(cli, "submit_cluster", launch)
    assert cli.main(["run", "--cluster", "inference.yaml", "job.py"]) == 1
    assert "Submission cause: HF413 body too large" in capsys.readouterr().err
    cluster.close.assert_called_once()


def test_invalid_group_never_uploads(project):
    spec, _, _, _ = cli.load_cluster(project / "inference.yaml", project, "job.py")
    with pytest.raises(ValueError, match="undeclared"):
        cli.prepare_spec(spec, project, "job.py", ["missing"], cli.HfApi())
    cli.HfApi().create_bucket.assert_not_called()


def test_manifest_is_public_and_atomic(tmp_path):
    cluster = Cluster("public", [])
    path = tmp_path / "manifest.json"
    cli.save_manifest(path, cluster)
    assert json.loads(path.read_text()) == {"cluster_id": "public", "jobs": []}
    assert list(tmp_path.iterdir()) == [path]
