"""YAML-driven batch submission of a locked, Git-selected source project."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from dataclasses import replace
from functools import partial
from importlib import import_module
from importlib.resources import files
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from huggingface_hub import HfApi, Volume

from .cluster import Cluster, Identity, LaunchError, submit_cluster
from .jobs import JobSpec

# Bound both upload size and extracted working-tree size.
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_FILES = 2000



def _mapping(value: Any, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {', '.join(sorted(unknown))}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _secret_path(path: PurePosixPath) -> bool:
    return any(
        part in {".git", ".venv", ".ssh", ".aws", ".gnupg", ".hfdask", "mise.local.toml",
                 ".mise.local.toml", ".netrc", ".pypirc", "credentials", "credentials.json",
                 "id_rsa", "id_ed25519", "secrets", ".secrets"}
        or part.startswith((".env", "secrets."))
        or part.endswith((".env", ".pem", ".key", ".p12", ".pfx"))
        for part in path.parts
    )


def package_project(root: Path, script: str, extras: list[str]) -> bytes:
    """Snapshot working-tree files, not HEAD; never follow project symlinks."""
    path = PurePosixPath(script)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise ValueError("script must be a project-relative .py path without '..'")
    script = path.as_posix()
    try:
        selected = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "."],
            cwd=root, check=True, capture_output=True,
        ).stdout.split(b"\0")
        # --exclude-standard only filters untracked files; also exclude ignored tracked files.
        ignored = set(subprocess.run(
            ["git", "ls-files", "--cached", "--ignored", "--exclude-standard", "-z", "--", "."],
            cwd=root, check=True, capture_output=True,
        ).stdout.split(b"\0"))
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Run from a Git project containing pyproject.toml and uv.lock") from error
    files: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    total = 0
    for raw in sorted(set(selected) - ignored - {b""}):
        name = os.fsdecode(raw)
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe project path: {name}")
        if _secret_path(relative):
            continue
        candidate = root
        for part in relative.parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError(f"Project symlinks are not supported: {name}; "
                                 "exclude with .gitignore")
        if not candidate.exists():  # Tracked files may be deleted in the working tree.
            continue
        metadata = candidate.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Project entry is not a regular file: {name}")
        total += metadata.st_size
        if total > MAX_SOURCE_BYTES or len(files) >= MAX_FILES:
            raise ValueError("Project source exceeds 8 MiB/2000 files; move data to bucket mounts "
                             "and exclude generated files with .gitignore")
        files[name] = candidate.read_bytes()
        modes[name] = 0o755 if metadata.st_mode & 0o111 else 0o644
    for required in ("pyproject.toml", "uv.lock", script):
        if required not in files:
            raise ValueError(f"Required project file {required} is missing or excluded; "
                             "run uv lock and check .gitignore")
    project = tomllib.loads(files["pyproject.toml"].decode()).get("project", {})
    optional = project.get("optional-dependencies", {})
    for extra in extras:
        if extra not in optional:
            raise ValueError(f"environment.extras contains undeclared project extra: {extra}")
    dependencies = list(project.get("dependencies", []))
    for extra in extras:
        dependencies.extend(optional[extra])
    own_project = project.get("name", "").lower().replace("_", "-") == "hfdask"
    if not own_project and not any(
        re.match(r"(?i)^hfdask\s*(?:\[|[<>=!~@;]|$)", dep) for dep in dependencies
    ):
        raise ValueError("Add hfdask to project dependencies (uv add hfdask) "
                         "and regenerate uv.lock; the runner must use the same locked environment")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = modes[name]
            archive.addfile(member, io.BytesIO(data))
    if buffer.tell() > MAX_ARCHIVE_BYTES:
        raise ValueError("Compressed source exceeds 8 MiB upload limit; move data to bucket "
                         "mounts and exclude generated files with .gitignore")
    return buffer.getvalue()


def prepare_spec(spec: JobSpec, root: Path, script: str, extras: list[str], api: HfApi) -> JobSpec:
    """Validate and stage source once; retained artifacts incur storage until deleted."""
    payload = package_project(root, script, extras)
    bucket_id = f"{spec.namespace}/jobs-artifacts"
    folder = f"hfdask-source/{uuid4()}"
    remote_path = f"{folder}/project.tar.gz"
    api.create_bucket(bucket_id, private=True, exist_ok=True)
    if api.bucket_info(bucket_id).private is not True:
        raise ValueError(f"Source bucket {bucket_id} must be verified private; "
                         "refusing upload without changing visibility")
    bootstrap = files("hfdask").joinpath("bootstrap.py").read_bytes()
    api.batch_bucket_files(bucket_id, add=[
        (payload, remote_path), (bootstrap, f"{folder}/bootstrap.py"),
    ])
    print(f"Source artifact: hf://buckets/{bucket_id}/{remote_path} "
          "(retained after run; storage charges may apply until manually deleted)", file=sys.stderr)
    return replace(spec, bootstrap=("python3", "/tmp/hfdask-source/bootstrap.py", json.dumps(extras),
                                    hashlib.sha256(payload).hexdigest()),
                   volumes=[*spec.volumes, Volume(type="bucket", source=bucket_id, path=folder,
                            mount_path="/tmp/hfdask-source", read_only=True)])


def load_cluster(
    path: Path, root: Path, script: str,
) -> tuple[JobSpec, dict[str, Any], float, list[str]]:
    # Load lazily so non-CLI library users do not need PyYAML at import time.
    yaml = import_module("yaml")

    config = _mapping(yaml.safe_load(path.read_text()), "cluster",
                      {"namespace", "coordinator", "workers", "environment", "timeout",
                       "mounts", "network"})
    coordinator = _mapping(config.get("coordinator", {}), "coordinator", {"flavor", "worker"})
    scheduler_worker = coordinator.get("worker", False)
    if type(scheduler_worker) is not bool:
        raise TypeError("coordinator.worker must be a boolean")
    workers = _mapping(config.get("workers", {}), "workers", {"flavor", "count"})
    environment = _mapping(config.get("environment", {}), "environment", {"image", "extras"})
    network = _mapping(config.get("network", {}), "network", {"public_relays"})
    if network.get("public_relays") is not True:
        raise ValueError("Explicitly set network.public_relays: true "
                         "to permit public discovery/relays")
    count = workers.get("count", 1)
    if type(count) is not int or not 1 <= count <= 63:
        raise ValueError("workers.count must be an integer between 1 and 63")
    extras = environment.get("extras", [])
    if not isinstance(extras, list) or any(
        not isinstance(extra, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", extra)
        for extra in extras
    ):
        raise ValueError("environment.extras must be a list of extra names")
    timeout = _text(config.get("timeout", "1h"), "timeout")
    duration = re.fullmatch(r"([1-9][0-9]*)(s|m|h|d)", timeout)
    if duration is None:
        raise ValueError("timeout must be a positive duration such as 30m or 1h")
    seconds = int(duration[1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[duration[2]]
    mounts = config.get("mounts", [])
    if not isinstance(mounts, list):
        raise TypeError("mounts must be a list")
    volumes: list[Volume] = []
    for item in mounts:
        mount = _mapping(item, "mount", {"source", "target", "read_only", "revision"})
        source = _text(mount.get("source"), "mount.source")
        target = _text(mount.get("target"), "mount.target")
        location = re.fullmatch(
            r"hf://(buckets|models|datasets|spaces)/([^/]+)/([^/]+)(?:/(.*))?", source,
        )
        if location is None:
            raise ValueError("mount.source must be hf://{buckets,models,datasets,spaces}/"
                             "namespace/name[/prefix]")
        kind, namespace, name, subfolder = location.groups()
        subfolder = subfolder or ""
        target_path = PurePosixPath(target)
        if (not target_path.is_absolute() or ".." in target_path.parts
                or target_path == PurePosixPath("/")):
            raise ValueError("mount.target must be an absolute non-root path without '..'")
        for reserved in ("/tmp/hfdask-project", "/tmp/hfdask-source"):
            reserved_path = PurePosixPath(reserved)
            if (target_path.is_relative_to(reserved_path)
                    or reserved_path.is_relative_to(target_path)):
                raise ValueError(f"mount.target must not overlap {reserved}")
        read_only = mount.get("read_only", True)
        if type(read_only) is not bool:
            raise TypeError("mount.read_only must be a boolean")
        revision = None
        if "revision" in mount:
            if kind == "buckets":
                raise ValueError("mount.revision is only supported for repositories, not buckets")
            revision = _text(mount["revision"], "mount.revision")
        if kind != "buckets" and not read_only:
            raise ValueError("Model, dataset, and Space mounts must be read-only")
        if ".." in PurePosixPath(subfolder).parts or subfolder.startswith("/"):
            raise ValueError("mount.source prefix must be relative without '..'")
        volumes.append(Volume(type=kind[:-1], source=f"{namespace}/{name}", revision=revision,
                              path=subfolder, mount_path=target, read_only=read_only))
    spec = JobSpec(
        namespace=_text(config.get("namespace"), "namespace"),
        image=_text(environment.get("image"), "environment.image (must contain uv and Python)"),
        entrypoint="hfdask.runner:run_script", kwargs={"script": script},
        # YAML counts remote workers; JobSpec includes the colocated worker.
        workers=count + int(scheduler_worker),
        flavor=_text(workers.get("flavor", "cpu-basic"), "workers.flavor"),
        timeout=timeout, volumes=volumes,
        env={"PYTHONPATH": "/tmp/hfdask-project:"
                          + str(PurePosixPath("/tmp/hfdask-project") / PurePosixPath(script).parent),
             "DASK_DISTRIBUTED__WORKER__DAEMON": "False",
             "DASK_DISTRIBUTED__WORKER__MULTIPROCESSING_METHOD": "spawn",
             "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
    )
    options: dict[str, Any] = {"public_relays": True, "scheduler_worker": scheduler_worker,
               "scheduler_flavor": _text(coordinator.get("flavor", "cpu-basic"),
                                         "coordinator.flavor")}
    return spec, options, float(seconds), extras


def save_manifest(path: Path, cluster: Cluster) -> None:
    """Atomic replacement; manifests contain only public recovery handles."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            json.dump(cluster.manifest(), output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hfdask")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run a project-relative Dask script on a YAML cluster")
    run.add_argument("--cluster", type=Path, required=True)
    run.add_argument("--manifest", type=Path,
                     help="Public recovery manifest (default: .hfdask/run-*.json)")
    run.add_argument("script")
    args = parser.parse_args(argv)
    cluster: Cluster | None = None
    manifest: Path = (args.manifest if args.manifest is not None
                      else Path(".hfdask") / f"run-{secrets.token_hex(8)}.json")
    try:
        spec, options, timeout, extras = load_cluster(args.cluster, Path.cwd(), args.script)
        nodes = spec.workers + 1 - int(options["scheduler_worker"])
        identities = [Identity(secrets.token_bytes(32)) for _ in range(nodes)]
        # Validate transport identities before reserving the manifest or submitting.
        for identity in identities:
            identity.public_id()

        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("x") as output:
            output.write('{}\n')
        print(f"Recovery manifest: {manifest}", file=sys.stderr)
        api = HfApi()
        spec = prepare_spec(spec, Path.cwd(), args.script, extras, api)
        try:
            cluster = submit_cluster(spec, identities, api=api, **options,
                                     on_submitted=partial(save_manifest, manifest))
        except LaunchError as error:
            cluster = error.cluster
            raise
        cluster.wait(timeout=timeout)
        return 0
    # CLI boundary: report failures and release known jobs before returning an exit code.
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001
        print(f"hfdask: {error or 'interrupted'}", file=sys.stderr)
        if isinstance(error, LaunchError) and error.__cause__ is not None:
            print(f"Submission cause: {error.__cause__}", file=sys.stderr)
        if cluster is not None:
            try:
                save_manifest(manifest, cluster)
            # A manifest failure must not prevent cleanup; print recovery handles instead.
            except Exception as save_error:  # noqa: BLE001
                print(f"Manifest write failed: {save_error}; known jobs: "
                      f"{json.dumps(cluster.manifest())}", file=sys.stderr)
            try:
                cluster.close()
            # Report any cleanup failure without claiming that capacity was released.
            except Exception as close_error:  # noqa: BLE001
                print(f"Cleanup unverified: {close_error}; retain {manifest}", file=sys.stderr)
        interrupted = isinstance(error, KeyboardInterrupt) or (
            isinstance(error, LaunchError) and isinstance(error.__cause__, KeyboardInterrupt)
        )
        return 130 if interrupted else 1


if __name__ == "__main__":
    raise SystemExit(main())
