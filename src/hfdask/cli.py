"""YAML-driven batch submission of a locked, Git-selected source project."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
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
from .config import MOUNT_SOURCE, ClusterConfig, PackageConfig, PackagePlan, ProjectConfig
from .jobs import JobSpec

# Bound both upload size and extracted working-tree size.
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_FILES = 2000


def _secret_path(path: PurePosixPath) -> bool:
    return any(
        part
        in {
            ".git",
            ".venv",
            ".ssh",
            ".aws",
            ".gnupg",
            ".hfdask",
            "mise.local.toml",
            ".mise.local.toml",
            ".netrc",
            ".pypirc",
            "credentials",
            "credentials.json",
            "id_rsa",
            "id_ed25519",
            "secrets",
            ".secrets",
        }
        or part.startswith((".env", "secrets."))
        or part.endswith((".env", ".pem", ".key", ".p12", ".pfx"))
        for part in path.parts
    )


def require_safe_project_file(root: Path, name: str) -> Path | None:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe project path: {name}")
    if _secret_path(relative):
        return None
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"Project symlinks are not supported: {name}; exclude with .gitignore")
    if not candidate.exists():  # Tracked files may be deleted in the working tree.
        return None
    if not stat.S_ISREG(candidate.stat().st_mode):
        raise ValueError(f"Project entry is not a regular file: {name}")
    return candidate


def require_source_budget(total: int, file_count: int) -> None:
    if total > MAX_SOURCE_BYTES or file_count > MAX_FILES:
        raise ValueError(
            "Project source exceeds 8 MiB/2000 files; move data to bucket mounts "
            "and exclude generated files with .gitignore"
        )


def require_project_inputs(files: dict[str, bytes], script: str) -> None:
    for required in ("pyproject.toml", "uv.lock", script):
        if required not in files:
            raise ValueError(
                f"Required project file {required} is missing or excluded; "
                "run uv lock and check .gitignore"
            )


def require_upload_budget(payload: bytes) -> bytes:
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise ValueError(
            "Compressed source exceeds 8 MiB upload limit; move data to bucket "
            "mounts and exclude generated files with .gitignore"
        )
    return payload


def package_project(root: Path, script: str, extras: list[str]) -> bytes:
    """Snapshot working-tree files, not HEAD; never follow project symlinks."""
    inputs = PackageConfig(script=script, extras=extras)
    script = inputs.script
    try:
        selected = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", "."],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        # --exclude-standard only filters untracked files; also exclude ignored tracked files.
        ignored = set(
            subprocess.run(
                ["git", "ls-files", "--cached", "--ignored", "--exclude-standard", "-z", "--", "."],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout.split(b"\0")
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("Run from a Git project containing pyproject.toml and uv.lock") from error
    files: dict[str, bytes] = {}
    modes: dict[str, int] = {}
    total = 0
    for raw in sorted(set(selected) - ignored - {b""}):
        name = os.fsdecode(raw)
        candidate = require_safe_project_file(root, name)
        if candidate is None:
            continue
        metadata = candidate.stat()
        total += metadata.st_size
        require_source_budget(total, len(files) + 1)
        files[name] = candidate.read_bytes()
        modes[name] = 0o755 if metadata.st_mode & 0o111 else 0o644
    require_project_inputs(files, script)
    project = tomllib.loads(files["pyproject.toml"].decode()).get("project", {})
    PackagePlan(script=script, extras=inputs.extras, project=ProjectConfig.model_validate(project))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            member.mode = modes[name]
            archive.addfile(member, io.BytesIO(data))
    return require_upload_budget(buffer.getvalue())


def require_private_source_bucket(api: HfApi, bucket_id: str) -> None:
    if api.bucket_info(bucket_id).private is not True:
        raise ValueError(
            f"Source bucket {bucket_id} must be verified private; "
            "refusing upload without changing visibility"
        )


def prepare_spec(spec: JobSpec, root: Path, script: str, extras: list[str], api: HfApi) -> JobSpec:
    """Validate and stage source once; retained artifacts incur storage until deleted."""
    payload = package_project(root, script, extras)
    bucket_id = f"{spec.namespace}/jobs-artifacts"
    folder = f"hfdask-source/{uuid4()}"
    remote_path = f"{folder}/project.tar.gz"
    api.create_bucket(bucket_id, private=True, exist_ok=True)
    require_private_source_bucket(api, bucket_id)
    bootstrap = files("hfdask").joinpath("bootstrap.py").read_bytes()
    api.batch_bucket_files(
        bucket_id,
        add=[
            (payload, remote_path),
            (bootstrap, f"{folder}/bootstrap.py"),
        ],
    )
    print(
        f"Source artifact: hf://buckets/{bucket_id}/{remote_path} "
        "(retained after run; storage charges may apply until manually deleted)",
        file=sys.stderr,
    )
    return replace(
        spec,
        bootstrap=(
            "python3",
            "/tmp/hfdask-source/bootstrap.py",
            json.dumps(extras),
            hashlib.sha256(payload).hexdigest(),
        ),
        volumes=[
            *spec.volumes,
            Volume(
                type="bucket",
                source=bucket_id,
                path=folder,
                mount_path="/tmp/hfdask-source",
                read_only=True,
            ),
        ],
    )


def load_cluster(
    path: Path,
    root: Path,
    script: str,
) -> tuple[JobSpec, dict[str, Any], float, list[str]]:
    # Load lazily so non-CLI library users do not need PyYAML at import time.
    yaml = import_module("yaml")

    config = ClusterConfig.model_validate(yaml.safe_load(path.read_text()))
    scheduler_worker = config.coordinator.worker
    volumes: list[Volume] = []
    for mount in config.mounts:
        location = MOUNT_SOURCE.fullmatch(mount.source)
        assert location is not None  # Validated by MountConfig; conversion only below.
        kind, namespace, name, subfolder = location.groups()
        volumes.append(
            Volume(
                type=kind[:-1],
                source=f"{namespace}/{name}",
                revision=mount.revision,
                path=subfolder or "",
                mount_path=mount.target,
                read_only=mount.read_only,
            )
        )
    spec = JobSpec(
        namespace=config.namespace,
        image=config.environment.image,
        entrypoint="hfdask.runner:run_script",
        kwargs={"script": script},
        # YAML counts remote workers; JobSpec includes the colocated worker.
        workers=config.workers.count + int(scheduler_worker),
        flavor=config.workers.flavor,
        timeout=config.timeout,
        volumes=volumes,
        env={
            "PYTHONPATH": "/tmp/hfdask-project:"
            + str(PurePosixPath("/tmp/hfdask-project") / PurePosixPath(script).parent),
            "DASK_DISTRIBUTED__WORKER__DAEMON": "False",
            "DASK_DISTRIBUTED__WORKER__MULTIPROCESSING_METHOD": "spawn",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        },
    )
    options: dict[str, Any] = {
        "public_relays": True,
        "scheduler_worker": scheduler_worker,
        "scheduler_flavor": config.coordinator.flavor,
    }
    return spec, options, config.timeout_seconds, config.environment.extras


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
    run.add_argument(
        "--manifest", type=Path, help="Public recovery manifest (default: .hfdask/run-*.json)"
    )
    run.add_argument("script")
    args = parser.parse_args(argv)
    cluster: Cluster | None = None
    manifest: Path = (
        args.manifest
        if args.manifest is not None
        else Path(".hfdask") / f"run-{secrets.token_hex(8)}.json"
    )
    try:
        spec, options, timeout, extras = load_cluster(args.cluster, Path.cwd(), args.script)
        nodes = spec.workers + 1 - int(options["scheduler_worker"])
        identities = [Identity(secrets.token_bytes(32)) for _ in range(nodes)]
        # Validate transport identities before reserving the manifest or submitting.
        for identity in identities:
            identity.public_id()

        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("x") as output:
            output.write("{}\n")
        print(f"Recovery manifest: {manifest}", file=sys.stderr)
        api = HfApi()
        spec = prepare_spec(spec, Path.cwd(), args.script, extras, api)
        try:
            cluster = submit_cluster(
                spec, identities, api=api, **options, on_submitted=partial(save_manifest, manifest)
            )
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
                print(
                    f"Manifest write failed: {save_error}; known jobs: "
                    f"{json.dumps(cluster.manifest())}",
                    file=sys.stderr,
                )
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
