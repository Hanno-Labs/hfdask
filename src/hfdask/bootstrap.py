"""Stage and launch a locked project using only the image's standard library."""

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

SOURCE = Path("/tmp/hfdask-source/project.tar.gz")
ROOT = Path("/tmp/hfdask-project")
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_FILES = 2000


def require_verified_payload(payload: bytes, checksum: str) -> None:
    if len(payload) > MAX_ARCHIVE_BYTES:
        raise SystemExit("hfdask compressed source exceeds 8 MiB")
    if hashlib.sha256(payload).hexdigest() != checksum:
        raise SystemExit("hfdask source SHA256 checksum mismatch")


def require_extraction_budget(file_count: int, total: int) -> None:
    if file_count > MAX_FILES or total > MAX_SOURCE_BYTES:
        raise SystemExit("hfdask source archive exceeds extraction limits")


def require_safe_archive_member(member: tarfile.TarInfo,
                                names: set[PurePosixPath]) -> PurePosixPath:
    path = PurePosixPath(member.name)
    if (not member.isfile() or path.is_absolute() or ".." in path.parts
            or not path.parts or member.size < 0 or path in names):
        raise SystemExit("Unsafe hfdask source archive")
    return path


def main() -> None:
    extras = json.loads(sys.argv[1])
    with SOURCE.open("rb") as source:
        payload = source.read(MAX_ARCHIVE_BYTES + 1)
    require_verified_payload(payload, sys.argv[2])
    print("hfdask: staging project source", flush=True)
    ROOT.mkdir(mode=0o700, parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members: list[tarfile.TarInfo] = []
        total = 0
        names: set[PurePosixPath] = set()
        for member in archive:
            total += member.size
            require_extraction_budget(len(members) + 1, total)
            path = require_safe_archive_member(member, names)
            names.add(path)
            members.append(member)
        for member in members:
            target = ROOT.joinpath(*PurePosixPath(member.name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise SystemExit("Unsafe hfdask source archive")
            with extracted, target.open("xb") as output:
                shutil.copyfileobj(extracted, output)
            target.chmod(member.mode & 0o777)
    os.chdir(ROOT)
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("environment.image must contain uv and Python")
    print("hfdask: syncing locked environment", flush=True)
    extra_args = [arg for extra in extras for arg in ("--extra", extra)]
    subprocess.run([uv, "sync", "--locked", "--no-dev", *extra_args], check=True)
    print("hfdask: starting runner", flush=True)
    os.execv(uv, [uv, "run", "--no-sync", *sys.argv[3:]])


if __name__ == "__main__":
    main()
