"""Strict input concepts and focused cross-field contracts, without execution side effects."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Any, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

CONFIG = ConfigDict(strict=True, extra="forbid", validate_default=True, hide_input_in_errors=True)
Text = Annotated[str, Field(pattern=r"\S")]
PositiveCount = Annotated[int, Field(strict=True, ge=1)]
OneThread = Annotated[int, Field(strict=True, ge=1, le=1)]
NodeIndex = Annotated[int, Field(strict=True, ge=0)]
Duration = Annotated[str, Field(pattern=r"^[1-9][0-9]*[smhd]$")]
GroupName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
_ENTRYPOINT = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*\Z")


def require_entrypoint(value: str) -> str:
    if not _ENTRYPOINT.fullmatch(value):
        raise ValueError("entrypoint must be module:function")
    return value


def require_positive_memory_limit(value: str) -> str:
    from dask.utils import parse_bytes

    if value != "auto" and int(parse_bytes(value)) <= 0:
        raise ValueError("memory_limit must be positive or auto")
    return value


def require_dask_memory_limit(value: str) -> str:
    from dask.utils import parse_bytes

    # Local Dask explicitly supports zero to disable its memory limit.
    if value != "auto" and int(parse_bytes(value)) < 0:
        raise ValueError("memory_limit must be nonnegative or auto")
    return value


def require_owned_placement(value: dict[str, Any]) -> dict[str, Any]:
    if {"workers", "allow_other_workers"}.intersection(value):
        raise ValueError("submit_on owns worker placement; use client.submit for custom placement")
    return value


def require_relative_script(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".py":
        raise ValueError("script must be a project-relative .py path without '..'")
    return path.as_posix()


def require_persistent_cluster(value: bool) -> bool:
    if not value:
        raise ValueError("A persistent cluster manifest is required")
    return value


Entrypoint = Annotated[str, AfterValidator(require_entrypoint)]
MemoryLimit = Annotated[str, AfterValidator(require_positive_memory_limit)]
DaskMemoryLimit = Annotated[str, AfterValidator(require_dask_memory_limit)]
RelativeScript = Annotated[str, AfterValidator(require_relative_script)]
PlacementKwargs = Annotated[dict[str, Any], AfterValidator(require_owned_placement)]
PersistentMode = Annotated[bool, AfterValidator(require_persistent_cluster)]


class ConfigModel(BaseModel):
    model_config = CONFIG


class WorkerConfig(ConfigModel):
    threads: OneThread = 1
    memory_limit: MemoryLimit = "auto"


class PlacementConfig(ConfigModel):
    kwargs: PlacementKwargs


class PackageConfig(ConfigModel):
    script: RelativeScript
    groups: list[GroupName]


class ProjectConfig(ConfigModel):
    name: str = ""
    dependencies: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="ignore")


class PackagePlan(PackageConfig):
    project: ProjectConfig
    dependency_groups: dict[str, list[str | dict[str, str]]] = Field(
        default_factory=dict, alias="dependency-groups"
    )

    @model_validator(mode="after")
    def require_locked_runner_dependency(self) -> Self:
        dependencies = list(self.project.dependencies)
        pending = list(self.groups)
        selected: set[str] = set()
        while pending:
            group = pending.pop()
            if group in selected:
                continue
            if group not in self.dependency_groups:
                raise ValueError(
                    f"environment.groups contains undeclared dependency group: {group}"
                )
            selected.add(group)
            for item in self.dependency_groups[group]:
                if isinstance(item, str):
                    dependencies.append(item)
                elif included := item.get("include-group"):
                    pending.append(included)
        if self.project.name.lower().replace("_", "-") != "hfdask" and not any(
            re.match(r"(?i)^hfdask\s*(?:\[|[<>=!~@;]|$)", dep) for dep in dependencies
        ):
            raise ValueError(
                "Add hfdask to project dependencies or a selected dependency group and "
                "regenerate uv.lock; the runner must use the same locked environment"
            )
        return self


class RunConfig(ConfigModel):
    entrypoint: Entrypoint
    workers: PositiveCount = 2
    threads_per_worker: OneThread = 1
    memory_limit: DaskMemoryLimit = "auto"


class PersistentWorkloadConfig(ConfigModel):
    """Persistent Jobs serve clients rather than running an embedded workload."""

    entrypoint: Annotated[str, Field(max_length=0)] = ""
    kwargs: Annotated[dict[str, Any], Field(max_length=0)] = Field(default_factory=dict)


class WaitConfig(ConfigModel):
    timeout: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    poll_interval: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class LaunchConfig(ConfigModel):
    startup_timeout: PositiveCount = 1200
    scheduler_worker: bool = False
    scheduler_flavor: Text | None = None
    worker_flavor: Text | None = None


class ConnectionConfig(ConfigModel):
    # Public manifests may carry runner metadata not consumed by the client.
    model_config = ConfigDict(extra="ignore")
    schema_version: Annotated[int, Field(strict=True, ge=2, le=2, alias="schema")]
    persistent: PersistentMode
    job_nodes: Annotated[int, Field(strict=True, ge=1, le=64)]
    scheduler_worker: bool


class RunnerClusterConfig(ConfigModel):
    """Public topology configuration checked before starting in-Job Dask services."""

    model_config = ConfigDict(extra="ignore")
    node: NodeIndex = Field(exclude=True)
    startup_timeout: PositiveCount
    node_flavors: list[Text] = Field(default_factory=list)
    node_tags: list[list[str]] = Field(default_factory=list)
    job_nodes: Annotated[int, Field(ge=1, le=64)]
    persistent: bool = False
    scheduler_worker: bool = False
    schema_version: Annotated[int, Field(strict=True, ge=2, le=2)] | None = Field(
        default=None, alias="schema"
    )

    @model_validator(mode="after")
    def require_runner_node_in_roster(self) -> Self:
        if self.node >= self.job_nodes:
            raise ValueError("Runner node is outside the Job roster")
        return self

    @model_validator(mode="after")
    def require_hardware_flavor_per_job(self) -> Self:
        if len(self.node_flavors) != self.job_nodes:
            raise ValueError("Provide one hardware flavor per Job")
        return self

    @model_validator(mode="after")
    def require_tag_collection_per_job(self) -> Self:
        if len(self.node_tags) != self.job_nodes:
            raise ValueError("Provide one tag collection per Job")
        return self

    @model_validator(mode="after")
    def require_persistent_connection_metadata(self) -> Self:
        if self.persistent:
            ConnectionConfig.model_validate(self.model_dump(by_alias=True))
        return self


class ConnectConfig(ConfigModel):
    timeout: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    connection: ConnectionConfig
    local_port: Annotated[int, Field(strict=True, ge=1024, le=65535)]


class CoordinatorConfig(ConfigModel):
    flavor: Text = "cpu-basic"
    worker: bool = False


class WorkersConfig(ConfigModel):
    flavor: Text = "cpu-basic"
    count: Annotated[int, Field(strict=True, ge=1, le=63)] = 1


class EnvironmentConfig(ConfigModel):
    image: Text
    groups: list[GroupName] = Field(default_factory=list)


MOUNT_SOURCE = re.compile(r"hf://(buckets|models|datasets|spaces)/([^/]+)/([^/]+)(?:/(.*))?\Z")


def require_mount_source(value: str) -> str:
    location = MOUNT_SOURCE.fullmatch(value)
    if location is None:
        raise ValueError(
            "mount.source must be hf://{buckets,models,datasets,spaces}/namespace/name[/prefix]"
        )
    subfolder = location[4] or ""
    if ".." in PurePosixPath(subfolder).parts or subfolder.startswith("/"):
        raise ValueError("mount.source prefix must be relative without '..'")
    return value


def require_safe_mount_target(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or path == PurePosixPath("/"):
        raise ValueError("mount.target must be an absolute non-root path without '..'")
    for reserved in ("/tmp/hfdask-project", "/tmp/hfdask-source"):
        reserved_path = PurePosixPath(reserved)
        if path.is_relative_to(reserved_path) or reserved_path.is_relative_to(path):
            raise ValueError(f"mount.target must not overlap {reserved}")
    return value


MountSource = Annotated[Text, AfterValidator(require_mount_source)]
MountTarget = Annotated[Text, AfterValidator(require_safe_mount_target)]


class MountConfig(ConfigModel):
    source: MountSource
    target: MountTarget
    read_only: bool = True
    revision: Text | None = None

    @model_validator(mode="after")
    def require_repository_revision(self) -> Self:
        if "revision" in self.model_fields_set:
            if self.source.startswith("hf://buckets/"):
                raise ValueError("mount.revision is only supported for repositories, not buckets")
            if self.revision is None:
                raise ValueError("mount.revision must be a nonempty string")
        return self

    @model_validator(mode="after")
    def require_read_only_repository(self) -> Self:
        if not self.source.startswith("hf://buckets/") and not self.read_only:
            raise ValueError("Model, dataset, and Space mounts must be read-only")
        return self


class ClusterConfig(ConfigModel):
    namespace: Text
    coordinator: CoordinatorConfig = Field(default_factory=CoordinatorConfig)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    environment: EnvironmentConfig
    timeout: Duration = "1h"
    mounts: list[MountConfig] = Field(default_factory=list)

    @property
    def timeout_seconds(self) -> float:
        return float(
            int(self.timeout[:-1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[self.timeout[-1]]
        )
