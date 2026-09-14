"""Strict input concepts and focused cross-field contracts, without execution side effects."""

from __future__ import annotations

import ipaddress
import re
from pathlib import PurePosixPath
from typing import Annotated, Any, Self, TypeVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

CONFIG = ConfigDict(strict=True, extra="forbid", validate_default=True, hide_input_in_errors=True)
Text = Annotated[str, Field(pattern=r"\S")]
PositiveCount = Annotated[int, Field(strict=True, ge=1)]
NonnegativeCount = Annotated[int, Field(strict=True, ge=0)]
OneThread = Annotated[int, Field(strict=True, ge=1, le=1)]
NodeIndex = Annotated[int, Field(strict=True, ge=0)]
WorkerOrdinal = Annotated[int, Field(strict=True, ge=0)]
Duration = Annotated[str, Field(pattern=r"^[1-9][0-9]*[smhd]$")]
RelayURL = Annotated[str, Field(pattern=r"^https://")]
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


def require_loopback(value: str) -> str:
    if not ipaddress.ip_address(value).is_loopback:
        raise ValueError("Dask proxies must bind only to loopback")
    return value


IdentityValue = TypeVar("IdentityValue", str, bytes)


def require_distinct_identities(value: tuple[IdentityValue, ...]) -> tuple[IdentityValue, ...]:
    if len(set(value)) != len(value):
        raise ValueError("Every job and client needs a distinct identity")
    return value


def require_distinct_public_identities(value: list[str]) -> list[str]:
    require_distinct_identities(tuple(value))
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


def require_public_relay_consent(value: bool) -> bool:
    if not value:
        raise ValueError(
            "Explicitly set network.public_relays: true to permit public discovery/relays"
        )
    return value


Entrypoint = Annotated[str, AfterValidator(require_entrypoint)]
MemoryLimit = Annotated[str, AfterValidator(require_positive_memory_limit)]
DaskMemoryLimit = Annotated[str, AfterValidator(require_dask_memory_limit)]
LoopbackAddress = Annotated[str, AfterValidator(require_loopback)]
RelativeScript = Annotated[str, AfterValidator(require_relative_script)]
PlacementKwargs = Annotated[dict[str, Any], AfterValidator(require_owned_placement)]
DistinctPeerIds = Annotated[
    tuple[bytes, ...], Field(min_length=1), AfterValidator(require_distinct_identities)
]
PublicRelayConsent = Annotated[bool, AfterValidator(require_public_relay_consent)]
PersistentMode = Annotated[bool, AfterValidator(require_persistent_cluster)]
DistinctPublicIds = Annotated[list[Text], AfterValidator(require_distinct_public_identities)]


class ConfigModel(BaseModel):
    model_config = CONFIG


class WorkerConfig(ConfigModel):
    threads: OneThread = 1
    memory_limit: MemoryLimit = "auto"


class WorkerTopologyConfig(ConfigModel):
    workers_per_node: tuple[NonnegativeCount, ...] = Field(min_length=1)


class ServiceConfig(ConfigModel):
    workers_per_node: tuple[NonnegativeCount, ...] = Field(min_length=1)
    node: NodeIndex
    ordinal: WorkerOrdinal

    @model_validator(mode="after")
    def require_node_in_roster(self) -> Self:
        if self.node >= len(self.workers_per_node):
            raise ValueError("Invalid worker service: node is outside roster")
        return self

    @model_validator(mode="after")
    def require_worker_on_node(self) -> Self:
        if self.ordinal >= self.workers_per_node[self.node]:
            raise ValueError("Invalid worker service: ordinal is outside node capacity")
        return self


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


class RelayConfig(ConfigModel):
    public_relays: bool = False
    relays: list[RelayURL] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_explicit_relay_policy(self) -> Self:
        if not self.public_relays and not self.relays:
            raise ValueError("Explicitly permit public relays or supply custom relay_urls")
        return self

    @model_validator(mode="after")
    def require_exclusive_relay_policy(self) -> Self:
        if self.public_relays and self.relays:
            raise ValueError("Choose public or custom relays, not both")
        return self


class LaunchConfig(RelayConfig):
    startup_timeout: PositiveCount = 1200
    scheduler_worker: bool = False
    scheduler_flavor: Text | None = None
    worker_flavor: Text | None = None


class ConnectionConfig(RelayConfig):
    # Public manifests may carry runner metadata not consumed by the client.
    model_config = ConfigDict(extra="ignore")
    schema_version: Annotated[int, Field(strict=True, ge=1, le=1, alias="schema")]
    persistent: PersistentMode
    job_nodes: Annotated[int, Field(strict=True, ge=1, le=64)]
    peers: DistinctPublicIds
    scheduler_worker: bool

    @model_validator(mode="after")
    def require_client_roster(self) -> Self:
        if len(self.peers) != self.job_nodes + 1:
            raise ValueError("Client identity or roster does not match")
        return self


class RunnerMeshConfig(RelayConfig):
    """Public wire configuration checked before binding an in-Job endpoint."""

    model_config = ConfigDict(extra="ignore")
    node: NodeIndex = Field(exclude=True)
    peers: DistinctPublicIds = Field(min_length=1)
    startup_timeout: PositiveCount
    hardware_detection: Annotated[bool, Field(strict=True)] = True
    node_flavors: list[Text] = Field(default_factory=list)
    node_tags: list[list[str]] = Field(default_factory=list)
    job_nodes: Annotated[int, Field(ge=1, le=64)] | None = None
    persistent: bool = False
    scheduler_worker: bool = False
    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)] | None = Field(
        default=None, alias="schema"
    )

    @property
    def nodes(self) -> int:
        return self.job_nodes if self.job_nodes is not None else len(self.peers)

    @model_validator(mode="after")
    def require_runner_node_in_roster(self) -> Self:
        if "job_nodes" in self.model_fields_set and self.job_nodes is None:
            raise ValueError("job_nodes must be a positive Job count when supplied")
        if self.node >= self.nodes or self.nodes > len(self.peers):
            raise ValueError("Runner node is outside the Job roster")
        return self

    @model_validator(mode="after")
    def require_hardware_flavor_per_job(self) -> Self:
        if len(self.node_flavors) != self.nodes:
            raise ValueError("Provide one hardware flavor per Job")
        return self

    @model_validator(mode="after")
    def require_detected_worker_topology(self) -> Self:
        if not self.hardware_detection:
            raise ValueError("Mesh clusters require hardware detection")
        return self

    @model_validator(mode="after")
    def require_tag_collection_per_job(self) -> Self:
        if "node_tags" in self.model_fields_set and len(self.node_tags) != self.nodes:
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
    public_id: str

    @model_validator(mode="after")
    def require_client_identity(self) -> Self:
        if self.public_id != self.connection.peers[-1]:
            raise ValueError("Client identity or roster does not match")
        return self


class MeshConfig(ConfigModel):
    index: NodeIndex
    base_port: Annotated[int, Field(strict=True, ge=1024, le=65535)] = 21000
    max_connections: PositiveCount = 256
    bind_host: LoopbackAddress = "127.0.0.1"
    services: tuple[NodeIndex, ...] = Field(min_length=1)
    peers: DistinctPeerIds
    endpoint_id: bytes

    @model_validator(mode="after")
    def require_service_owners_in_roster(self) -> Self:
        if any(owner >= len(self.peers) for owner in self.services):
            raise ValueError("Every service must belong to a roster node")
        return self

    @model_validator(mode="after")
    def require_endpoint_index_in_roster(self) -> Self:
        if self.index >= len(self.peers):
            raise ValueError("Invalid index in peer roster")
        return self

    @model_validator(mode="after")
    def require_available_peer_port_range(self) -> Self:
        if self.base_port > 65535 - len(self.services):
            raise ValueError("Invalid peer-port range")
        return self

    @model_validator(mode="after")
    def require_endpoint_identity(self) -> Self:
        if self.endpoint_id != self.peers[self.index]:
            raise ValueError("Roster must match this endpoint identity")
        return self


class CoordinatorConfig(ConfigModel):
    flavor: Text = "cpu-basic"
    worker: bool = False


class WorkersConfig(ConfigModel):
    flavor: Text = "cpu-basic"
    count: Annotated[int, Field(strict=True, ge=1, le=63)] = 1


class EnvironmentConfig(ConfigModel):
    image: Text
    groups: list[GroupName] = Field(default_factory=list)


class NetworkConfig(ConfigModel):
    public_relays: PublicRelayConsent = False


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
    network: NetworkConfig
    timeout: Duration = "1h"
    mounts: list[MountConfig] = Field(default_factory=list)

    @property
    def timeout_seconds(self) -> float:
        return float(
            int(self.timeout[:-1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[self.timeout[-1]]
        )
