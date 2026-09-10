"""Category eligibility without consuming artificial Dask resource slots."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from distributed import Client, Future
from pydantic import TypeAdapter

from .config import CONFIG, PlacementConfig

_tag_collection = TypeAdapter(tuple[str, ...], config=CONFIG)


def require_tag_collection(tags: Iterable[str]) -> set[str]:
    if isinstance(tags, str):
        raise TypeError("tags must be a collection of strings, not a string")
    return set(_tag_collection.validate_python(tuple(tags)))


def workers_with(client: Client, *, tags: Iterable[str]) -> list[str]:
    """Snapshot workers whose hfdask metadata contains every requested tag.

    Args:
        client: Connected Dask client with access to scheduler worker metadata.
        tags: Required categorical tags, such as `["HAS_GPU"]` or custom group tags.
            An empty collection matches all currently registered workers.

    Returns:
        Sorted worker addresses. Refresh the snapshot after workers join or leave.

    Raises:
        TypeError: If tags is a single string rather than a collection.
        ValueError: If tags fail validation or no workers match; there is no fallback.
    """
    required = require_tag_collection(tags)
    inventories = {
        address: worker.get("hfdask", {})
        for address, worker in client.scheduler_info()["workers"].items()
    }
    eligible = sorted(
        address for address, info in inventories.items() if required.issubset(info.get("tags", []))
    )
    return require_matching_workers(eligible, required)


def require_matching_workers(eligible: list[str], required: set[str]) -> list[str]:
    if not eligible:
        raise ValueError(f"No workers match tags: {sorted(required)}")
    return eligible


def submit_on(
    client: Client,
    function: Callable[..., Any],
    *args: object,
    tags: Iterable[str],
    resources: dict[str, float] | None = None,
    **kwargs: object,
) -> Future:
    """Submit with hard categorical affinity plus native numeric reservations.

    Args:
        client: Connected Dask client.
        function: Task callable to execute remotely.
        *args: Positional arguments forwarded to the task through `client.submit`.
        tags: Required worker tags, resolved by `workers_with` at submission time.
        resources: Native Dask resource requirements, such as `{"GPU": 1}`.
            Tags select workers; numeric resources reserve their capacity.
        **kwargs: Additional `client.submit` options and task keyword arguments.
            `workers` and `allow_other_workers` are owned by this helper.

    Returns:
        A Dask future restricted to the matching worker addresses.

    Raises:
        ValueError: If placement is overridden, tags are invalid, or no workers match.
        TypeError: If tags is a single string.
    """
    PlacementConfig(kwargs=kwargs)
    return client.submit(
        function,
        *args,
        workers=workers_with(client, tags=tags),
        allow_other_workers=False,
        resources=resources,
        **kwargs,
    )
