"""Category eligibility without consuming artificial Dask resource slots."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from pydantic import TypeAdapter

from .config import CONFIG, PlacementConfig

_tag_collection = TypeAdapter(tuple[str, ...], config=CONFIG)


def require_tag_collection(tags: Iterable[str]) -> set[str]:
    if isinstance(tags, str):
        raise TypeError("tags must be a collection of strings, not a string")
    return set(_tag_collection.validate_python(tuple(tags)))


def workers_with(client: Any, *, tags: Iterable[str]) -> list[str]:
    """Snapshot eligible workers. Refresh after scaling; never falls back silently."""
    required = require_tag_collection(tags)
    inventories = {address: worker.get("hfdask", {}) for address, worker
                   in client.scheduler_info()["workers"].items()}
    eligible = sorted(address for address, info in inventories.items()
                      if required.issubset(info.get("tags", [])))
    return require_matching_workers(eligible, required)


def require_matching_workers(eligible: list[str], required: set[str]) -> list[str]:
    if not eligible:
        raise ValueError(f"No workers match tags: {sorted(required)}")
    return eligible


def submit_on(client: Any, function: Callable[..., Any], *args: Any,
              tags: Iterable[str], resources: dict[str, float] | None = None,
              **kwargs: Any) -> Any:
    """Submit with hard categorical affinity plus native numeric reservations."""
    PlacementConfig(kwargs=kwargs)
    return client.submit(function, *args, workers=workers_with(client, tags=tags),
                         allow_other_workers=False, resources=resources, **kwargs)
