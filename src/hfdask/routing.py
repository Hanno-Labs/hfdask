"""Category eligibility without consuming artificial Dask resource slots."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def workers_with(client: Any, *, tags: Iterable[str]) -> list[str]:
    """Snapshot eligible workers. Refresh after scaling; never falls back silently."""
    required = set(tags)
    if isinstance(tags, str):
        raise TypeError("tags must be a collection of strings, not a string")
    inventories = {address: worker.get("hfdask", {}) for address, worker
                   in client.scheduler_info()["workers"].items()}
    eligible = sorted(address for address, info in inventories.items()
                      if required.issubset(info.get("tags", [])))
    if not eligible:
        raise ValueError(f"No workers match tags: {sorted(required)}")
    return eligible


def submit_on(client: Any, function: Callable[..., Any], *args: Any,
              tags: Iterable[str], resources: dict[str, float] | None = None,
              **kwargs: Any) -> Any:
    """Submit with hard categorical affinity plus native numeric reservations."""
    if "workers" in kwargs or "allow_other_workers" in kwargs:
        raise ValueError("submit_on owns worker placement; use client.submit for custom placement")
    return client.submit(function, *args, workers=workers_with(client, tags=tags),
                         allow_other_workers=False, resources=resources, **kwargs)
