"""CPU-only Dask DataFrame example with no hfdask-specific task annotations."""

import json

import dask.dataframe as dd
import pandas as pd
from distributed import get_client


def calculate(rows: int = 1_000_000) -> dict[str, int]:
    client = get_client()
    workers = len(client.scheduler_info()["workers"])
    partitions = max(1, workers * 4)
    frame = dd.from_pandas(pd.DataFrame({"value": range(rows)}), npartitions=partitions)
    total = int(frame.assign(doubled=frame["value"] * 2)["doubled"].sum().compute())
    return {"rows": rows, "workers": workers, "partitions": partitions, "sum": total}


def main() -> None:
    print(json.dumps(calculate()), flush=True)


if __name__ == "__main__":
    main()
