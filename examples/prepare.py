"""Generate disposable node identities for the three-job synthetic example."""

import argparse
import json
import os
from pathlib import Path

import iroh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--allow-public-relays", action="store_true", required=True)
    parser.add_argument("--scheduler-worker", action="store_true")
    parser.add_argument("--flavors", nargs="+", help="Scheduler flavor followed by worker flavors")
    args = parser.parse_args()
    if args.flavors and len(args.flavors) < 2:
        parser.error("--flavors requires a scheduler and at least one remote worker")
    args.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    count = len(args.flavors) if args.flavors else 3 - int(args.scheduler_worker)
    keys = [iroh.SecretKey.generate() for _ in range(count)]
    config = {"peers": [key.public().to_bytes().hex() for key in keys],
              "relays": [], "public_relays": True, "startup_timeout": 300}
    if args.flavors:
        config.update(hardware_detection=True, node_flavors=args.flavors,
                      node_tags=[[] for _ in keys])
    (args.directory / "mesh.json").write_text(json.dumps(config))
    for index, key in enumerate(keys):
        path = args.directory / f"node-{index}.env"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"HFDASK_NODE_KEY={key.to_bytes().hex()}\n")
    print(f"Created public mesh.json and {len(keys)} private node secret files; do not upload secrets.")


if __name__ == "__main__":
    main()
