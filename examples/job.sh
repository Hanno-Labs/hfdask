#!/bin/sh
# Mount the repository at /source (read-only) and a durable bucket at /output.
set -eu
node="$1"
shift
mkdir -p /tmp/hfdask-example
cp -R /source/. /tmp/hfdask-example/
cd /tmp/hfdask-example
pip install --disable-pip-version-check --no-cache-dir '.[p2p]'
export PYTHONPATH=/tmp/hfdask-example/examples
exec python -u -m hfdask.runner workload:proof --workers 2 \
  --threads-per-worker 1 --memory-limit 1GiB \
  --mesh "$(cat /source/mesh.json)" --node "$node" "$@"
