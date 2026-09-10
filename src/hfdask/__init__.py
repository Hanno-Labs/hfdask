"""Run Dask workloads on Hugging Face Jobs.

`JobSpec` describes a workload and `submit` launches it on one paid HF Job.
Multi-machine and persistent clusters use `hfdask.cluster`; the CLI ships a
locked Git project and runs an ordinary Python script from a YAML definition.

## Module reference

- [Jobs](hfdask/jobs.html) — workload specifications and single-Job lifecycle
- [Clusters](hfdask/cluster.html) — worker groups, identities, launch, and recovery
- [Clients](hfdask/client.html) — persistent Dask connections over the encrypted mesh
- [Routing](hfdask/routing.html) — categorical worker affinity and resource reservations
- [Configuration](hfdask/config.html) — strict input schemas and validation contracts
- [CLI](hfdask/cli.html) — source packaging, staging, and script submission
- [Runner](hfdask/runner.html) — in-Job workload execution
- [Hardware](hfdask/hardware.html) — detected worker profiles and resources
- [Network](hfdask/network.html) — authenticated transport and loopback proxies

## Lifecycle and safety

Submission reserves paid capacity. Keep recovery manifests until termination
is verified; a local timeout or client disconnect does not release remote Jobs.
HF credentials stay with the submitter. Only run trusted workloads and grant
mounts the minimum access they need.
"""

from .cluster import WorkerGroup
from .jobs import Job, JobFailed, JobSpec, submit

__all__ = ["Job", "JobFailed", "JobSpec", "WorkerGroup", "submit"]
