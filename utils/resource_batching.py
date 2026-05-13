"""Helpers for sizing parallel/serial client execution waves.

Flower's Ray backend can only run as many virtual clients at the same time as
can be placed with the configured per-client CPU/GPU resources.  These helpers
compute a conservative wave size by repeatedly halving the requested sampled
client count until one wave fits the available resources.  The remaining waves
run after earlier clients finish and release their resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor


@dataclass(frozen=True)
class ClientBatchPlan:
    """Resource-aware plan for one federated training round."""

    requested_clients: int
    max_parallel_clients: int
    batch_size: int
    num_batches: int

    @property
    def uses_serial_batches(self) -> bool:
        """Return whether clients must run in more than one resource wave."""

        return self.num_batches > 1


def _capacity_for_resource(
    total_resource: float, per_client_resource: float
) -> int | None:
    """Return client capacity for a single resource, or ``None`` if unconstrained."""

    if per_client_resource <= 0.0:
        return None
    if total_resource <= 0.0:
        return 0

    # Fractional Ray resources (for example 14 CPUs / 50 clients = 0.28 CPU
    # per client) can produce ratios like 49.999999999 due to floating point
    # representation.  Add a small tolerance before flooring so exact SLURM
    # splits are not under-counted by one client.
    return int(floor((total_resource / per_client_resource) + 1e-9))


def plan_client_batches(
    requested_clients: int,
    *,
    total_cpus: float,
    per_client_cpus: float,
    total_gpus: float = 0.0,
    per_client_gpus: float = 0.0,
) -> ClientBatchPlan:
    """Build a parallel-plus-serial resource plan for sampled clients.

    If all sampled clients fit at once, the plan contains one parallel batch. If
    not, the sampled-client count is divided by two repeatedly until a batch can
    fit.  The server can still wait for and aggregate all sampled client results;
    Ray executes only one fitting wave at a time and starts the next wave when
    resources are released.
    """

    requested = max(1, int(requested_clients))
    resource_capacities = [
        capacity
        for capacity in (
            _capacity_for_resource(float(total_cpus), float(per_client_cpus)),
            _capacity_for_resource(float(total_gpus), float(per_client_gpus)),
        )
        if capacity is not None
    ]

    if resource_capacities:
        max_parallel = max(1, min(resource_capacities))
    else:
        max_parallel = requested

    batch_size = requested
    while batch_size > max_parallel and batch_size > 1:
        batch_size = max(1, batch_size // 2)

    return ClientBatchPlan(
        requested_clients=requested,
        max_parallel_clients=max_parallel,
        batch_size=batch_size,
        num_batches=int(ceil(requested / batch_size)),
    )
