import jax
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, PartitionSpec


def initialize(coordinator, num_processes, process_id):
    # one process per GPU: warp physics cannot shard across devices, so
    # each process simulates locally and only the update is global
    args = coordinator, num_processes, process_id
    jax.distributed.initialize(*args, local_device_ids=[process_id])


def build_mesh():
    if jax.process_count() == 1:
        devices = jax.local_devices()[:1]
    else:
        devices = jax.devices()
    return Mesh(np.array(devices), ("shards",))


def shard(mesh, values):
    # each process contributes its own rollout as one shard
    spec = PartitionSpec("shards")
    return multihost_utils.host_local_array_to_global_array(values, mesh, spec)


def replicate(mesh, values):
    spec = PartitionSpec()
    return multihost_utils.host_local_array_to_global_array(values, mesh, spec)


def localize(values):
    # the local copy of replicated values, so the per-process rollout does
    # not pull the process-local simulation into the global update program
    return jax.tree.map(lambda value: value.addressable_data(0), values)


def global_mean(value):
    # average a per-process scalar so the curricula stay synchronized
    return float(np.mean(multihost_utils.process_allgather(value)))
