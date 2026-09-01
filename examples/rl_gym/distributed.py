import jax
import jax.numpy as jp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec


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
    return jax.sharding.Mesh(np.array(devices), ("shards",))


def shard_experience(mesh, experience):
    sharding = NamedSharding(mesh, PartitionSpec("shards"))
    num_shards = mesh.devices.size

    def globalize(local):
        shape = (num_shards * local.shape[0],) + local.shape[1:]
        return jax.make_array_from_single_device_arrays(shape, sharding, [local])  # fmt: skip

    return jax.tree.map(globalize, experience)


def global_mean(mesh, value):
    # average a per-process scalar so curricula stay synchronized; the
    # result comes back as a local value, ready for per-process use
    sharding = NamedSharding(mesh, PartitionSpec("shards"))
    shape = (mesh.devices.size,)
    args = shape, sharding, [value.reshape(1)]
    values = jax.make_array_from_single_device_arrays(*args)
    return localize(compute_mean(values))


compute_mean = jax.jit(jp.mean)


def localize(values):
    # local view of replicated values, so the per-process rollout does not
    # pull the process-local simulation into the global update program
    return jax.tree.map(lambda value: value.addressable_data(0), values)


def replicate(mesh, values):
    sharding = NamedSharding(mesh, PartitionSpec())

    def globalize(local):
        args = local.shape, sharding, [local]
        return jax.make_array_from_single_device_arrays(*args)

    return jax.tree.map(globalize, values)
