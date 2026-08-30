from collections import namedtuple
from pathlib import Path

import jax
import numpy as np

Checkpoint = namedtuple("Checkpoint", "stdv, optimizer_state, learning_rate, iteration, max_speed, normalizer_leaves")  # fmt: skip


def save(directory, iteration, actor, critic, training, max_speed, normalizers):  # fmt: skip
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assign_variables(actor, training.parameters.actor)
    assign_variables(critic, training.parameters.critic)
    actor.save(directory / f"actor_{iteration:06d}.keras")
    critic.save(directory / f"critic_{iteration:06d}.keras")
    arrays = pack_arrays(training, iteration, max_speed)
    for slot, leaf in enumerate(jax.tree.leaves(normalizers)):
        arrays[f"normalizer_{slot}"] = np.asarray(leaf)
    np.savez(directory / f"training_{iteration:06d}.npz", **arrays)


def restore_normalizers(loaded, normalizers):
    if not loaded.normalizer_leaves:
        return normalizers
    leaves = [np.asarray(leaf) for leaf in loaded.normalizer_leaves]
    return jax.tree.unflatten(jax.tree.structure(normalizers), leaves)


def load(directory, actor, critic):
    directory = Path(directory)
    iteration = find_latest_iteration(directory)
    actor.load_weights(directory / f"actor_{iteration:06d}.keras")
    critic.load_weights(directory / f"critic_{iteration:06d}.keras")
    arrays = np.load(directory / f"training_{iteration:06d}.npz")
    return unpack_arrays(arrays, iteration)


def find_latest_iteration(directory):
    iterations = []
    for path in directory.glob("training_*.npz"):
        iterations.append(int(path.stem.split("_")[-1]))
    if not iterations:
        raise FileNotFoundError(f"no training_*.npz checkpoints in {directory}")  # fmt: skip
    return max(iterations)


def assign_variables(model, values):
    for variable, value in zip(model.trainable_variables, values):
        variable.assign(np.asarray(value))


def pack_arrays(training, iteration, max_speed):
    arrays = {"stdv": np.asarray(training.parameters.stdv)}
    arrays["learning_rate"] = np.asarray(training.learning_rate)
    arrays["iteration"] = np.asarray(iteration)
    arrays["max_speed"] = np.asarray(max_speed)
    for slot, value in enumerate(training.optimizer_state):
        arrays[f"optimizer_{slot}"] = np.asarray(value)
    return arrays


def unpack_arrays(arrays, iteration):
    stdv = arrays["stdv"]
    learning_rate = float(arrays["learning_rate"])
    max_speed = float(arrays["max_speed"])
    optimizer_state = []
    while f"optimizer_{len(optimizer_state)}" in arrays:
        optimizer_state.append(arrays[f"optimizer_{len(optimizer_state)}"])
    normalizer_leaves = []
    while f"normalizer_{len(normalizer_leaves)}" in arrays:
        normalizer_leaves.append(arrays[f"normalizer_{len(normalizer_leaves)}"])  # fmt: skip
    args = stdv, optimizer_state, learning_rate, iteration, max_speed
    return Checkpoint(*args, normalizer_leaves)
