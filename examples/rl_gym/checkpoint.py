from pathlib import Path

import keras
import numpy as np


def save(directory, iteration, actor, critic, parameters):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    assign_variables(actor, parameters.actor)
    assign_variables(critic, parameters.critic)
    actor.save(directory / f"actor_{iteration:06d}.keras")
    critic.save(directory / f"critic_{iteration:06d}.keras")
    np.save(directory / f"stdv_{iteration:06d}.npy", np.asarray(parameters.stdv))  # fmt: skip


def load_actor(directory, iteration):
    path = Path(directory) / f"actor_{iteration:06d}.keras"
    return keras.models.load_model(path)


def load_stdv(directory, iteration):
    return np.load(Path(directory) / f"stdv_{iteration:06d}.npy")


def assign_variables(model, values):
    for variable, value in zip(model.trainable_variables, values):
        variable.assign(np.asarray(value))


def find_latest_iteration(directory):
    iterations = []
    for path in Path(directory).glob("actor_*.keras"):
        iterations.append(int(path.stem.split("_")[-1]))
    if not iterations:
        raise FileNotFoundError(f"no actor_*.keras checkpoints in {directory}")
    return max(iterations)
