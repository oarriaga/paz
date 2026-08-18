"""Keras actor for the unitree_rl_lab G1 velocity policy.

The checkpoints are rsl_rl ActorCritic saves. Deployment only needs the
actor, so the critic and the optimizer state are dropped here. The layer
sizes come from actor_hidden_dims in the run's params/agent.yaml.
"""

from functools import partial
from pathlib import Path

import jax
import keras
import numpy as np
import torch

# On a GPU, XLA defaults float32 matmuls to TF32, which moves an action
# by a few parts in a thousand against the checkpoint it came from.
jax.config.update("jax_default_matmul_precision", "float32")

NUM_JOINTS = 29
NUM_HISTORY_FRAMES = 5
FRAME_DIM = 96
OBSERVATION_DIM = FRAME_DIM * NUM_HISTORY_FRAMES
HIDDEN_UNITS = (512, 256, 128)


def load_actor(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    weights = checkpoint["model_state_dict"]
    actor = build_actor()
    for index, layer in enumerate(actor.layers[1:]):
        set_layer_weights(layer, weights, 2 * index)
    return actor


def build_actor():
    inputs = keras.Input(shape=(OBSERVATION_DIM,))
    outputs = inputs
    for units in HIDDEN_UNITS:
        outputs = keras.layers.Dense(units, activation="elu")(outputs)
    return keras.Model(inputs, keras.layers.Dense(NUM_JOINTS)(outputs))


def set_layer_weights(layer, weights, position):
    # rsl_rl saves the actor as a Sequential with an ELU between the
    # linear layers, so weights sit at every other position.
    kernel = weights[f"actor.{position}.weight"].numpy().transpose()
    layer.set_weights([kernel, weights[f"actor.{position}.bias"].numpy()])


def compile_actor(actor):
    # Called eagerly, the actor costs milliseconds against a 20 ms
    # control period. Compile and warm it up before the loop starts.
    call = jax.jit(partial(actor, training=False))
    call(np.zeros((1, OBSERVATION_DIM), "float32"))
    return call


def find_latest_checkpoint(experiment_dir):
    paths = sorted(Path(experiment_dir).glob("*/model_*.pt"), key=order_by_run)
    if not paths:
        raise SystemExit(f"No rsl_rl model_*.pt under {experiment_dir}")
    return paths[-1]


def order_by_run(path):
    return path.parent.name, int(path.stem.split("_")[1])
