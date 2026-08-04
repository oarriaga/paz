"""Keras architecture for the ported GEAR-WBC lower-body controller."""

import keras
from keras import layers

NUM_HISTORY_FRAMES = 6
FRAME_DIM = 86
OBSERVATION_DIM = NUM_HISTORY_FRAMES * FRAME_DIM
VELOCITY_DIM = 3
LATENT_DIM = 32
ACTION_DIM = 15


def build_actor():
    observation = keras.Input((OBSERVATION_DIM,), name="observation")
    estimate = compute_estimate(observation)
    velocity = estimate[:, :VELOCITY_DIM]
    latent = compute_unit_norm(estimate[:, VELOCITY_DIM:])
    current_frame = observation[:, -FRAME_DIM:]
    parts = [current_frame, velocity, latent]
    features = keras.ops.concatenate(parts, axis=-1)
    args = features, (512, 256, 256), ACTION_DIM, "actor"
    action = compute_dense_stack(*args)
    return keras.Model(observation, action, name="gear_wbc_actor")


def compute_estimate(observation):
    # Estimates the base linear velocity and a motion latent that the actor
    # cannot read off the current frame alone.
    output_dim = VELOCITY_DIM + LATENT_DIM
    args = observation, (256, 256), output_dim, "estimator"
    return compute_dense_stack(*args)


def compute_unit_norm(latent):
    # keras.ops.norm traces to an unhashable axis under the JAX backend, so
    # the release's ReduceL2 is spelled out here instead.
    squares = keras.ops.sum(keras.ops.square(latent), axis=-1, keepdims=True)
    norm = keras.ops.sqrt(squares)
    return latent / keras.ops.maximum(norm, 1e-12)


def compute_dense_stack(inputs, hidden_dims, output_dim, prefix):
    x, layer_index = inputs, 0
    for units in hidden_dims:
        x = layers.Dense(units, name=f"{prefix}_{layer_index}")(x)
        x = layers.Activation("elu", name=f"{prefix}_elu_{layer_index}")(x)
        layer_index = layer_index + 2
    return layers.Dense(output_dim, name=f"{prefix}_{layer_index}")(x)
