"""Fixed-step Euler sampling for the FLOWER rectified flow.

Flow time runs from 1 (pure noise) to 0 (actions); each step subtracts
``velocity / num_steps`` and the final chunk is clamped to [-1, 1], exactly
matching the reference sampler.
"""
import jax.numpy as jp
from jax import random


def sample_noise(key, batch_size, config):
    shape = (batch_size, config.num_actions, config.action_dim)
    return random.normal(key, shape, "float32")


def sample_actions(dit, context_tokens, noise, num_steps):
    context_tokens = jp.asarray(context_tokens)
    noise = jp.asarray(noise)
    context_mask = jp.ones(context_tokens.shape[:2], noise.dtype)
    actions = noise
    delta_time = 1.0 / num_steps
    for step in range(num_steps, 0, -1):
        flow_time = jp.full((noise.shape[0],), step / num_steps, noise.dtype)
        inputs = [context_tokens, context_mask, actions, flow_time]
        velocity = dit(inputs, training=False)
        actions = actions - delta_time * velocity
    return jp.clip(actions, -1.0, 1.0)
