"""Deterministic DDIM sampling with separated classifier-free guidance.

Each denoising step runs the denoiser once on a batch of three copies:
text only, constraints only, and unconditional. The guided prediction is
``u + w_text (a - u) + w_constraint (b - u)``, the reference's
``separated`` mode. Weights are passed as jit arguments through
``paz.call_stateless`` so the 1 GB of parameters is never constant-folded.
"""
from collections import namedtuple

import jax
import jax.numpy as jp

import paz
from paz.models.foundation.kimodo import diffusion, features

SamplingConfig = namedtuple(
    "SamplingConfig",
    "num_frames num_steps text_weight constraint_weight heading_angle",
)


def sample(key, denoiser, text, observed_motion, motion_mask, config):
    """Samples normalized ``(B, num_frames, 417)`` features for ``text``."""
    shape = (text.shape[0], config.num_frames, features.FEATURE_DIM)
    noise = jax.random.normal(key, shape, "float32")
    args = (denoiser, text, observed_motion, motion_mask, config)
    return denoise_from_noise(noise, *args)


def denoise_from_noise(noise, denoiser, text, observed_motion, motion_mask,
                       config):
    schedule = diffusion.build_schedule(config.num_steps)
    variables = paz.snapshot_variables(denoiser)
    weights = jp.array([config.text_weight, config.constraint_weight])
    heading = jp.full((noise.shape[0],), config.heading_angle, "float32")
    motion = noise
    for step in reversed(range(config.num_steps)):
        args = (variables, schedule, motion, text, observed_motion,
                motion_mask, heading, weights, step)
        motion = denoise_step(denoiser, *args)
    return motion


def build_empty_constraints(batch_size, num_frames):
    shape = (batch_size, num_frames, features.FEATURE_DIM)
    return jp.zeros(shape, "float32"), jp.zeros(shape, "float32")


def _denoise_step(denoiser, variables, schedule, motion, text, observed,
                  mask, heading, weights, step):
    base_timestep = schedule.base_timesteps[step]
    timestep = jp.full((motion.shape[0],), base_timestep, "int32")
    args = (variables, motion, text, observed, mask, timestep, heading)
    predicted = guide(denoiser, *args, weights)
    return diffusion.ddim_step(motion, predicted, schedule, step)


denoise_step = jax.jit(_denoise_step, static_argnums=0)


def guide(denoiser, variables, motion, text, observed, mask, timestep,
          heading, weights):
    no_text, no_mask = jp.zeros_like(text), jp.zeros_like(mask)
    inputs = [
        jp.tile(motion, (3, 1, 1)),
        jp.concatenate([no_mask, mask, no_mask]),
        jp.tile(observed, (3, 1, 1)),
        jp.concatenate([text, no_text, no_text]),
        jp.tile(timestep, 3),
        jp.tile(heading, 3),
    ]
    outputs = paz.call_stateless(denoiser, variables, inputs)
    with_text, with_constraints, unconditional = jp.split(outputs, 3)
    text_term = weights[0] * (with_text - unconditional)
    constraint_term = weights[1] * (with_constraints - unconditional)
    return unconditional + text_term + constraint_term
