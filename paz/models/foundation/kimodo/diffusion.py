"""Cosine-schedule diffusion with the respaced DDIM step Kimodo samples with.

The schedule arithmetic mirrors the reference in float32, including its
respacing over the full base-step range, so sampled motions match.
"""
import math
from collections import namedtuple

import jax.numpy as jp
import numpy as np

NUM_BASE_STEPS = 1000
Schedule = namedtuple(
    "Schedule",
    "base_timesteps sqrt_recip_alphas sqrt_recipm1_alphas alphas_prev",
)


def build_schedule(num_steps, num_base_steps=NUM_BASE_STEPS):
    """Respaces the base schedule onto ``num_steps`` denoising steps."""
    base_alphas = np.cumprod(1.0 - build_cosine_betas(num_base_steps))
    base_timesteps = space_timesteps(num_steps, num_base_steps)
    alphas = base_alphas[base_timesteps]
    previous = np.concatenate([[np.float32(1.0)], alphas[:-1]])
    alphas = np.cumprod(1.0 - (1.0 - alphas / previous)).astype(np.float32)
    alphas = np.maximum(alphas, np.float32(1e-9))[:num_steps]
    alphas_prev = np.concatenate([[np.float32(1.0)], alphas[:-1]])
    sqrt_recip = 1.0 / np.sqrt(alphas)
    sqrt_recipm1 = 1.0 / np.sqrt(alphas / (1.0 - alphas))
    args = (base_timesteps[:num_steps], sqrt_recip, sqrt_recipm1, alphas_prev)
    return Schedule(*[jp.asarray(value) for value in args])


def build_cosine_betas(num_base_steps, max_beta=0.999):
    betas = []
    for step in range(num_base_steps):
        start = compute_alpha_bar(step / num_base_steps)
        end = compute_alpha_bar((step + 1) / num_base_steps)
        betas.append(min(1.0 - end / start, max_beta))
    return np.asarray(betas, np.float32)


def compute_alpha_bar(time):
    return math.cos((time + 0.008) / 1.008 * math.pi / 2) ** 2


def space_timesteps(num_steps, num_base_steps):
    # The reference spaces all base steps, not only the first num_steps.
    stride = (num_base_steps - 1) / max(1, num_steps - 1)
    scaled = np.arange(num_base_steps, dtype=np.float32) * np.float32(stride)
    return np.minimum(np.round(scaled), num_base_steps - 1).astype(np.int64)


def ddim_step(noisy, predicted_clean, schedule, step):
    """Deterministic DDIM update from step ``step`` to ``step - 1``."""
    noise = noisy * schedule.sqrt_recip_alphas[step] - predicted_clean
    noise = noise / schedule.sqrt_recipm1_alphas[step]
    alpha_prev = schedule.alphas_prev[step]
    clean_term = predicted_clean * jp.sqrt(alpha_prev)
    return clean_term + jp.sqrt(1 - alpha_prev) * noise
