import jax
import jax.numpy as jp
import numpy as np

import paz
from paz.models.foundation.kimodo import conditioning
from paz.models.foundation.kimodo import model as denoiser_model
from paz.models.foundation.kimodo import sampling
from paz.models.foundation.kimodo.model_test import TINY, build_unit_stats


def build_setup(num_frames=6, num_steps=3):
    stats = build_unit_stats()
    denoiser = denoiser_model.build(stats, **TINY)
    text = jax.random.normal(jax.random.PRNGKey(1), (2, 1, TINY["text_dim"]))
    empty = conditioning.normalize(conditioning.build_empty(num_frames), stats)
    shape = (2, num_frames, empty.observed.shape[-1])
    constraints = (jp.broadcast_to(empty.observed, shape),
                   jp.broadcast_to(empty.mask, shape))
    config = sampling.SamplingConfig(num_frames, num_steps, 2.0, 2.0, 0.0)
    return denoiser, text, constraints, config


def test_sample_shape_and_determinism():
    denoiser, text, (observed, mask), config = build_setup()
    key = jax.random.PRNGKey(0)
    motion = sampling.sample(key, denoiser, text, observed, mask, config)
    assert motion.shape == (2, 6, 417)
    assert jp.isfinite(motion).all()
    again = sampling.sample(key, denoiser, text, observed, mask, config)
    assert jp.array_equal(motion, again)
    other_key = jax.random.PRNGKey(7)
    other = sampling.sample(other_key, denoiser, text, observed, mask, config)
    assert not jp.allclose(motion, other)


def test_denoise_step_compiles_once_per_shape():
    denoiser, text, (observed, mask), config = build_setup()
    size_before = sampling.denoise_step._cache_size()
    for seed in (0, 1):
        key = jax.random.PRNGKey(seed)
        sampling.sample(key, denoiser, text, observed, mask, config)
    assert sampling.denoise_step._cache_size() == size_before + 1


def test_guidance_with_unit_text_weight_is_the_conditional_pass():
    denoiser, text, (observed, mask), config = build_setup()
    variables = paz.snapshot_variables(denoiser)
    motion = jax.random.normal(jax.random.PRNGKey(2), (2, 6, 417))
    timestep = jp.array([5, 5], "int32")
    heading = jp.zeros((2,), "float32")
    args = (variables, motion, text, observed, mask, timestep, heading)
    guided = sampling.guide(denoiser, *args, jp.array([1.0, 0.0]))
    inputs = [motion, mask, observed, text, timestep, heading]
    conditional = np.asarray(denoiser(inputs))
    assert np.allclose(np.asarray(guided), conditional, atol=1e-5)
