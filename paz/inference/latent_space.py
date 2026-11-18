from collections import namedtuple

import jax.numpy as jp

from paz.backend import bijectors
from paz.inference.types import SampleType


LatentSpace = namedtuple("LatentSpace", ["names", "bijectors", "Sample"])


def build_latent_space(latent_nodes):
    names = [node.name for node in latent_nodes]
    Sample = SampleType(names)
    bijectors = {node.name: _get_latent_bijector(node) for node in latent_nodes}
    return LatentSpace(names, bijectors, Sample)


def _get_latent_bijector(node):
    bijector = node.bijector
    return bijectors.Identity() if bijector is None else bijector


def as_latent_samples(latent_space, samples):
    if isinstance(samples, latent_space.Sample):
        return samples
    if isinstance(samples, dict):
        values = samples
    elif hasattr(samples, "_asdict"):
        values = samples._asdict()
    else:
        raise TypeError("Latent samples must be a dict or namedtuple.")
    missing = [name for name in latent_space.names if name not in values]
    if missing:
        raise ValueError("Missing latent samples: " + ", ".join(missing))
    filtered = {name: values[name] for name in latent_space.names}
    return latent_space.Sample(**filtered)


def as_one_sample(latent_space, samples, shapes):
    samples = as_latent_samples(latent_space, samples)
    for name in latent_space.names:
        shape = jp.shape(getattr(samples, name))
        expected = getattr(shapes, name)
        if shape != expected:
            message = f"'{name}' has shape {shape} but one sample has shape "
            message += f"{expected}. Score one sample per call and use "
            message += "jax.vmap for many."
            raise ValueError(message)
    return samples


def to_forward_samples(latent_space, inverse_samples):
    inverse_samples = as_latent_samples(latent_space, inverse_samples)
    forward = {}
    for name in latent_space.names:
        bijector = latent_space.bijectors[name]
        forward[name] = bijector(getattr(inverse_samples, name))
    return latent_space.Sample(**forward)


def to_inverse_samples(latent_space, forward_samples):
    forward_samples = as_latent_samples(latent_space, forward_samples)
    inverse = {}
    for name in latent_space.names:
        bijector = latent_space.bijectors[name]
        inverse[name] = bijector.inverse(getattr(forward_samples, name))
    return latent_space.Sample(**inverse)
