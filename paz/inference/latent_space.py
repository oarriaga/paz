from collections import namedtuple

from tensorflow_probability.substrates import jax as tfp

from paz.inference.metadata import get_bijector
from paz.inference.types import SampleType

tfb = tfp.bijectors

LatentSpace = namedtuple("LatentSpace", ["names", "bijectors", "Sample"])


def build_latent_space(latent_nodes):
    names = [node.name for node in latent_nodes]
    Sample = SampleType(names)
    bijectors = {node.name: _get_latent_bijector(node) for node in latent_nodes}
    return LatentSpace(names, bijectors, Sample)


def _get_latent_bijector(node):
    bijector = get_bijector(node)
    return tfb.Identity() if bijector is None else bijector


def as_latent_samples(latent_space, samples):
    if isinstance(samples, dict):
        return latent_space.Sample(**samples)
    if hasattr(samples, "_asdict"):
        return latent_space.Sample(**samples._asdict())
    raise TypeError("Latent samples must be a dict or namedtuple.")


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
