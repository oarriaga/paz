import jax.numpy as jp

from paz.inference.latent import Latent
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


# A variable that exists only when its gate is on. Off, it collapses to a
# point mass at 0.0 whose log-probability is zero, so excluded branches of
# trans-dimensional models drop out of log_prob_sum exactly.
def Gated(distribution_fn, name=None):
    if not callable(distribution_fn):
        raise ValueError(f"Input {distribution_fn} must be a callable")

    def build_gated(gate, *args):
        weights = tfd.Categorical(probs=jp.stack([1.0 - gate, gate]))
        components = [tfd.Deterministic(0.0), distribution_fn(*args)]
        return tfd.Mixture(weights, components)

    return Latent(build_gated, name=name)
