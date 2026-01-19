import jax.numpy as jp
from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions
tfb = tfp.bijectors


def discretize(base_distribution, min_val, max_val, num_steps):
    step_count = float(num_steps - 1)
    scale = step_count / (max_val - min_val)
    shift = -min_val * scale
    grid_bijector = tfb.Chain(
        [
            tfb.Shift(shift=-0.5),
            tfb.Shift(shift=shift),
            tfb.Scale(scale=scale),
        ]
    )
    transformed_dist = tfd.TransformedDistribution(
        distribution=base_distribution, bijector=grid_bijector
    )
    quantizer = tfd.QuantizedDistribution(distribution=transformed_dist)
    grid_indices = jp.arange(num_steps, dtype=jp.float32)
    unnormalized_logits = quantizer.log_prob(grid_indices)
    unnormalized_logits = jp.where(
        jp.isfinite(unnormalized_logits), unnormalized_logits, -jp.inf
    )
    return tfd.Categorical(logits=unnormalized_logits)


def get_grid_values(min_val, max_val, num_steps):
    return jp.linspace(min_val, max_val, num_steps)


def indices_to_values(indices, min_val, max_val, num_steps):
    fraction = indices.astype(jp.float32) / (num_steps - 1)
    return min_val + fraction * (max_val - min_val)
