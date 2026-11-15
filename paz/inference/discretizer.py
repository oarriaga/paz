import jax.numpy as jp
from paz.backend import distributions
from paz.backend import bijectors


def discretize(base_distribution, min_val, max_val, num_steps):
    step_count = float(num_steps - 1)
    scale = step_count / (max_val - min_val)
    shift = -min_val * scale
    grid_bijector = bijectors.Chain(
        [
            bijectors.Shift(shift=-0.5),
            bijectors.Shift(shift=shift),
            bijectors.Scale(scale=scale),
        ]
    )
    transformed_dist = distributions.TransformedDistribution(
        distribution=base_distribution, bijector=grid_bijector
    )
    quantizer = distributions.QuantizedDistribution(transformed_dist)
    grid_indices = jp.arange(num_steps, dtype=jp.float32)
    unnormalized_logits = quantizer.log_prob(grid_indices)
    unnormalized_logits = jp.where(
        jp.isfinite(unnormalized_logits), unnormalized_logits, -jp.inf
    )
    return distributions.Categorical(logits=unnormalized_logits)


def get_grid_values(min_val, max_val, num_steps):
    return jp.linspace(min_val, max_val, num_steps)


def discretize_to_grid(
    distribution, min_val, max_val, num_steps, log_space=False
):
    """Discretize distribution by evaluating log_prob at grid points."""
    # TODO most likely try to make the API fit to one single function
    if log_space:
        log_min = jp.log(min_val)
        log_max = jp.log(max_val)
        values = jp.exp(jp.linspace(log_min, log_max, num_steps))
    else:
        values = get_grid_values(min_val, max_val, num_steps)
    log_probs = distribution.log_prob(values)
    return values, log_probs


def discretize_circular(
    distribution, num_steps, min_angle=-jp.pi, max_angle=jp.pi
):
    """Discretize circular distribution (VonMises) over angle range."""
    # TODO most likely try to make the API fit to one single function
    angles = jp.linspace(min_angle, max_angle, num_steps, endpoint=False)
    log_probs = distribution.log_prob(angles)
    return angles, log_probs


def indices_to_values(indices, min_val, max_val, num_steps):
    fraction = indices.astype(jp.float32) / (num_steps - 1)
    return min_val + fraction * (max_val - min_val)
