import jax.numpy as jp


def apply_huber(squared_errors, scale):
    is_small = squared_errors <= scale * scale
    safe_squared_errors = jp.where(is_small, 1.0, squared_errors)
    linear = scale * jp.sqrt(safe_squared_errors) - 0.5 * scale * scale
    return jp.where(is_small, 0.5 * squared_errors, linear)


def huber_weights(residual_norms, scale):
    is_small = residual_norms <= scale
    safe_norms = jp.where(is_small, 1.0, residual_norms)
    weights = scale / safe_norms
    return jp.where(is_small, jp.ones_like(weights), weights)


def apply_cauchy(squared_errors, scale):
    scale_squared = scale * scale
    return 0.5 * scale_squared * jp.log1p(squared_errors / scale_squared)


def cauchy_weights(residual_norms, scale):
    return 1.0 / (1.0 + (residual_norms / scale) ** 2)
