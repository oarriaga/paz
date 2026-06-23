import jax.numpy as jp
from jax.nn import softmax
from jax.scipy.special import logsumexp

EPSILON = 1e-6
LOG_2PI = jp.log(2.0 * jp.pi)


def normalize_points(points, H, W):
    x = ((points[..., 0] + 0.5) - (W / 2.0)) / (W / 2.0)
    y = ((H - 0.5 - points[..., 1]) - (H / 2.0)) / (H / 2.0)
    return jp.stack([x, y], axis=-1)


def denormalize_points(points, H, W):
    x = jp.clip(points[..., 0], -1, 1) * (W / 2.0) + (W / 2.0) - 0.5
    y = H - 0.5 - (jp.clip(points[..., 1], -1, 1) * (H / 2.0) + (H / 2.0))
    return jp.stack([x, y], axis=-1)


def build_grid_means(grid_size):
    x, y = jp.meshgrid(jp.arange(grid_size), jp.arange(grid_size))
    grid = jp.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
    return normalize_points(jp.asarray(grid, "float32"), grid_size, grid_size)


def unpack(component_map):
    weights = softmax(component_map[:, 0])
    scales = jp.abs(component_map[:, 1]) + EPSILON
    offsets = component_map[:, 2:4]
    return weights, scales, offsets


def component_means(component_map, grid_means):
    return grid_means + component_map[:, 2:4]


def mixture_mean(component_map, grid_means):
    weights, _, _ = unpack(component_map)
    means = component_means(component_map, grid_means)
    return jp.sum(weights[:, None] * means, axis=0)


def gaussian_log_prob(point, means, scales):
    squared = jp.sum((point - means) ** 2, axis=-1)
    return -0.5 * squared / scales**2 - 2.0 * jp.log(scales) - LOG_2PI


def mixture_log_prob(component_map, grid_means, point):
    weights, scales, _ = unpack(component_map)
    means = component_means(component_map, grid_means)
    return logsumexp(jp.log(weights) + gaussian_log_prob(point, means, scales))


def density(component_map, grid_means, eval_points):
    weights, scales, _ = unpack(component_map)
    means = component_means(component_map, grid_means)
    log_prob = gaussian_log_prob(eval_points[:, None, :], means, scales)
    return jp.sum(weights * jp.exp(log_prob), axis=-1)
