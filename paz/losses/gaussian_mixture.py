import jax
import jax.numpy as jp
from keras.saving import register_keras_serializable
from paz.backend import gaussian_mixture as gm


@register_keras_serializable("gaussian_mixture", "nll")
def gaussian_mixture_nll(y_true, y_pred):
    grid_size = int(round(y_pred.shape[2] ** 0.5))
    grid_means = gm.build_grid_means(grid_size)
    log_prob = compute_log_prob(y_pred, grid_means, y_true)
    return -jp.sum(log_prob, axis=-1)


def compute_log_prob(maps, grid_means, points):
    over_keypoints = jax.vmap(gm.mixture_log_prob, in_axes=(0, None, 0))
    over_batch = jax.vmap(over_keypoints, in_axes=(0, None, 0))
    return over_batch(maps, grid_means, points)
