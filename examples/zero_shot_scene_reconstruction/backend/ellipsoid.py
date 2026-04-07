from functools import partial

import jax.numpy as jp
import optax
import paz
from tensorflow_probability.substrates import jax as tfp

from . import plane

tfd = tfp.distributions


def initialize_parameters(observations, num_stdvs):
    x, y, z = paz.pointcloud.split(observations)
    x_mean = x.mean()
    y_mean = y.mean()
    z_mean = z.mean()
    x_axis = x.std() * num_stdvs
    y_axis = y.std() * num_stdvs
    z_axis = z.std() * num_stdvs
    return jp.array([x_mean, y_mean, z_mean, x_axis, y_axis, z_axis])


def compute_surface_equation(x, y, z, parameters):
    x_hat, y_hat, z_hat = parameters[:3]
    a = parameters[3] ** 2
    b = parameters[4] ** 2
    c = parameters[5] ** 2
    zeta_0 = (x - x_hat) ** 2 / a
    zeta_1 = (y - y_hat) ** 2 / b
    zeta_2 = (z - z_hat) ** 2 / c
    return zeta_0 + zeta_1 + zeta_2 - 1.0


def to_log_normal(mean, variance):
    scale = jp.log((variance / mean**2) + 1)
    mu = jp.log(mean) - (scale / 2)
    return mu, jp.sqrt(scale)


def negative_log_likelihood(x_scale, observations, parameters):
    x, y, z = paz.pointcloud.split(observations)
    x_mean, y_mean, z_mean = parameters[:3]

    x_mean_prior = tfd.Laplace(x.mean(), 0.1).log_prob(x_mean)
    y_mean_prior = tfd.Laplace(y.mean(), 0.1).log_prob(y_mean)
    z_mean_prior = tfd.Laplace(z.mean(), 0.1).log_prob(z_mean)
    translation_prior_log_prob = x_mean_prior + y_mean_prior + z_mean_prior

    x_axis, y_axis, z_axis = parameters[3:]
    x_stdv, y_stdv, z_stdv = x.std(), y.std(), z.std()
    log_mean, log_scale = to_log_normal(x_scale * x_stdv, 0.1)
    x_axis_prior = tfd.LogNormal(log_mean, log_scale)
    y_axis_prior = tfd.TruncatedNormal(2.0 * y_stdv, 0.1, 0.0, 3.0 * y_stdv)
    z_axis_prior = tfd.TruncatedNormal(2.0 * z_stdv, 0.1, 0.0, 3.0 * z_stdv)
    x_log_prob = x_axis_prior.log_prob(x_axis)
    y_log_prob = y_axis_prior.log_prob(y_axis)
    z_log_prob = z_axis_prior.log_prob(z_axis)
    axis_prior_log_prob = x_log_prob + y_log_prob + z_log_prob
    zeros_pred = compute_surface_equation(x, y, z, parameters)
    likelihood = tfd.Laplace(zeros_pred, scale=1.0)
    log_likelihood = likelihood.log_prob(jp.zeros_like(zeros_pred)).sum()
    return -(log_likelihood + axis_prior_log_prob + translation_prior_log_prob)


def scale_pointcloud(depth, scale, K):
    scaled_depth = scale * depth
    pointcloud = paz.pointcloud.from_depth(scaled_depth, K)
    return pointcloud, scaled_depth


def transform_scale_back(parameters, scale):
    return parameters / scale


def rescale_points(seed, depth, scale, K, max_d, mask):
    points, _ = scale_pointcloud(depth, scale, K)
    max_depth_scaled = scale * max_d
    scene_points = paz.pointcloud.bound(points, max_depth_scaled)
    H, W = mask.shape[:2]
    plane_args = (seed, scene_points, (H, W), K)
    camera_to_plane = plane.fit_camera_to_plane(*plane_args)
    points = paz.pointcloud.mask(points, mask, max_depth_scaled)
    points = paz.pointcloud.transform(points, camera_to_plane)
    return points


def fit_scene(seed, points, depth, masks, scale, num_stds, K, max_d, x_scale):
    del points
    shifts, scales = [], []
    linesearch = paz.optimization.LineSearch(50, "armijo")
    optimizer = optax.lbfgs(10.0, 10, True, linesearch)
    stop_fn = paz.optimization.grad_norm_stop(1e-2)
    rescale_args = (seed, depth, scale, K, max_d)
    rescale = partial(rescale_points, *rescale_args)
    compute_loss = partial(negative_log_likelihood, x_scale)
    for mask in masks:
        points = rescale(mask=mask)
        parameters = initialize_parameters(points, num_stds)
        loss = partial(compute_loss, points)
        optimization_args = (parameters, loss, optimizer, 150, stop_fn)
        _, parameters, _ = paz.minimize(*optimization_args)
        parameters = transform_scale_back(parameters, scale)
        shifts.append(parameters[:3])
        scales.append(parameters[3:])
    return jp.array(shifts), jp.array(scales)
