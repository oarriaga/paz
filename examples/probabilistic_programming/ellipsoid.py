from collections import namedtuple
import jax
import jax.numpy as jp
import numpy as np
import trimesh
from tensorflow_probability.substrates import jax as tfp

import paz
from paz.inference.latent_space import to_forward_samples, to_inverse_samples
from optimizers import optimize

tfd = tfp.distributions
tfb = tfp.bijectors

PointcloudStatistics = namedtuple(
    "PointcloudStatistics",
    ["x_mean", "y_mean", "z_mean", "x_stdv", "y_stdv", "z_stdv"],
)


def fit(model, pointcloud, num_stdvs, loss_fn, optimizer, max_steps=150, tolerance=1e-2):  # fmt: skip
    statistics = pointcloud_compute_statistics(pointcloud)
    initial = initialize_unconstrained(model, statistics, num_stdvs)
    result = optimize(initial, loss_fn, optimizer, max_steps, tolerance)
    forward = to_forward_samples(model.latent_space, result.parameters)
    center = jp.array([forward.x_mean, forward.y_mean, forward.z_mean])
    axes = jp.array([forward.x_axis, forward.y_axis, forward.z_axis])
    return center, axes, result


def RobustEllipsoid(pointcloud, statistics):
    mean_priors = build_mean_priors(statistics)
    x_axis = build_bounded_axis_prior(statistics.x_stdv, "x_axis")
    y_axis = build_bounded_axis_prior(statistics.y_stdv, "y_axis")
    z_axis = build_bounded_axis_prior(statistics.z_stdv, "z_axis")
    priors = mean_priors + [x_axis, y_axis, z_axis]
    surface_likelihood = build_surface_likelihood(pointcloud)
    surface = paz.Observable(surface_likelihood, name="surface")(*priors)
    model = paz.PGM(priors, [surface], "ellipsoid")
    return model, {"surface": jp.zeros(len(pointcloud))}


def build_mean_priors(statistics):
    names = ["x_mean", "y_mean", "z_mean"]
    means = [statistics.x_mean, statistics.y_mean, statistics.z_mean]
    stdvs = [statistics.x_stdv, statistics.y_stdv, statistics.z_stdv]
    # center should not wander multiple cloud-stds without strong evidence
    scales = [0.10 * s for s in stdvs]  # try 0.25*s if too tight
    iterator = zip(means, scales, names)
    return [paz.Prior(tfd.Laplace(m, sc), name=n) for m, sc, n in iterator]


# def build_bounded_axis_prior(stdv, name):
#     upper = 3.0 * stdv
#     bijector = tfb.Chain([tfb.Shift(0.0), tfb.Scale(upper), tfb.Sigmoid()])
#     distribution = tfd.TruncatedNormal(2.0 * stdv, 0.1, 0.0, upper)
#     return paz.Prior(distribution, bijector=bijector, name=name)


def build_bounded_axis_prior(axis_standard_deviation, axis_name):
    """
    Scale-aware bounded prior for an ellipsoid axis length using a TruncatedNormal.

    Motivation:
      - Hard-coding `scale=0.1` is unit-dependent and will be too tight/loose depending on the pointcloud scale.
      - Axis uncertainty is better expressed *relative* to its expected magnitude.

    Construction:
      1) Expected axis length (prior mean):
           axis_mean = axis_mean_multiplier * axis_standard_deviation

         A reasonable default is axis_mean_multiplier ≈ sqrt(3) for roughly uniform surface sampling of a sphere,
         but we use 1.8 as a practical slightly-conservative value.

      2) Prior spread (standard deviation) set by a coefficient of variation:
           axis_scale = axis_coefficient_of_variation * axis_mean

      3) Hard bounds set relative to the mean:
           axis_low  = epsilon_relative_to_mean * axis_mean
           axis_high = axis_upper_factor * axis_mean

    Defaults (visible axes):
      - axis_mean_multiplier = 1.8
      - axis_coefficient_of_variation = 0.20
      - axis_upper_factor = 3.0

    If you want an occluded/back axis to be more permissive, override it in your
    `build_back_axis_prior(...)` by using a larger coefficient_of_variation and upper_factor.
    """
    axis_mean_multiplier = 1.8
    axis_coefficient_of_variation = 0.05
    axis_upper_factor = 2.0
    epsilon_relative_to_mean = 1e-6

    axis_mean = axis_mean_multiplier * axis_standard_deviation
    axis_scale = axis_coefficient_of_variation * axis_mean

    axis_low = epsilon_relative_to_mean * axis_mean
    axis_high = axis_upper_factor * axis_mean

    truncated_normal_distribution = tfd.TruncatedNormal( loc=axis_mean, scale=axis_scale, low=axis_low, high=axis_high)  # fmt: skip
    return paz.Prior(truncated_normal_distribution, name=axis_name)


def build_back_axis_prior(stdv, name):
    upper = 6.0 * stdv
    bijector = tfb.Chain([tfb.Shift(0.0), tfb.Scale(upper), tfb.Sigmoid()])
    distribution = tfd.TruncatedNormal(2.0 * stdv, 1.0, 0.0, upper)
    return paz.Prior(distribution, bijector=bijector, name=name)


def build_map_objective(model, data):
    def negative_log_posterior(inverse_samples):
        log_prior = model.prior.log_prob_inverse(inverse_samples).log_prob_sum
        likelihood = model.likelihood.log_prob_inverse(inverse_samples, data)
        return -(log_prior + likelihood.log_prob_sum)

    return negative_log_posterior


def build_surface_likelihood(observations):
    x, y, z = paz.pointcloud.split(observations)

    def distribution_fn(x_mean, y_mean, z_mean, x_axis, y_axis, z_axis):
        args = (x, y, z, x_mean, y_mean, z_mean, x_axis, y_axis, z_axis)
        residuals = compute_surface_equation(*args)
        return tfd.Laplace(residuals, scale=0.01)

    return distribution_fn


def verify_prior_predictive(model, key, num_samples=1000):
    forward_samples = model.prior.sample(key, num_samples)
    for name in model.latent_space.names:
        values = getattr(forward_samples, name)
        print(
            f"{name}: mean={float(values.mean()):.4f}, "
            f"stdv={float(values.std()):.4f}, "
            f"min={float(values.min()):.4f}, "
            f"max={float(values.max()):.4f}"
        )
    key_rt, _ = jax.random.split(key)
    inverse_samples = model.prior.sample_inverse(key_rt, num_samples)
    forward_from_inverse = to_forward_samples(
        model.latent_space, inverse_samples
    )
    for name in model.latent_space.names:
        forward_val = getattr(forward_from_inverse, name)
        inverse_val = getattr(inverse_samples, name)
        recovered = model.latent_space.bijectors[name].inverse(forward_val)
        is_close = jp.allclose(inverse_val, recovered, atol=1e-5)
        print(f"{name} bijector round-trip: {'OK' if is_close else 'FAILED'}")
    return forward_samples


def initialize_unconstrained(model, statistics, num_stdvs):
    forward_sample = model.latent_space.Sample(
        x_mean=statistics.x_mean,
        y_mean=statistics.y_mean,
        z_mean=statistics.z_mean,
        x_axis=statistics.x_stdv * num_stdvs,
        y_axis=statistics.y_stdv * num_stdvs,
        z_axis=statistics.z_stdv * num_stdvs,
    )
    return to_inverse_samples(model.latent_space, forward_sample)


def compute_surface_equation(
    x, y, z, x_mean, y_mean, z_mean, x_axis, y_axis, z_axis
):
    zeta_0 = (x - x_mean) ** 2 / x_axis**2
    zeta_1 = (y - y_mean) ** 2 / y_axis**2
    zeta_2 = (z - z_mean) ** 2 / z_axis**2
    return zeta_0 + zeta_1 + zeta_2 - 1.0


def to_log_normal(mean, variance):
    scale = jp.log((variance / mean**2) + 1)
    mu = jp.log(mean) - (scale / 2)
    return mu, jp.sqrt(scale)


def pointcloud_compute_statistics(observations):
    x, y, z = paz.pointcloud.split(observations)
    args = x.mean(), y.mean(), z.mean(), x.std(), y.std(), z.std()
    return PointcloudStatistics(*args)


def build_vertices(a, b, c, u_segments, v_segments):
    vertices = []
    for u_segment_arg in range(u_segments + 1):
        theta = u_segment_arg * (np.pi / u_segments)
        for v_segment_arg in range(v_segments + 1):
            phi = v_segment_arg * 2 * np.pi / v_segments
            x = a * np.sin(theta) * np.cos(phi)
            y = b * np.sin(theta) * np.sin(phi)
            z = c * np.cos(theta)
            vertices.append([x, y, z])
    return vertices


def build_faces(u_segments, v_segments):
    faces = []
    for u_segment_arg in range(u_segments):
        for v_segment_arg in range(v_segments):
            v1 = u_segment_arg * (v_segments + 1) + v_segment_arg
            v2 = (u_segment_arg + 1) * (v_segments + 1) + v_segment_arg
            v3 = (u_segment_arg + 1) * (v_segments + 1) + (v_segment_arg + 1)
            v4 = u_segment_arg * (v_segments + 1) + (v_segment_arg + 1)
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v4])
    return faces


def Mesh(a, b, c, u_segments=100, v_segments=100):
    vertices = build_vertices(a, b, c, u_segments, v_segments)
    faces = build_faces(u_segments, v_segments)
    return trimesh.Trimesh(vertices=vertices, faces=faces)
