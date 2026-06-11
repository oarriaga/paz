from collections import namedtuple

import jax
import jax.numpy as jp
import optax

from .bijectors import Chain, Scale, Shift, Sigmoid
from .distributions import Independent, Normal, Uniform
from .utils import log_prob_inverse

TrueParameters = namedtuple(
    "TrueParameters",
    [
        "mu_slope",
        "sigma_slope",
        "mu_intercept",
        "sigma_intercept",
        "sigma_obs",
    ],
)


def build_sigma_bijector(low, high):
    return Chain([Shift(low), Scale(high - low), Sigmoid()])


def generate_data(key, num_groups, num_points, parameters):
    key_0, key_1, key_2 = jax.random.split(key, 3)
    slopes = parameters.mu_slope
    slopes = slopes + parameters.sigma_slope * jax.random.normal(
        key_0, (num_groups,)
    )
    intercepts = parameters.mu_intercept
    intercepts = intercepts + parameters.sigma_intercept * jax.random.normal(
        key_1, (num_groups,)
    )
    kwargs = dict(minval=0.0, maxval=1.0)
    x = jax.random.uniform(key_2, (num_groups, num_points), **kwargs)
    x = x.reshape(-1)
    group_idx = jp.repeat(jp.arange(num_groups), num_points)
    noise_key = jax.random.PRNGKey(23)
    noise = parameters.sigma_obs * jax.random.normal(noise_key, x.shape)
    means = slopes[group_idx] * x + intercepts[group_idx]
    y = means + noise
    return x, y, group_idx


def unpack_parameters(inverse_parameters, num_groups, sigma_bijector):
    mu_slope = inverse_parameters[0]
    mu_intercept = inverse_parameters[1]
    sigma_slope = sigma_bijector(inverse_parameters[2])
    sigma_intercept = sigma_bijector(inverse_parameters[3])
    z_slopes = inverse_parameters[4 : 4 + num_groups]
    z_intercepts = inverse_parameters[4 + num_groups : 4 + 2 * num_groups]
    sigma_obs = sigma_bijector(inverse_parameters[-1])
    slopes = mu_slope + sigma_slope * z_slopes
    intercepts = mu_intercept + sigma_intercept * z_intercepts
    return slopes, intercepts, sigma_slope, sigma_intercept, sigma_obs


def compute_log_joint(
    inverse_parameters, x, y, group_idx, num_groups, sigma_bijector
):
    unpacked = unpack_parameters(inverse_parameters, num_groups, sigma_bijector)
    slopes, intercepts, sigma_slope, sigma_intercept, sigma_obs = unpacked
    normal = Normal(0.0, 1.0)
    group_prior = Independent(Normal(jp.zeros(num_groups), 1.0), 1)
    sigma_prior = Uniform(0.05, 1.0)
    log_prob = normal.log_prob(inverse_parameters[0])
    log_prob = log_prob + normal.log_prob(inverse_parameters[1])
    log_prob = log_prob + log_prob_inverse(
        sigma_prior, sigma_bijector, inverse_parameters[2]
    )
    log_prob = log_prob + log_prob_inverse(
        sigma_prior, sigma_bijector, inverse_parameters[3]
    )
    log_prob = log_prob + group_prior.log_prob(
        inverse_parameters[4 : 4 + num_groups]
    )
    log_prob = log_prob + group_prior.log_prob(
        inverse_parameters[4 + num_groups : 4 + 2 * num_groups]
    )
    log_prob = log_prob + log_prob_inverse(
        sigma_prior, sigma_bijector, inverse_parameters[-1]
    )
    means = slopes[group_idx] * x + intercepts[group_idx]
    likelihood = Normal(means, sigma_obs)
    return log_prob + likelihood.log_prob(y).sum()


def fit_inverse_parameters(
    x,
    y,
    group_idx,
    num_groups,
    sigma_bijector,
    inverse_parameters,
    learning_rate=3e-2,
    steps=500,
):
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(inverse_parameters)

    def loss_fn(inverse_parameters):
        return -compute_log_joint(
            inverse_parameters,
            x,
            y,
            group_idx,
            num_groups,
            sigma_bijector,
        )

    @jax.jit
    def update(inverse_parameters, optimizer_state):
        loss, gradients = jax.value_and_grad(loss_fn)(inverse_parameters)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, inverse_parameters
        )
        inverse_parameters = optax.apply_updates(inverse_parameters, updates)
        return inverse_parameters, optimizer_state, loss

    losses = []
    for _ in range(steps):
        args = inverse_parameters, optimizer_state
        inverse_parameters, optimizer_state, loss = update(*args)
        losses.append(float(loss))
    return inverse_parameters, losses


num_groups = 4
num_points = 20
sigma_bijector = build_sigma_bijector(0.05, 1.0)
true_parameters = TrueParameters(1.0, 0.3, -0.5, 0.2, 0.15)
data_key = jax.random.PRNGKey(11)
x, y, group_idx = generate_data(
    data_key, num_groups, num_points, true_parameters
)
num_inverse = 5 + 2 * num_groups
initial_inverse = jp.zeros((num_inverse,))
fitted_inverse, losses = fit_inverse_parameters(
    x, y, group_idx, num_groups, sigma_bijector, initial_inverse
)
fitted = unpack_parameters(fitted_inverse, num_groups, sigma_bijector)

print("true parameters:", true_parameters)
print("fitted sigma slope:", float(fitted[2]))
print("fitted sigma intercept:", float(fitted[3]))
print("fitted sigma obs:", float(fitted[4]))
print("first fitted slopes:", fitted[0][:2])
print("first fitted intercepts:", fitted[1][:2])
print("initial loss:", losses[0])
print("final loss:", losses[-1])
