from collections import namedtuple

import jax
import jax.numpy as jp
import optax

from .bijectors import Chain, Scale, Shift, Sigmoid
from .distributions import Normal, Uniform
from .utils import log_prob_inverse


TrueParameters = namedtuple("TrueParameters", ["mean", "bias", "stdv"])


def build_stdv_bijector(low, high):
    return Chain([Shift(low), Scale(high - low), Sigmoid()])


def generate_data(key, num_points, true_parameters):
    x = jp.linspace(-1.0, 1.0, num_points)
    noise = true_parameters.stdv * jax.random.normal(key, (num_points,))
    y = true_parameters.mean * x + true_parameters.bias + noise
    return x, y


def unpack_parameters(inverse_parameters, stdv_bijector):
    mean = inverse_parameters[0]
    bias = inverse_parameters[1]
    stdv = stdv_bijector(inverse_parameters[2])
    return mean, bias, stdv


def compute_log_joint(inverse_parameters, x, y, stdv_bijector):
    mean, bias, stdv = unpack_parameters(inverse_parameters, stdv_bijector)
    mean_prior = Normal(0.0, 1.0)
    bias_prior = Normal(0.0, 1.0)
    stdv_prior = Uniform(0.05, 1.0)
    log_prob = mean_prior.log_prob(mean) + bias_prior.log_prob(bias)
    log_prob = log_prob + log_prob_inverse(
        stdv_prior, stdv_bijector, inverse_parameters[2]
    )
    likelihood = Normal(mean * x + bias, stdv)
    return log_prob + likelihood.log_prob(y).sum()


def fit_inverse_parameters(
    x, y, stdv_bijector, inverse_parameters, learning_rate=5e-2, steps=400
):
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(inverse_parameters)

    def loss_fn(inverse_parameters):
        return -compute_log_joint(inverse_parameters, x, y, stdv_bijector)

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


true_parameters = TrueParameters(1.75, -0.2, 0.25)
data_key = jax.random.PRNGKey(5)
x, y = generate_data(data_key, 64, true_parameters)
stdv_bijector = build_stdv_bijector(0.05, 1.0)
initial_inverse = jp.array([0.0, 0.0, 0.0])
fitted_inverse, losses = fit_inverse_parameters(
    x, y, stdv_bijector, initial_inverse
)
fitted_parameters = unpack_parameters(fitted_inverse, stdv_bijector)

print("true parameters:", true_parameters)
print("fitted parameters:", tuple(float(value) for value in fitted_parameters))
print("initial loss:", losses[0])
print("final loss:", losses[-1])
