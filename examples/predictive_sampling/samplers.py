from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jp
import jax.random as jr

from spline import interpolate_zero

Parameters = namedtuple("Parameters", "knot_times mean")
sampler_fields = "plan_horizon interpolate initialize warm_start sample update"
Sampler = namedtuple("Sampler", sampler_fields)
initialize_fields = "num_knots num_actuators plan_horizon interpolate"
InitializeArgs = namedtuple("InitializeArgs", initialize_fields)


def KnotSampler(model, num_samples, num_knots, plan_horizon, stdv, interpolate):
    sample_shape = (num_samples, num_knots, model.num_actuators)
    noise_scale = compute_noise_scale(stdv, model.u_min, model.u_max)
    sample_args = sample_shape, noise_scale, model.u_min, model.u_max
    sample = partial(sample_knots_with_pin, *sample_args)
    init_values = model, num_knots, plan_horizon, interpolate
    init_args = build_initialize_args(*init_values)
    init = partial(initialize, init_args)
    warm = partial(warm_start, num_knots, plan_horizon, interpolate)
    return Sampler(plan_horizon, interpolate, init, warm, sample, argmin_update)


def MPPISampler(model, num_samples, num_knots, plan_horizon, stdv, temperature, interpolate):  # fmt: skip
    sample_shape = (num_samples, num_knots, model.num_actuators)
    noise_scale = compute_noise_scale(stdv, model.u_min, model.u_max)
    sample_args = sample_shape, noise_scale, model.u_min, model.u_max
    sample = partial(sample_knots, *sample_args)
    init_values = model, num_knots, plan_horizon, interpolate
    init_args = build_initialize_args(*init_values)
    init = partial(initialize, init_args)
    warm = partial(warm_start, num_knots, plan_horizon, interpolate)
    update = partial(softmax_weighted_mean_update, temperature)
    return Sampler(plan_horizon, interpolate, init, warm, sample, update)


def compute_noise_scale(stdv, u_min, u_max):
    finite = jp.isfinite(u_min) & jp.isfinite(u_max)
    control_scale = 0.5 * (u_max - u_min)
    return jp.where(finite, stdv * control_scale, stdv)


def build_initialize_args(model, num_knots, plan_horizon, interpolate):
    values = num_knots, model.num_actuators, plan_horizon, interpolate
    return InitializeArgs(*values)


def sample_knots_with_pin(shape, noise_scale, u_min, u_max, key, parameters):

    def keep_previous_best(knots, previous_best):
        return knots.at[0].set(previous_best)

    noise = jr.normal(key, shape)
    knots = parameters.mean + noise_scale * noise
    knots = keep_previous_best(knots, parameters.mean)
    return jp.clip(knots, u_min, u_max)


def sample_knots(shape, noise_scale, u_min, u_max, key, parameters):
    noise = jr.normal(key, shape)
    knots = parameters.mean + noise_scale * noise
    return jp.clip(knots, u_min, u_max)


def initialize(config, initial_knots=None):
    if initial_knots is None:
        initial_knots = jp.zeros((config.num_knots, config.num_actuators))
    values = config.num_knots, config.plan_horizon, config.interpolate
    knot_times = compute_knot_times(*values)
    return Parameters(knot_times=knot_times, mean=initial_knots)


def warm_start(num_knots, plan_horizon, interpolate, parameters, state):
    knot_times = compute_knot_times(num_knots, plan_horizon, interpolate)
    knot_times = knot_times + state.time
    knots = parameters.mean[None]
    mean = interpolate(knot_times, parameters.knot_times, knots)[0]
    return parameters._replace(knot_times=knot_times, mean=mean)


def compute_knot_times(num_knots, plan_horizon, interpolate):
    if interpolate is interpolate_zero:
        return plan_horizon * jp.arange(num_knots) / num_knots
    return jp.linspace(0.0, plan_horizon, num_knots)


def argmin_update(parameters, rollouts):
    best = jp.argmin(compute_total_costs(rollouts))
    return parameters._replace(mean=rollouts.knots[best])


def softmax_weighted_mean_update(temperature, parameters, rollouts):
    costs = compute_total_costs(rollouts)
    weights = jax.nn.softmax(-costs / temperature, axis=0)
    mean = jp.sum(weights[:, None, None] * rollouts.knots, axis=0)
    return parameters._replace(mean=mean)


def compute_total_costs(rollouts):
    return jp.mean(rollouts.costs, axis=1)
