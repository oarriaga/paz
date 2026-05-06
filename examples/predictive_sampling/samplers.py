from functools import partial

import jax
import jax.numpy as jp
import jax.random as jr

from structures import Parameters, Sampler


def KnotSampler(model, num_samples, num_knots, plan_horizon, stdv, interpolate):
    sample_shape = (num_samples, num_knots, model.num_actuators)
    sample_args = sample_shape, stdv, model.u_min, model.u_max
    sample = partial(sample_knots_with_pin, *sample_args)
    init = partial(initialize, num_knots, model.num_actuators, plan_horizon)
    warm = partial(warm_start, num_knots, plan_horizon, interpolate)
    return Sampler(plan_horizon, interpolate, init, warm, sample, argmin_update)


def MPPISampler(model, num_samples, num_knots, plan_horizon, stdv, temperature, interpolate):  # fmt: skip
    sample_shape = (num_samples, num_knots, model.num_actuators)
    sample_args = sample_shape, stdv, model.u_min, model.u_max
    sample = partial(sample_knots, *sample_args)
    init = partial(initialize, num_knots, model.num_actuators, plan_horizon)
    warm = partial(warm_start, num_knots, plan_horizon, interpolate)
    update = partial(softmax_weighted_mean_update, temperature)
    return Sampler(plan_horizon, interpolate, init, warm, sample, update)


def sample_knots_with_pin(shape, stdv, u_min, u_max, key, parameters):

    def keep_previous_best(knots, previous_best):
        return knots.at[0].set(previous_best)

    noise = jr.normal(key, shape)
    knots = parameters.mean + stdv * noise
    knots = keep_previous_best(knots, parameters.mean)
    return jp.clip(knots, u_min, u_max)


def sample_knots(shape, stdv, u_min, u_max, key, parameters):
    noise = jr.normal(key, shape)
    knots = parameters.mean + stdv * noise
    return jp.clip(knots, u_min, u_max)


def initialize(num_knots, num_actuators, plan_horizon, initial_knots=None):
    if initial_knots is None:
        initial_knots = jp.zeros((num_knots, num_actuators))
    knot_times = jp.linspace(0.0, plan_horizon, num_knots)
    return Parameters(knot_times=knot_times, mean=initial_knots)


def warm_start(num_knots, plan_horizon, interpolate, parameters, state):
    knot_times = jp.linspace(0.0, plan_horizon, num_knots) + state.time
    knots = parameters.mean[None]
    mean = interpolate(knot_times, parameters.knot_times, knots)[0]
    return parameters._replace(knot_times=knot_times, mean=mean)


def argmin_update(parameters, rollouts):
    best = jp.argmin(jp.sum(rollouts.costs, axis=1))
    return parameters._replace(mean=rollouts.knots[best])


def softmax_weighted_mean_update(temperature, parameters, rollouts):
    costs = jp.sum(rollouts.costs, axis=1)
    weights = jax.nn.softmax(-costs / temperature, axis=0)
    mean = jp.sum(weights[:, None, None] * rollouts.knots, axis=0)
    return parameters._replace(mean=mean)
