from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jp
import jax.random as jr
from mujoco import mjx

trajectory_fields = "controls knots costs trace_sites"
Trajectory = namedtuple("Trajectory", trajectory_fields)
controller_fields = "task model sampler iterations num_control_steps optimize"
Controller = namedtuple("Controller", controller_fields)


def PredictiveSampler(task, model, sampler, iterations):
    num_control_steps = int(round(sampler.plan_horizon / model.time_delta))
    optimize_args = task, model, sampler, num_control_steps, iterations
    optimize_fn = partial(optimize, *optimize_args)
    args = task, model, sampler, iterations, num_control_steps, optimize_fn
    return Controller(*args)


def optimize(task, model, sampler, num_control_steps, iterations, key, state, parameters):  # fmt: skip
    parameters = sampler.warm_start(parameters, state)
    knot_times = parameters.knot_times
    _rollout = partial(rollout, task, model, sampler, num_control_steps, state)

    def step(parameters, key):
        knots = sampler.sample(key, parameters)
        rollouts = _rollout(knot_times, knots)
        return sampler.update(parameters, rollouts), rollouts

    keys = jr.split(key, iterations)
    parameters, rollouts = jax.lax.scan(step, parameters, keys)
    rollouts = jax.tree.map(lambda value: value[-1], rollouts)
    return parameters, rollouts


def rollout(task, model, sampler, num_control_steps, state, knot_times, knots):
    start_time = knot_times[0]
    query_times = start_time + model.time_delta * jp.arange(num_control_steps)
    controls = sampler.interpolate(query_times, knot_times, knots)
    evaluate = jax.vmap(partial(evaluate_sample, task, model, state), (0, 0))
    return evaluate(controls, knots)


def evaluate_sample(task, model, state, controls, knots):

    def step(state, control):
        state = state.replace(ctrl=control)
        state = mjx.step(model.model, state)
        cost = task.running_cost(state, control)
        trace_sites = model.get_trace_positions(state)
        return state, (cost, trace_sites)

    final_state, (costs, trace_sites) = jax.lax.scan(step, state, controls)
    final_cost = task.running_cost(final_state, controls[-1])
    final_sites = model.get_trace_positions(final_state)
    costs = jp.append(costs, final_cost)
    trace_sites = jp.append(trace_sites, final_sites[None], axis=0)
    return Trajectory(controls, knots, costs, trace_sites)
