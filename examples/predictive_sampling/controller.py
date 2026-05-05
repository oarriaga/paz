from functools import partial

import jax
import jax.numpy as jp
import jax.random as jr
from mujoco import mjx

from structures import Controller, Trajectory


def PredictiveSampler(task, model, sampler, iterations):
    num_control_steps = int(round(sampler.plan_horizon / model.time_delta))
    optimize_args = task, model, sampler, num_control_steps, iterations
    optimize_fn = partial(optimize, *optimize_args)
    args = task, model, sampler, iterations, num_control_steps, optimize_fn
    return Controller(*args)


def optimize(task, model, sampler, num_control_steps, iterations, key, state, parameters):  # fmt: skip
    parameters = sampler.warm_start(parameters, state)
    knot_times = parameters.knot_times

    def step(parameters, key):
        knots = sampler.sample(key, parameters)
        args = task, model, sampler, num_control_steps
        rollouts = rollout(*args, state, knot_times, knots)
        best = jp.argmin(jp.sum(rollouts.costs, axis=1))
        return parameters._replace(mean=rollouts.knots[best]), rollouts

    keys = jr.split(key, iterations)
    parameters, rollouts = jax.lax.scan(step, parameters, keys)
    rollouts = jax.tree.map(lambda value: value[-1], rollouts)
    return parameters, rollouts


def rollout(task, model, sampler, num_control_steps, state, knot_times, knots):
    start_time, end_time = knot_times[0], knot_times[-1]
    query_times = jp.linspace(start_time, end_time, num_control_steps)
    controls = sampler.interpolate(query_times, knot_times, knots)
    eval_fn = partial(evaluate_sample, task, model)
    return jax.vmap(eval_fn, in_axes=(None, 0, 0))(state, controls, knots)


def evaluate_sample(task, model, state, controls, knots):

    def step(state, control):
        state = state.replace(ctrl=control)
        state = mjx.step(model.model, state)
        cost = model.time_delta * task.running_cost(state, control)
        trace_sites = model.get_trace_positions(state)
        return state, (cost, trace_sites)

    final_state, (costs, trace_sites) = jax.lax.scan(step, state, controls)
    final_cost = task.terminal_cost(final_state)
    final_sites = model.get_trace_positions(final_state)
    costs = jp.append(costs, final_cost)
    trace_sites = jp.append(trace_sites, final_sites[None], axis=0)
    return Trajectory(controls, knots, costs, trace_sites)
