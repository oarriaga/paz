from functools import partial

import jax
import jax.numpy as jp
import jax.random as jr
from mujoco import mjx

try:
    from .structures import Controller
    from .structures import ControllerArgs
    from .structures import Parameters
    from .structures import Trajectory
except ImportError:
    from structures import Controller
    from structures import ControllerArgs
    from structures import Parameters
    from structures import Trajectory


def PredictiveSampling(*args):
    task, num_samples, noise_level, plan_horizon = args[:4]
    interpolate, num_knots, iterations = args[4:]
    if iterations < 1:
        raise ValueError("iterations must be greater than 0.")
    num_control_steps = int(round(plan_horizon / task.time_delta))
    values = num_samples, noise_level, plan_horizon, num_knots
    values += iterations, num_control_steps, interpolate
    controller = ControllerArgs(*values)
    initialize = partial(initialize_parameters, controller, task)
    optimize_function = partial(optimize, controller, task)
    return Controller(task, *values, initialize, optimize_function)


def initialize_parameters(controller, task, initial_knots=None):
    if initial_knots is None:
        mean = build_zero_mean(controller, task)
    else:
        mean = initial_knots
    knot_times = jp.linspace(0.0, controller.plan_horizon, controller.num_knots)
    return Parameters(knot_times=knot_times, mean=mean)


def build_zero_mean(controller, task):
    return jp.zeros((controller.num_knots, task.model.nu))


def optimize(controller, task, state, parameters, key):
    knot_times = jp.linspace(0.0, controller.plan_horizon, controller.num_knots)
    knot_times = knot_times + state.time
    knots = parameters.mean[None]
    mean = controller.interpolate(knot_times, parameters.knot_times, knots)
    parameters = parameters._replace(knot_times=knot_times, mean=mean[0])
    key, keys = build_sample_keys(key, controller.iterations)

    def step(parameters, key):
        knots = sample_knots(controller, task, parameters, key)
        knots = jp.clip(knots, task.u_min, task.u_max)
        rollouts = rollout(controller, task, state, knot_times, knots)
        parameters = update_parameters(parameters, rollouts)
        return parameters, rollouts

    parameters, rollouts = jax.lax.scan(step, parameters, keys)
    rollouts = jax.tree.map(lambda value: value[-1], rollouts)
    return parameters, rollouts, key


def build_sample_keys(key, iterations):

    def step(key, _):
        key, sample_key = jr.split(key)
        key, _domain_key = jr.split(key)
        return key, sample_key

    return jax.lax.scan(step, key, jp.arange(iterations))


def sample_knots(controller, task, parameters, key):
    shape = controller.num_samples, controller.num_knots
    shape += (task.model.nu,)
    noise = jr.normal(key, shape)
    controls = parameters.mean + controller.noise_level * noise
    return controls.at[0].set(parameters.mean)


def rollout(controller, task, state, knot_times, knots):
    states = build_rollout_states(state)
    start_time, end_time = knot_times[0], knot_times[-1]
    num_steps = controller.num_control_steps
    query_times = jp.linspace(start_time, end_time, num_steps)
    controls = controller.interpolate(query_times, knot_times, knots)
    eval_fn = partial(evaluate_rollouts, controller, task)
    eval_rollouts = jax.vmap(eval_fn, in_axes=(0, None, None))
    _, rollouts = eval_rollouts(states, controls, knots)
    return unpack_rollouts(rollouts)


def build_rollout_states(state):
    return jax.tree.map(lambda value: jp.expand_dims(value, axis=0), state)


def evaluate_rollouts(controller, task, state, controls, knots):
    eval_fn = partial(evaluate_rollout, controller, task)
    eval_rollout = jax.vmap(eval_fn, in_axes=(None, 0, 0))
    return eval_rollout(state, controls, knots)


def unpack_rollouts(rollouts):
    data = dict(costs=rollouts.costs[0], controls=rollouts.controls[0])
    data["knots"] = rollouts.knots[0]
    data["trace_sites"] = rollouts.trace_sites[0]
    return rollouts._replace(**data)


def evaluate_rollout(controller, task, state, controls, knots):

    def step(state, control):
        state = state.replace(ctrl=control)
        state = mjx.step(task.model, state)
        cost = task.time_delta * task.running_cost(state, control)
        trace_sites = task.get_trace_sites(state)
        return state, (state, cost, trace_sites)

    scan_result = jax.lax.scan(step, state, controls)
    final_state, (states, costs, trace_sites) = scan_result
    final_cost = task.terminal_cost(final_state)
    final_sites = task.get_trace_sites(final_state)
    costs = jp.append(costs, final_cost)
    trace_sites = jp.append(trace_sites, final_sites[None], axis=0)
    return states, Trajectory(controls, knots, costs, trace_sites)


def update_parameters(parameters, rollouts):
    costs = jp.sum(rollouts.costs, axis=1)
    best_index = jp.argmin(costs)
    mean = rollouts.knots[best_index]
    return parameters._replace(mean=mean)
