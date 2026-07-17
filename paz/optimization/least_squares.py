from collections import namedtuple

import jax
import jax.numpy as jp

RESULT_FIELDS = ("parameters", "cost", "gradient_norm", "step_norm",
                 "num_iterations", "valid", "cost_trace")
LeastSquaresResult = namedtuple("LeastSquaresResult", RESULT_FIELDS)
DampedLeastSquaresResult = namedtuple(
    "DampedLeastSquaresResult", RESULT_FIELDS + ("damping_trace",))


def gauss_newton(residual_fn, parameters, num_iterations):
    args = (residual_fn, parameters, _add, parameters.size, num_iterations)
    return gauss_newton_on_manifold(*args)


def levenberg_marquardt(residual_fn, parameters, num_iterations,
                        initial_damping):
    args = (residual_fn, parameters, _add, parameters.size)
    args += (num_iterations, initial_damping)
    return levenberg_marquardt_on_manifold(*args)


def gauss_newton_on_manifold(residual_fn, parameters, retract_fn,
                             tangent_size, num_iterations):
    args = (residual_fn, parameters, retract_fn, tangent_size)
    args += (num_iterations, 0.0, _keep_damping)
    result, _ = _solve(*args)
    return result


def levenberg_marquardt_on_manifold(residual_fn, parameters, retract_fn,
                                    tangent_size, num_iterations,
                                    initial_damping):
    args = (residual_fn, parameters, retract_fn, tangent_size)
    args += (num_iterations, initial_damping, _scale_damping)
    result, damping_trace = _solve(*args)
    return DampedLeastSquaresResult(*result, damping_trace)


def solve_normal_equations(JtJ, Jtr, damping):
    floor = 1e-12 * jp.eye(JtJ.shape[0], dtype=JtJ.dtype)
    damped = JtJ + damping * jp.diag(jp.diag(JtJ)) + floor
    return jp.linalg.solve(damped, -Jtr)


def _solve(residual_fn, parameters, retract_fn, tangent_size,
           num_iterations, initial_damping, update_damping):
    def step(iteration, carry):
        parameters, cost, damping, step_norm = carry[:4]
        cost_trace, damping_trace = carry[4:]
        args = (residual_fn, parameters, retract_fn, tangent_size, damping)
        delta, new_parameters, new_cost = _try_step(*args)
        accept = _is_acceptable(delta, new_parameters, new_cost, cost)
        old_state = (parameters, cost, step_norm)
        new_state = (new_parameters, new_cost, jp.linalg.norm(delta))
        parameters, cost, step_norm = _select(accept, old_state, new_state)
        damping_trace = damping_trace.at[iteration].set(damping)
        damping = update_damping(damping, accept)
        cost_trace = cost_trace.at[iteration + 1].set(cost)
        return parameters, cost, damping, step_norm, cost_trace, damping_trace

    cost = _compute_cost(residual_fn(parameters))
    cost_trace = jp.zeros(num_iterations + 1).at[0].set(cost)
    damping = jp.zeros(()) + initial_damping
    carry = (parameters, cost, damping, jp.zeros(()))
    carry += (cost_trace, jp.zeros(num_iterations))
    carry = jax.lax.fori_loop(0, num_iterations, step, carry)
    parameters, cost, _, step_norm, cost_trace, damping_trace = carry
    args = (residual_fn, parameters, retract_fn, tangent_size)
    gradient_norm = _compute_gradient_norm(*args)
    valid = _all_finite(parameters, cost, gradient_norm)
    result_args = (parameters, cost, gradient_norm, step_norm)
    result_args += (num_iterations, valid, cost_trace)
    return LeastSquaresResult(*result_args), damping_trace


def _try_step(residual_fn, parameters, retract_fn, tangent_size, damping):
    residual = residual_fn(parameters)
    args = (residual_fn, parameters, retract_fn, tangent_size)
    jacobian = _compute_jacobian(*args)
    gradient = jacobian.T @ residual
    delta = solve_normal_equations(jacobian.T @ jacobian, gradient, damping)
    new_parameters = retract_fn(parameters, delta)
    new_cost = _compute_cost(residual_fn(new_parameters))
    return delta, new_parameters, new_cost


def _compute_jacobian(residual_fn, parameters, retract_fn, tangent_size):
    def tangent_residual(delta):
        return residual_fn(retract_fn(parameters, delta))

    return jax.jacfwd(tangent_residual)(jp.zeros(tangent_size))


def _compute_gradient_norm(residual_fn, parameters, retract_fn, tangent_size):
    args = (residual_fn, parameters, retract_fn, tangent_size)
    jacobian = _compute_jacobian(*args)
    gradient = jacobian.T @ residual_fn(parameters)
    return jp.max(jp.abs(gradient))


def _is_acceptable(delta, new_parameters, new_cost, cost):
    delta_is_finite = jp.all(jp.isfinite(delta))
    parameters_are_finite = jp.all(jp.isfinite(new_parameters))
    is_finite = delta_is_finite & parameters_are_finite
    return is_finite & jp.isfinite(new_cost) & (new_cost < cost)


def _select(accept, old_state, new_state):
    def keep():
        return old_state

    def update():
        return new_state

    return jax.lax.cond(accept, update, keep)


def _compute_cost(residual):
    return 0.5 * jp.sum(residual**2)


def _all_finite(parameters, cost, gradient_norm):
    values_are_finite = jp.isfinite(cost) & jp.isfinite(gradient_norm)
    return jp.all(jp.isfinite(parameters)) & values_are_finite


def _add(parameters, delta):
    return parameters + delta


def _keep_damping(damping, accept):
    return damping


def _scale_damping(damping, accept):
    scaled = jp.where(accept, damping / 3.0, damping * 3.0)
    return jp.clip(scaled, 1e-9, 1e6)
