import jax
import jax.numpy as jp
import optax

import paz
from paz.optimization.restart import restart_on_stall


def run_updates(optimizer, losses, params):
    state = optimizer.init(params)
    gradients = jp.ones_like(params)
    for loss in losses:
        args = (gradients, state, params)
        _, state = optimizer.update(*args, value=jp.array(loss))
    return state


def test_restart_resets_inner_state_after_patience_stalled_steps():
    optimizer = restart_on_stall(optax.adam(0.1), patience=3, min_delta=0.0)
    losses = [1.0, 1.0, 1.0, 1.0]
    state = run_updates(optimizer, losses, jp.zeros(2))
    # the first loss improves on inf, then three equal losses stall it
    assert int(state.num_restarts) == 1
    # adam counted only the update made right after the reset
    assert int(state.inner_state[0].count) == 1


def test_restart_does_not_trigger_while_improving():
    optimizer = restart_on_stall(optax.adam(0.1), patience=2, min_delta=0.0)
    losses = [5.0, 4.0, 3.0, 2.0, 1.0]
    state = run_updates(optimizer, losses, jp.zeros(2))
    assert int(state.num_restarts) == 0
    assert int(state.inner_state[0].count) == len(losses)


def test_restart_respects_min_delta():
    optimizer = restart_on_stall(optax.adam(0.1), patience=2, min_delta=1.0)
    losses = [5.0, 4.5, 4.2]
    state = run_updates(optimizer, losses, jp.zeros(2))
    # improvements of 0.5 and 0.3 are below min_delta, so it restarts once
    assert int(state.num_restarts) == 1
    assert jp.isclose(state.best_loss, 5.0)


def test_restart_minimizes_quadratic_with_lbfgs():
    def loss_fn(params):
        return jp.sum((params - jp.array([3.0, -2.0])) ** 2)

    linesearch = paz.optimization.wolfe_linesearch(30)
    optimizer = restart_on_stall(optax.lbfgs(linesearch=linesearch))
    args = (jp.zeros(2), loss_fn, optimizer, 50)
    _, params, _ = paz.optimization.minimize(*args)
    assert jp.allclose(params, jp.array([3.0, -2.0]), atol=1e-4)


def test_restart_works_inside_jit():
    def loss_fn(params):
        return jp.sum(jp.cos(params) + (0.1 * params**2))

    optimizer = restart_on_stall(optax.lbfgs(), patience=2)

    @jax.jit
    def minimize(params):
        args = (params, loss_fn, optimizer, 100)
        return paz.optimization.minimize(*args)

    _, params, history = minimize(jp.array([0.5, -0.5]))
    assert jp.all(jp.isfinite(params))
    assert loss_fn(params) < loss_fn(jp.array([0.5, -0.5]))
