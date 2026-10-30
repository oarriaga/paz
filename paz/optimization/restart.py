from collections import namedtuple

import jax
import jax.numpy as jp
import optax

RESTART_FIELDS = ["inner_state", "best_loss", "num_waiting", "num_restarts"]
RestartState = namedtuple("RestartState", RESTART_FIELDS)


def restart_on_stall(optimizer, patience=5, min_delta=1e-3):
    """Restarts an optax optimizer when the loss stops improving.

    After `patience` updates without lowering the loss by `min_delta`, the
    inner optimizer state is re-initialized at the current parameters. For
    LBFGS this clears its memory, which recovers from a failed line search
    that would otherwise repeat the same rejected step forever.
    The loss is read from the `value` keyword passed to `update`, as
    `paz.minimize` does.
    """

    def init(params):
        inner_state = optimizer.init(params)
        return RestartState(inner_state, jp.array(jp.inf), 0, 0)

    def update(updates, state, params=None, **extra_args):
        loss = extra_args["value"]
        is_improving = loss < (state.best_loss - min_delta)
        best_loss = jp.where(is_improving, loss, state.best_loss)
        num_waiting = jp.where(is_improving, 0, state.num_waiting + 1)
        is_stalled = num_waiting >= patience
        inner_args = (is_stalled, optimizer.init(params), state.inner_state)
        inner_state = select_state(*inner_args)
        num_waiting = jp.where(is_stalled, 0, num_waiting)
        num_restarts = state.num_restarts + is_stalled.astype(jp.int32)
        update_args = (updates, inner_state, params)
        updates, inner_state = optimizer.update(*update_args, **extra_args)
        counts = (num_waiting, num_restarts)
        return updates, RestartState(inner_state, best_loss, *counts)

    return optax.GradientTransformationExtraArgs(init, update)


def select_state(is_stalled, fresh_state, state):
    def select(fresh, current):
        return jp.where(is_stalled, fresh, current).astype(current.dtype)

    return jax.tree.map(select, fresh_state, state)
