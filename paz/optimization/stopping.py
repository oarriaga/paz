import jax.numpy as jp
import optax.tree_utils as otu

MAX_STEPS_REACHED = 0
STOP_FN_MET = 1


class StatefulStop:
    def __init__(self, init, update, message=None):
        self.init = init
        self.update = update
        self.message = message


def grad_norm_stop(tolerance):
    def stop_fn(_step_arg, _params, _loss, gradients):
        return otu.tree_norm(gradients) < tolerance

    stop_fn.message = "stop=grad_norm"
    return stop_fn


def loss_stop(tolerance):
    def stop_fn(_step_arg, _params, loss, _gradients):
        return loss < tolerance

    stop_fn.message = "stop=loss"
    return stop_fn


def patience_stop(min_delta, patience):
    def init():
        return jp.inf, jp.array(0, dtype=jp.int32)

    def update(state, _step_arg, _params, loss, _gradients):
        best_loss, wait = state
        improved = loss < (best_loss - min_delta)
        best_loss = jp.where(improved, loss, best_loss)
        wait = jp.where(improved, 0, wait + 1)
        return (best_loss, wait), wait >= patience

    return StatefulStop(init, update, "stop=patience")


def _never_stop(_step_arg, _params, _loss, _gradients):
    return jp.array(False)


_never_stop.message = None


def _build_status(has_to_stop):
    return jp.where(has_to_stop, STOP_FN_MET, MAX_STEPS_REACHED)


def _get_stop_message(stop_fn):
    return getattr(stop_fn, "message", "stop=stop_fn")


def _init_stop_state(stop_fn):
    init = getattr(stop_fn, "init", None)
    if init is None:
        return jp.array(False)
    return init()


def _run_stop_fn(stop_fn, state, step_arg, params, loss, gradients):
    update = getattr(stop_fn, "update", None)
    if update is None:
        return state, stop_fn(step_arg, params, loss, gradients)
    return update(state, step_arg, params, loss, gradients)
