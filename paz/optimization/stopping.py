import jax.numpy as jp
import optax.tree_utils as otu

MAX_STEPS_REACHED = 0
STOP_FN_MET = 1


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


def _never_stop(_step_arg, _params, _loss, _gradients):
    return jp.array(False)


_never_stop.message = None


def _build_status(has_to_stop):
    return jp.where(has_to_stop, STOP_FN_MET, MAX_STEPS_REACHED)


def _get_stop_message(stop_fn):
    return getattr(stop_fn, "message", "stop=stop_fn")
