from typing import NamedTuple

import jax
import optax
import optax.tree_utils as otu


class InfoState(NamedTuple):
    iter_num: int


def print_info():
    def init_fn(params):
        del params
        return InfoState(iter_num=0)

    def update_fn(updates, state, params, *, value, grad, **extra_args):
        del params, extra_args
        jax.debug.print(
            "Iteration: {i}, Value: {v}, Gradient norm: {e}",
            i=state.iter_num,
            v=value,
            e=otu.tree_norm(grad),
        )
        return updates, InfoState(iter_num=state.iter_num + 1)

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


def LBFGS(learning_rate, memory_size, verbose, linesearch):
    optimizer = optax.lbfgs(learning_rate, memory_size, True, linesearch)
    return optax.chain(print_info(), optimizer) if verbose else optimizer
