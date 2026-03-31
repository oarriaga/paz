import optax

from paz.optimization.minimization import minimize
from paz.optimization.stopping import grad_norm_stop
from paz.optimization.history import trim_trace

def LBFGS(parameters, loss_fn, learning_rate, max_steps, tolerance, memory_size, linesearch, metrics=None, callbacks=None):  # fmt: skip
    optimizer = optax.lbfgs(learning_rate, memory_size, True, linesearch)
    stop_fn = grad_norm_stop(tolerance)
    args = (parameters, loss_fn, optimizer, max_steps, stop_fn, metrics, 1, callbacks, True)  # fmt: skip
    _, parameters, history = minimize(*args)
    history = trim_trace(history)
    return parameters, history
