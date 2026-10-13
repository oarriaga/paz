import keras
import optax

from paz.optimization.linesearch import LineSearch
from paz.optimization.linesearch import armijo_linesearch
from paz.optimization.linesearch import wolfe_linesearch
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


class LayerwiseAdamW(keras.optimizers.AdamW):
    """AdamW with a per-variable learning-rate scale.

    Fine-tuning wants a smaller rate deep inside a pretrained backbone than
    on a fresh head, and Keras keeps a single rate per optimizer. Scales are
    keyed by ``variable.path`` and default to one. Weight decay stays global:
    only the learning rate is scaled.
    """

    def __init__(self, scales, learning_rate, **kwargs):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.scales = dict(scales)

    def update_step(self, gradient, variable, learning_rate):
        scale = self.scales.get(variable.path, 1.0)
        super().update_step(gradient, variable, scale * learning_rate)

    def get_config(self):
        config = super().get_config()
        config["scales"] = self.scales
        return config
