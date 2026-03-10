from copy import deepcopy
import json
import math
import keras
import keras.ops as k
import numpy as np

class ModelEma(object):
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of each weight as a NumPy array keyed by
    variable path.  After each training step ``update()`` blends the
    current model weights into the running average.  The EMA weights
    can later be applied to any compatible model via ``apply_to()``.

    Attributes:
        model_weights (dict): ``{variable_path: np.ndarray}`` shadow
            weights.
        decay (float): EMA decay factor.
        tau (float): Optional time-constant for a warm-up ramp on the
            decay.  Set to ``0`` to disable.
        updates (int): Number of ``update()`` calls so far.
    """
    def __init__(self, model, decay=0.9997, tau=0, device=None):
        self.model_weights = {
            w.path: w.numpy().copy() for w in model.weights
        }
        self.decay = decay
        self.tau = tau
        self.updates = 1

    def _get_decay(self):
        """Return the effective decay, optionally warmed up via *tau*."""
        if self.tau == 0:
            decay = self.decay
        else:
            decay = self.decay * (1 - math.exp(-self.updates / self.tau))
        return decay

    def update(self, model):
        """Blend current model weights into the EMA shadow copy.

        Args:
            model: Keras model whose weights are blended in.
        """
        decay = self._get_decay()
        for w in model.weights:
            key = w.path
            new = w.numpy()
            if key in self.model_weights:
                self.model_weights[key] = (
                    decay * self.model_weights[key] + (1. - decay) * new
                )
            else:
                # Variable appeared after init (e.g. via lazy build)
                self.model_weights[key] = new.copy()
        self.updates += 1

    def set(self, model):
        """Replace the EMA shadow with the model's current weights."""
        self.model_weights = {
            w.path: w.numpy().copy() for w in model.weights
        }

    def apply_to(self, model):
        """Applies the EMA weights to a model instance."""
        for w in model.weights:
            key = w.path
            if key in self.model_weights:
                w.assign(self.model_weights[key])


class BestMetricSingle(object):
    """Track the single best metric value and the epoch it was achieved.

    Attributes:
        best_res (float): Best metric value observed.
        best_ep (int): Epoch at which *best_res* was recorded.
        better (str): ``'large'`` if higher is better, ``'small'`` if
            lower is better.
    """
    def __init__(self, init_res=0.0, better='large'):
        self.init_res = init_res
        self.best_res = init_res
        self.best_ep = -1

        self.better = better
        assert better in ['large', 'small']

    def isbetter(self, new_res, old_res):
        """Return ``True`` if *new_res* improves upon *old_res*."""
        if self.better == 'large':
            return new_res > old_res
        elif self.better == 'small':
            return new_res < old_res
        else:
            raise ValueError(f"Unexpected value for 'better': {self.better!r}")

    def update(self, new_res, ep):
        """Update the best metric if *new_res* is an improvement.

        Returns:
            bool: ``True`` if the best metric was updated.
        """
        if self.isbetter(new_res, self.best_res):
            self.best_res = new_res
            self.best_ep = ep
            return True
        return False

    def __str__(self):
        return "best_res: {}\t best_ep: {}".format(self.best_res, self.best_ep)

    def __repr__(self):
        return self.__str__()

    def summary(self):
        return {
            'best_res': self.best_res,
            'best_ep': self.best_ep,
        }


class BestMetricHolder(object):
    """Track the overall, regular, and EMA best metrics.

    When EMA is enabled, independently records the best metric from
    the regular model and the EMA model alongside the global best.

    Attributes:
        best_all (BestMetricSingle): Global best across both sources.
        best_ema (BestMetricSingle): Best from EMA evaluations only.
        best_regular (BestMetricSingle): Best from regular evaluations.
        use_ema (bool): Whether EMA tracking is active.
    """
    def __init__(self, init_res=0.0, better='large', use_ema=False):
        self.best_all = BestMetricSingle(init_res, better)
        self.use_ema = use_ema
        if use_ema:
            self.best_ema = BestMetricSingle(init_res, better)
            self.best_regular = BestMetricSingle(init_res, better)

    def update(self, new_res, epoch, is_ema=False):
        """Update the best metric trackers.

        Args:
            new_res (float): New metric value.
            epoch (int): Current epoch number.
            is_ema (bool): Whether *new_res* comes from the EMA model.

        Returns:
            bool: ``True`` if the global best was updated.
        """
        if not self.use_ema:
            return self.best_all.update(new_res, epoch)
        else:
            if is_ema:
                self.best_ema.update(new_res, epoch)
                return self.best_all.update(new_res, epoch)
            else:
                self.best_regular.update(new_res, epoch)
                return self.best_all.update(new_res, epoch)

    def summary(self):
        if not self.use_ema:
            return self.best_all.summary()

        res = {}
        res.update({f'all_{k}':v for k,v in self.best_all.summary().items()})
        res.update({f'regular_{k}':v for k,v in self.best_regular.summary().items()})
        res.update({f'ema_{k}':v for k,v in self.best_ema.summary().items()})
        return res

    def __repr__(self):
        return json.dumps(self.summary(), indent=2)

    def __str__(self):
        return self.__repr__()
