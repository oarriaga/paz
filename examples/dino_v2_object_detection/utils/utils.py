from copy import deepcopy
import json
import math
import keras
import keras.ops as k
import numpy as np

class ModelEma(object):
    """EMA Model for Keras"""
    def __init__(self, model, decay=0.9997, tau=0, device=None):
        # make a copy of the model for accumulating moving average of weights
        # In Keras, we can't easily deepcopy a model if it's functional/subclassed complexly
        # without serialization.
        # Instead, we will store the weights.
        
        self.model_weights = [w.numpy() for w in model.weights]
        self.decay = decay
        self.tau = tau
        self.updates = 1
        # Device handling is implicit in Keras usually

    def _get_decay(self):
        if self.tau == 0:
            decay = self.decay
        else:
            decay = self.decay * (1 - math.exp(-self.updates / self.tau))
        return decay

    def update(self, model):
        decay = self._get_decay()
        
        new_weights = model.get_weights()
        self.model_weights = [
            decay * ema + (1. - decay) * new
            for ema, new in zip(self.model_weights, new_weights)
        ]
        self.updates += 1

    def set(self, model):
        self.model_weights = [w.numpy() for w in model.weights]
        
    def apply_to(self, model):
        """Applies the EMA weights to a model instance"""
        model.set_weights(self.model_weights)


class BestMetricSingle(object):
    def __init__(self, init_res=0.0, better='large'):
        self.init_res = init_res
        self.best_res = init_res
        self.best_ep = -1

        self.better = better
        assert better in ['large', 'small']

    def isbetter(self, new_res, old_res):
        if self.better == 'large':
            return new_res > old_res
        elif self.better == 'small':
            return new_res < old_res
        else:
            raise ValueError(f"Unexpected value for 'better': {self.better!r}")

    def update(self, new_res, ep):
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
    def __init__(self, init_res=0.0, better='large', use_ema=False):
        self.best_all = BestMetricSingle(init_res, better)
        self.use_ema = use_ema
        if use_ema:
            self.best_ema = BestMetricSingle(init_res, better)
            self.best_regular = BestMetricSingle(init_res, better)

    def update(self, new_res, epoch, is_ema=False):
        """
        return if the results is the best.
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
