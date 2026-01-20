from copy import deepcopy
import torch
import json
from collections import OrderedDict
import math


class ModelEma(torch.nn.Module):
    """EMA Model"""

    def __init__(self, model, decay=0.9997, tau=0, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        self.decay = decay
        self.tau = tau
        self.updates = 1
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _get_decay(self):
        if self.tau == 0:
            decay = self.decay
        else:
            decay = self.decay * (1 - math.exp(-self.updates / self.tau))
        return decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        decay = self._get_decay()
        self._update(model, update_fn=lambda e, m: decay * e + (1.0 - decay) * m)
        self.updates += 1

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class BestMetricSingle:
    def __init__(self, init_res=0.0, better="large") -> None:
        self.init_res = init_res
        self.best_res = init_res
        self.best_ep = -1

        self.better = better
        assert better in ["large", "small"]

    def isbetter(self, new_res, old_res):
        if self.better == "large":
            return new_res > old_res
        if self.better == "small":
            return new_res < old_res

    def update(self, new_res, ep):
        if self.isbetter(new_res, self.best_res):
            self.best_res = new_res
            self.best_ep = ep
            return True
        return False

    def __str__(self) -> str:
        return "best_res: {}\t best_ep: {}".format(self.best_res, self.best_ep)

    def __repr__(self) -> str:
        return self.__str__()

    def summary(self) -> dict:
        return {
            "best_res": self.best_res,
            "best_ep": self.best_ep,
        }


class BestMetricHolder:
    def __init__(self, init_res=0.0, better="large", use_ema=False) -> None:
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
        res.update({f"all_{k}": v for k, v in self.best_all.summary().items()})
        res.update({f"regular_{k}": v for k, v in self.best_regular.summary().items()})
        res.update({f"ema_{k}": v for k, v in self.best_ema.summary().items()})
        return res

    def __repr__(self) -> str:
        return json.dumps(self.summary(), indent=2)

    def __str__(self) -> str:
        return self.__repr__()


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict
