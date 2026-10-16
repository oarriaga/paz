import functools
import json
import math
from types import SimpleNamespace

MODEL_EMA_METHODS = ("_get_decay", "update", "set", "apply_to")
BEST_SINGLE_METHODS = ("isbetter", "update", "summary", "to_str")
BEST_HOLDER_METHODS = ("update", "summary", "to_str")


# device is unused: it mirrors the torch ModelEma API so ported call sites
# keep working; Keras/JAX place tensors implicitly.
def ModelEma(model, decay=0.9997, tau=0, device=None):
    ns = SimpleNamespace()
    ns.model_weights = {w.path: w.numpy().copy() for w in model.weights}
    ns.decay = decay
    ns.tau = tau
    ns.updates = 1
    functions = (read_ema_decay, update_model_ema, set_model_ema, apply_model_ema)  # fmt: skip
    for name, function in zip(MODEL_EMA_METHODS, functions):
        setattr(ns, name, functools.partial(function, ns))
    return ns


def read_ema_decay(ns):
    decay = ns.decay
    if ns.tau != 0:
        decay = ns.decay * (1 - math.exp(-ns.updates / ns.tau))
    return decay


def update_model_ema(ns, model):
    decay = ns._get_decay()
    for weight in model.weights:
        key = weight.path
        value = weight.numpy()
        if key in ns.model_weights:
            blended = decay * ns.model_weights[key] + (1.0 - decay) * value
            ns.model_weights[key] = blended
        else:
            # Variable appeared after init (e.g. via lazy build)
            ns.model_weights[key] = value.copy()
    ns.updates += 1


def set_model_ema(ns, model):
    ns.model_weights = {w.path: w.numpy().copy() for w in model.weights}


def apply_model_ema(ns, model):
    for weight in model.weights:
        if weight.path in ns.model_weights:
            weight.assign(ns.model_weights[weight.path])


def BestMetricSingle(init_res=0.0, better='large'):
    ns = SimpleNamespace()
    ns.init_res = init_res
    ns.best_res = init_res
    ns.best_ep = -1
    ns.better = better
    assert better in ['large', 'small']
    functions = (is_better_metric, update_best_metric, summarize_best_metric, format_best_metric)  # fmt: skip
    for name, function in zip(BEST_SINGLE_METHODS, functions):
        setattr(ns, name, functools.partial(function, ns))
    return ns


def is_better_metric(ns, new_res, old_res):
    if ns.better not in ('large', 'small'):
        raise ValueError(f"Unexpected value for 'better': {ns.better!r}")
    return new_res > old_res if ns.better == 'large' else new_res < old_res


def update_best_metric(ns, new_res, ep):
    improved = ns.isbetter(new_res, ns.best_res)
    if improved:
        ns.best_res = new_res
        ns.best_ep = ep
    return improved


def summarize_best_metric(ns):
    return {'best_res': ns.best_res, 'best_ep': ns.best_ep}


def format_best_metric(ns):
    return "best_res: {}\t best_ep: {}".format(ns.best_res, ns.best_ep)


def BestMetricHolder(init_res=0.0, better='large', use_ema=False):
    ns = SimpleNamespace()
    ns.best_all = BestMetricSingle(init_res, better)
    ns.use_ema = use_ema
    if use_ema:
        ns.best_ema = BestMetricSingle(init_res, better)
        ns.best_regular = BestMetricSingle(init_res, better)
    functions = (update_best_holder, summarize_best_holder, format_best_holder)
    for name, function in zip(BEST_HOLDER_METHODS, functions):
        setattr(ns, name, functools.partial(function, ns))
    return ns


def update_best_holder(ns, new_res, epoch, is_ema=False):
    if ns.use_ema:
        tracked = ns.best_ema if is_ema else ns.best_regular
        tracked.update(new_res, epoch)
    return ns.best_all.update(new_res, epoch)


def summarize_best_holder(ns):
    summary = ns.best_all.summary()
    if ns.use_ema:
        summary = {f'all_{k}': v for k, v in summary.items()}
        regular = ns.best_regular.summary()
        summary.update({f'regular_{k}': v for k, v in regular.items()})
        ema = ns.best_ema.summary()
        summary.update({f'ema_{k}': v for k, v in ema.items()})
    return summary


def format_best_holder(ns):
    return json.dumps(ns.summary(), indent=2)
