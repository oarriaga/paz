from functools import partial

import jax.numpy as jp
import mujoco
from mujoco import mjx

try:
    from .structures import Task
except ImportError:
    from structures import Task


def build_task(mj_model, trace_sites=(), impl="jax", **functions):
    assert isinstance(mj_model, mujoco.MjModel)
    model = mjx.put_model(mj_model, impl=impl)
    trace_site_ids = build_trace_site_ids(mj_model, trace_sites)
    get_sites = partial(get_trace_sites, trace_site_ids)
    make = partial(make_data, mj_model, model.impl)
    running_cost = functions.get("running_cost", missing_running_cost)
    terminal_cost = functions.get("terminal_cost", missing_terminal_cost)
    args = mj_model, model, compute_min_control(mj_model)
    args += compute_max_control(mj_model), mj_model.opt.timestep
    args += trace_site_ids, running_cost
    args += terminal_cost, make, get_sites
    return Task(*args)


def missing_running_cost(_state, _control):
    raise NotImplementedError


def missing_terminal_cost(_state):
    raise NotImplementedError


def make_data(mj_model, impl, **kwargs):
    return mjx.make_data(mj_model, impl=impl, **kwargs)


def get_trace_sites(trace_site_ids, state):
    if len(trace_site_ids) == 0:
        return jp.zeros((0, 3))
    return state.site_xpos[trace_site_ids]


def compute_min_control(mj_model):
    limited = mj_model.actuator_ctrllimited
    return jp.where(limited, mj_model.actuator_ctrlrange[:, 0], -jp.inf)


def compute_max_control(mj_model):
    limited = mj_model.actuator_ctrllimited
    return jp.where(limited, mj_model.actuator_ctrlrange[:, 1], jp.inf)


def build_trace_site_ids(mj_model, trace_sites):
    site_ids = [mj_model.site(name).id for name in trace_sites]
    return jp.array(site_ids)
