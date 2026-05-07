from collections import namedtuple
from functools import partial

import jax.numpy as jp
import mujoco
from mujoco import mjx

dynamics_fields = ["mujoco_model", "model", "u_min", "u_max"]
dynamics_fields += ["num_actuators", "time_delta", "trace_site_ids"]
dynamics_fields += ["make_state", "get_trace_positions"]
Dynamics = namedtuple("Dynamics", dynamics_fields)


def Model(path, trace_sites=(), backend="jax", **state_kwargs):
    mujoco_model = mujoco.MjModel.from_xml_path(str(path))
    model = mjx.put_model(mujoco_model, impl=backend)
    trace_site_ids = build_trace_site_ids(mujoco_model, trace_sites)
    get_positions = partial(get_trace_positions, trace_site_ids)
    make = partial(make_state, mujoco_model, model, **state_kwargs)
    args = mujoco_model, model, min_control(mujoco_model)
    args += max_control(mujoco_model), mujoco_model.nu
    args += mujoco_model.opt.timestep, trace_site_ids, make, get_positions
    return Dynamics(*args)


def make_state(mujoco_model, model, **kwargs):
    state = mjx.make_data(mujoco_model, impl=model.impl, **kwargs)
    return mjx.forward(model, state)


def get_trace_positions(trace_site_ids, state):
    if len(trace_site_ids) == 0:
        return jp.zeros((0, 3))
    return state.site_xpos[trace_site_ids]


def min_control(mujoco_model):
    limited = mujoco_model.actuator_ctrllimited
    return jp.where(limited, mujoco_model.actuator_ctrlrange[:, 0], -jp.inf)


def max_control(mujoco_model):
    limited = mujoco_model.actuator_ctrllimited
    return jp.where(limited, mujoco_model.actuator_ctrlrange[:, 1], jp.inf)


def build_trace_site_ids(mujoco_model, trace_sites):
    site_ids = [mujoco_model.site(name).id for name in trace_sites]
    return jp.array(site_ids)
